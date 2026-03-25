"""
train.py — Minimal SAM 2 LoRA fine-tuning for tillage residue segmentation
Usage:
    python train.py                         # uses config.yaml
    python train.py --resume runs/best.pt
"""
import argparse
import importlib.util
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TillageDataset, get_transforms
from lora_sam2 import inject_lora


# ── Reproducibility ───────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ── Find SAM 2 config YAML on disk ───────────────────────────
def find_sam2_config(config_file: str) -> Path:
    p = Path(config_file)
    if p.exists():
        return p
    spec = importlib.util.find_spec("sam2")
    if spec:
        for loc in (spec.submodule_search_locations or []):
            c = Path(loc) / "configs" / config_file
            if c.exists():
                return c
    for clone in ["sam2_install", "segment-anything-2", "sam2"]:
        c = Path(clone) / "sam2" / "configs" / config_file
        if c.exists():
            return c
    raise FileNotFoundError(f"SAM 2 config not found: {config_file}")


# ── Load SAM 2 + inject LoRA ─────────────────────────────────
def load_model(cfg, device):
    print(f"\n[Model] Loading SAM 2 ...")
    yaml_path = find_sam2_config(cfg["model"]["config_file"])
    print(f"  Config     : {yaml_path}")
    print(f"  Checkpoint : {cfg['model']['checkpoint']}")

    hydra_cfg = OmegaConf.load(yaml_path)
    OmegaConf.resolve(hydra_cfg)
    model = instantiate(hydra_cfg.model, _recursive_=True)

    sd = torch.load(cfg["model"]["checkpoint"], map_location="cpu", weights_only=True)
    if "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model = model.to(device)

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    # Inject LoRA into image encoder attention layers
    lc = cfg["lora"]
    n = inject_lora(model, rank=lc["rank"], alpha=lc["alpha"],
                    dropout=lc["dropout"], targets=lc["targets"])
    print(f"  LoRA injected into {n} layers")

    # Unfreeze mask decoder fully
    for p in model.sam_mask_decoder.parameters():
        p.requires_grad_(True)
    print("  Mask decoder: trainable")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    if cfg["training"].get("grad_checkpoint", True):
        if hasattr(model.image_encoder, "trunk"):
            model.image_encoder.trunk.use_checkpoint = True
            print("  Gradient checkpointing: ON")

    return model


# ── Forward pass ─────────────────────────────────────────────
# KEY FIX: use model.forward_image() NOT model.image_encoder()
# forward_image() also runs conv_s0/conv_s1 projections on the FPN
# features that the mask decoder's upsampling path requires.
# Calling image_encoder() directly skips those projections and causes
# channel-size mismatches in the decoder (the errors we kept seeing).

def forward_pass(model, batch, device):
    images = batch["image"].to(device)           # (B, 3, H, W)
    B, _, H, W = images.shape

    # Point prompts — always fixed size so batches collate cleanly
    coords = batch["point_coords"].to(device)    # (B, N, 2)
    labels = batch["point_labels"].to(device)    # (B, N)
    labels = labels.clamp(min=0)                 # replace -1 padding with 0

    # Box prompts — only when every item in the batch has one
    boxes = None
    if batch["box_valid"].to(device).all():
        boxes = batch["box"].to(device)          # (B, 4)

    # 1. Encode image  ← THE FIX: forward_image, not image_encoder
    backbone_out = model.forward_image(images)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)

    # Add no_mem_embed if the model uses it (SAM 2.1 does)
    if getattr(model, "directly_add_no_mem_embed", False):
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    # Build feats matching SAM2ImagePredictor._process_and_cache_sample exactly.
    #
    # vision_feats is fine→coarse. After zip([::-1], [::-1]) the list is coarse→fine.
    # The [::-1] at the END flips it back to fine→coarse:
    #   feats[0]  = finest   (32ch,  256×256)  → feat_s0 for decoder
    #   feats[1]  = medium   (64ch,  128×128)  → feat_s1 for decoder
    #   feats[-1] = coarsest (256ch,  64×64)   → image_embed
    feats = [
        f.permute(1, 2, 0).view(B, -1, *sz)
        for f, sz in zip(vision_feats[::-1], feat_sizes[::-1])
    ][::-1]                          # ← this reversal is essential
    image_embed    = feats[-1]       # coarsest (256ch)
    high_res_feats = feats[:-1]      # [finest(256×256), medium(128×128)]

    # 2. Encode prompts
    sparse_emb, dense_emb = model.sam_prompt_encoder(
        points=(coords, labels), boxes=boxes, masks=None
    )

    # 3. Decode mask
    low_res_masks, iou_preds, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_feats,
    )

    # 4. Upsample to input size
    masks = F.interpolate(low_res_masks, (H, W), mode="bilinear", align_corners=False)
    return masks, iou_preds          # (B,1,H,W), (B,1)


# ── Loss ─────────────────────────────────────────────────────
def dice_loss(logits, targets):
    p = logits.sigmoid().view(logits.size(0), -1)
    t = targets.float().view(targets.size(0), -1)
    inter = (p * t).sum(1)
    return 1 - (2*inter + 1) / (p.sum(1) + t.sum(1) + 1)

def compute_loss(logits, gt, iou_preds):
    t = gt.unsqueeze(1).float()
    bce  = F.binary_cross_entropy_with_logits(logits, t)
    dice = dice_loss(logits, t).mean()
    # IoU prediction auxiliary loss
    with torch.no_grad():
        pred_bin = (logits.sigmoid() > 0.5).float()
        inter = (pred_bin * t).sum((2, 3))
        union = (pred_bin + t).clamp(0, 1).sum((2, 3))
        true_iou = inter / (union + 1e-6)
    iou_loss = F.mse_loss(iou_preds, true_iou)
    return bce + dice + 0.5 * iou_loss

@torch.no_grad()
def iou_metric(logits, gt):
    pred = (logits.sigmoid() > 0.5).float().view(logits.size(0), -1)
    tgt  = gt.float().view(gt.size(0), -1)
    inter = (pred * tgt).sum(1)
    union = (pred + tgt).clamp(0,1).sum(1)
    return (inter / (union + 1e-6)).mean().item()


# ── Data helpers ─────────────────────────────────────────────
def make_loaders(cfg):
    from sklearn.model_selection import train_test_split
    img_dir = Path(cfg["data"]["root"]) / cfg["data"]["image_subdir"]
    msk_dir = Path(cfg["data"]["root"]) / cfg["data"]["mask_subdir"]
    ext_i   = cfg["data"]["image_ext"]
    ext_m   = cfg["data"]["mask_ext"]

    imgs = sorted(img_dir.glob(f"*{ext_i}"))
    pairs = [(p, msk_dir / (p.stem + ext_m)) for p in imgs
             if (msk_dir / (p.stem + ext_m)).exists()]
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found in {img_dir}")

    random.seed(42)
    random.shuffle(pairs)
    n_test = max(1, int(len(pairs) * cfg["data"]["test_split"]))
    n_val  = max(1, int(len(pairs) * cfg["data"]["val_split"]))
    test_p  = pairs[:n_test]
    val_p   = pairs[n_test:n_test + n_val]
    train_p = pairs[n_test + n_val:]
    print(f"  Split — train:{len(train_p)}  val:{len(val_p)}  test:{len(test_p)}")

    sz = cfg["data"]["image_size"]
    p  = cfg["prompts"]
    kw = dict(num_pos=p["num_positive_points"], num_neg=p["num_negative_points"],
              use_box=p["use_box_prompts"])

    bs = cfg["training"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    train_ds = TillageDataset(train_p, get_transforms(sz, aug=True),  **kw)
    val_ds   = TillageDataset(val_p,   get_transforms(sz, aug=False), **kw)
    test_ds  = TillageDataset(test_p,  get_transforms(sz, aug=False), **kw)

    return (DataLoader(train_ds, bs,   shuffle=True,  num_workers=nw,
                       pin_memory=True, drop_last=True),
            DataLoader(val_ds,   1,    shuffle=False, num_workers=nw, pin_memory=True),
            DataLoader(test_ds,  1,    shuffle=False, num_workers=nw, pin_memory=True))


# ── Main training loop ────────────────────────────────────────
def train(cfg, resume=None):
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}" + (f" — {torch.cuda.get_device_name(0)}"
                                   if device.type == "cuda" else ""))

    run_dir = Path(cfg["output"]["run_dir"]) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Data]")
    train_loader, val_loader, _ = make_loaders(cfg)

    model = load_model(cfg, device)

    tc       = cfg["training"]
    params   = [p for p in model.parameters() if p.requires_grad]
    opt      = AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    epochs   = tc["epochs"]
    sched    = CosineAnnealingLR(opt, T_max=epochs, eta_min=tc["min_lr"])
    scaler   = GradScaler("cuda", enabled=(device.type == "cuda"))
    dtype    = torch.bfloat16 if tc.get("bf16", True) and device.type == "cuda" else torch.float32

    start_epoch = 0
    best_iou    = 0.0

    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_iou    = ckpt.get("best_iou", 0.0)
        print(f"[Resume] epoch {start_epoch}, best IoU {best_iou:.4f}")

    print(f"\n[Train] {epochs} epochs — saving to {run_dir}\n")

    for epoch in range(start_epoch, epochs):
        # ── Train ──────────────────────────────────────────
        model.train()
        t_loss, t_iou = 0.0, 0.0
        accum = tc.get("grad_accum", 4)
        opt.zero_grad()

        pbar = tqdm(train_loader, desc=f"Ep{epoch+1:03d} train", leave=False)
        for i, batch in enumerate(pbar):
            with autocast("cuda", dtype=dtype):
                logits, iou_preds = forward_pass(model, batch, device)
                loss = compute_loss(logits, batch["mask"].to(device), iou_preds) / accum

            scaler.scale(loss).backward()

            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad()

            t_loss += loss.item() * accum
            t_iou  += iou_metric(logits, batch["mask"].to(device))
            pbar.set_postfix(loss=f"{loss.item()*accum:.3f}",
                             iou=f"{t_iou/(i+1):.3f}")

        sched.step()

        # ── Val ────────────────────────────────────────────
        model.eval()
        v_loss, v_iou = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                with autocast("cuda", dtype=dtype):
                    logits, iou_preds = forward_pass(model, batch, device)
                    loss = compute_loss(logits, batch["mask"].to(device), iou_preds)
                v_loss += loss.item()
                v_iou  += iou_metric(logits, batch["mask"].to(device))

        n_train = len(train_loader)
        n_val   = len(val_loader)
        val_iou = v_iou / n_val

        print(f"Epoch {epoch+1:03d}  "
              f"train loss={t_loss/n_train:.4f} iou={t_iou/n_train:.4f}  |  "
              f"val loss={v_loss/n_val:.4f} iou={val_iou:.4f}")

        # ── Save ───────────────────────────────────────────
        is_best = val_iou > best_iou
        if is_best:
            best_iou = val_iou

        if (epoch + 1) % cfg["output"]["save_every"] == 0 or is_best:
            ckpt_path = run_dir / ("best.pt" if is_best else f"ckpt_ep{epoch+1:03d}.pt")
            torch.save({
                "epoch":     epoch,
                "model":     {k: v for k, v in model.state_dict().items()
                               if any(x in k for x in ["lora_A","lora_B","sam_mask_decoder"])},
                "optimizer": opt.state_dict(),
                "best_iou":  best_iou,
            }, ckpt_path)
            tag = " ★ best" if is_best else ""
            print(f"  Saved → {ckpt_path.name}{tag}")

    print(f"\n[Done] Best val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args.resume)