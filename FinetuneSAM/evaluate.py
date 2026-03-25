"""
evaluate.py
───────────
Run evaluation on the held-out test set and report IoU, Dice, F1.

Usage
-----
  python evaluate.py \
      --checkpoint runs/.../best.pt \
      --config config.yaml \
      --output_dir eval_results/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import autocast
from tqdm import tqdm

from dataset import build_dataloaders
from lora_sam2 import configure_sam2_for_finetuning, load_lora_weights
from losses import compute_metrics
from train import forward_pass, build_model

# SAM 2
from sam2.build_sam import build_sam2


def evaluate(cfg: dict, checkpoint: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────
    sam2, _ = build_model(cfg, device)
    load_lora_weights(sam2, checkpoint)
    sam2.eval()
    # lora, sam, semantic segmentation . computer vision

    # ── Test set ────────────────────────────────────────────
    _, _, test_loader = build_dataloaders(cfg)
    dtype = torch.bfloat16 if cfg["training"]["mixed_precision"] == "bf16" else torch.float16

    all_metrics: Dict[str, List[float]] = {
        k: [] for k in ["iou", "dice", "precision", "recall", "f1"]
    }

    pbar = tqdm(test_loader, desc="Evaluating")
    for i, batch in enumerate(pbar):
        gt_masks = batch["mask"].to(device, non_blocking=True)

        with torch.no_grad(), autocast(dtype=dtype):
            pred_logits, _ = forward_pass(sam2, batch, device)

        m = compute_metrics(pred_logits.float(), gt_masks)
        for k in all_metrics:
            all_metrics[k].append(m[k])

        # Save a few qualitative examples
        if i < 20:
            _save_example(
                batch, pred_logits, gt_masks,
                save_path=out_dir / f"example_{i:03d}.jpg"
            )

        pbar.set_postfix(iou=f"{m['iou']:.3f}", dice=f"{m['dice']:.3f}")

    # ── Aggregate ───────────────────────────────────────────
    summary = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    summary["n_samples"] = len(test_loader.dataset)

    print("\n╔══════════════════════════════════════╗")
    print("║       Test Set Evaluation Results    ║")
    print("╠══════════════════════════════════════╣")
    for k, v in summary.items():
        if k != "n_samples":
            print(f"║  {k:<12}  {v:.4f}                  ║")
    print(f"║  {'samples':<12}  {summary['n_samples']}                     ║")
    print("╚══════════════════════════════════════╝")

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {results_path}")

    _plot_metric_distributions(all_metrics, out_dir / "metric_distributions.png")
    return summary


def _save_example(batch, pred_logits, gt_masks, save_path: Path):
    """Save side-by-side: image | GT mask | predicted mask."""
    # Denormalise image
    img = batch["image"][0].cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img * std + mean).clip(0, 1)

    gt   = gt_masks[0].cpu().numpy()
    pred = (pred_logits[0, 0].sigmoid().cpu().numpy() > 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);        axes[0].set_title("Image");         axes[0].axis("off")
    axes[1].imshow(gt, cmap="gray");   axes[1].set_title("GT Mask");      axes[1].axis("off")
    axes[2].imshow(pred, cmap="gray"); axes[2].set_title("Predicted Mask"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def _plot_metric_distributions(
    all_metrics: Dict[str, List[float]], save_path: Path
):
    """Histogram of per-sample metric values."""
    fig, axes = plt.subplots(1, len(all_metrics), figsize=(5 * len(all_metrics), 4))
    for ax, (k, vals) in zip(axes, all_metrics.items()):
        ax.hist(vals, bins=20, color="steelblue", edgecolor="white")
        ax.axvline(np.mean(vals), color="red", linestyle="--",
                   label=f"mean={np.mean(vals):.3f}")
        ax.set_title(k.upper())
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend()
    plt.suptitle("Per-sample metric distributions (test set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Distribution plot saved → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--output_dir", default="eval_results/")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args.checkpoint, args.output_dir)
