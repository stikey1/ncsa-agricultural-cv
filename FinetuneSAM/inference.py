"""
inference.py
────────────
Run fine-tuned SAM 2 for tillage residue segmentation.

Supports two modes
------------------
  1. PROMPTED   — provide point and/or box prompts
  2. AUTOMATIC  — grid-based automatic mask generation

Usage
-----
  # Prompted (single image)
  python inference.py \
      --mode prompted \
      --image path/to/field.jpg \
      --checkpoint runs/.../best.pt \
      --config config.yaml

  # Automatic (single image)
  python inference.py \
      --mode auto \
      --image path/to/field.jpg \
      --checkpoint runs/.../best.pt \
      --config config.yaml

  # Batch (directory)
  python inference.py \
      --mode auto \
      --image_dir path/to/images/ \
      --checkpoint runs/.../best.pt \
      --config config.yaml \
      --output_dir predictions/
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# SAM 2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from lora_sam2 import configure_sam2_for_finetuning, load_lora_weights


# ─────────────────────────────────────────────────────────────
#  Model loader
# ─────────────────────────────────────────────────────────────

def load_finetuned_model(cfg: dict, checkpoint_path: str, device: torch.device):
    """
    Load SAM 2, inject LoRA, restore fine-tuned weights.
    """
    print(f"[Model] Loading SAM 2 base ({cfg['model']['variant']}) ...")
    sam2 = build_sam2(
        config_file=cfg["model"]["config_file"],
        ckpt_path=cfg["model"]["checkpoint"],
        device=device,
    )

    lora_cfg = cfg["lora"]
    configure_sam2_for_finetuning(
        sam2,
        lora_rank=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=0.0,                       # no dropout during inference
        lora_target_modules=lora_cfg["target_modules"],
        train_image_encoder=lora_cfg["train_image_encoder"],
        train_mask_decoder=lora_cfg["train_mask_decoder"],
        train_prompt_encoder=lora_cfg["train_prompt_encoder"],
    )

    load_lora_weights(sam2, checkpoint_path)
    sam2.eval()
    print("  Model ready for inference.")
    return sam2


# ─────────────────────────────────────────────────────────────
#  Prompted inference
# ─────────────────────────────────────────────────────────────

def predict_prompted(
    predictor: SAM2ImagePredictor,
    image: np.ndarray,                        # (H, W, 3) RGB uint8
    point_coords: Optional[np.ndarray] = None,  # (N, 2) [x, y]
    point_labels: Optional[np.ndarray] = None,  # (N,)   {0, 1}
    box: Optional[np.ndarray] = None,           # (4,) [x1, y1, x2, y2]
    multimask_output: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (masks, scores, logits)
      masks  : (N_masks, H, W)  bool
      scores : (N_masks,)       float — predicted IoU per mask
      logits : (N_masks, H, W)  float
    """
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=multimask_output,
    )
    # Sort by score descending
    order = np.argsort(scores)[::-1]
    return masks[order], scores[order], logits[order]


# ─────────────────────────────────────────────────────────────
#  Automatic mask generation
# ─────────────────────────────────────────────────────────────

def predict_automatic(
    generator: SAM2AutomaticMaskGenerator,
    image: np.ndarray,                     # (H, W, 3) RGB uint8
    residue_score_threshold: float = 0.5,  # filter by predicted IoU
) -> List[dict]:
    """
    Returns a list of mask dicts (same format as SAM's output):
      {
        "segmentation": (H, W) bool,
        "area": int,
        "bbox": [x, y, w, h],
        "predicted_iou": float,
        "stability_score": float,
      }
    Filtered to masks above `residue_score_threshold`.
    """
    masks = generator.generate(image)
    filtered = [m for m in masks
                if m["predicted_iou"] >= residue_score_threshold]
    filtered.sort(key=lambda m: m["predicted_iou"], reverse=True)
    return filtered


def masks_to_binary(masks_list: List[dict], shape: Tuple[int, int]) -> np.ndarray:
    """Merge a list of SAM mask dicts into a single binary mask."""
    combined = np.zeros(shape, dtype=np.uint8)
    for m in masks_list:
        combined[m["segmentation"]] = 1
    return combined


# ─────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────

def visualise_result(
    image: np.ndarray,
    mask: np.ndarray,
    title: str = "Residue Segmentation",
    save_path: Optional[str] = None,
    point_coords: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    box: Optional[np.ndarray] = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Predicted mask
    axes[1].imshow(mask, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_title("Predicted Residue Mask")
    axes[1].axis("off")

    # Overlay
    overlay = image.copy().astype(float) / 255.0
    axes[2].imshow(overlay)
    axes[2].imshow(mask, cmap="Reds", alpha=0.45, vmin=0, vmax=1)

    if point_coords is not None and point_labels is not None:
        for (x, y), lbl in zip(point_coords, point_labels):
            colour = "lime" if lbl == 1 else "red"
            axes[2].plot(x, y, "o", color=colour, markersize=8, markeredgecolor="white", lw=2)

    if box is not None:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=2, edgecolor="cyan", facecolor="none")
        axes[2].add_patch(rect)

    axes[2].set_title("Overlay (red = residue)")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved visualisation → {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────
#  Batch processing
# ─────────────────────────────────────────────────────────────

def run_batch(
    mode: str,
    image_dir: str,
    checkpoint: str,
    cfg: dict,
    output_dir: str,
    image_ext: str = ".jpg",
    residue_threshold: float = 0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2 = load_finetuned_model(cfg, checkpoint, device)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    vis_dir   = out_dir / "visualisations"
    masks_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    image_paths = sorted(Path(image_dir).glob(f"*{image_ext}"))
    print(f"\n[Batch] {len(image_paths)} images found in {image_dir}")

    if mode == "auto":
        generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=32,
            pred_iou_thresh=residue_threshold,
            stability_score_thresh=0.92,
            box_nms_thresh=0.7,
            min_mask_region_area=200,       # filter tiny noise regions
        )
    else:
        predictor = SAM2ImagePredictor(sam2)

    for img_path in image_paths:
        image_bgr = cv2.imread(str(img_path))
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        if mode == "auto":
            masks_list = predict_automatic(generator, image, residue_threshold)
            binary_mask = masks_to_binary(masks_list, (H, W))
        else:
            # Prompted: auto-generate centre point as fallback
            cx, cy = W // 2, H // 2
            point_coords = np.array([[cx, cy]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int64)
            masks, _, _ = predict_prompted(predictor, image,
                                           point_coords=point_coords,
                                           point_labels=point_labels,
                                           multimask_output=False)
            binary_mask = masks[0].astype(np.uint8)

        # Save mask
        mask_save_path = masks_dir / (img_path.stem + "_mask.png")
        cv2.imwrite(str(mask_save_path), binary_mask * 255)

        # Save visualisation
        vis_save_path = vis_dir / (img_path.stem + "_vis.jpg")
        visualise_result(image, binary_mask, title=img_path.name,
                         save_path=str(vis_save_path))

    print(f"\n[Done] Masks saved to: {masks_dir}")
    print(f"       Visualisations : {vis_dir}")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 2 tillage residue inference")
    parser.add_argument("--mode",       choices=["prompted", "auto"], required=True)
    parser.add_argument("--image",      type=str, default=None,
                        help="Single image path")
    parser.add_argument("--image_dir",  type=str, default=None,
                        help="Directory of images (batch mode)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned LoRA checkpoint (.pt)")
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--output_dir", type=str, default="predictions/")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Residue IoU / score threshold (auto mode)")
    parser.add_argument("--ext",        type=str, default=".jpg",
                        help="Image extension for batch mode")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Single image ────────────────────────────────────────
    if args.image:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam2 = load_finetuned_model(cfg, args.checkpoint, device)

        image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        if args.mode == "auto":
            generator = SAM2AutomaticMaskGenerator(
                model=sam2, points_per_side=32,
                pred_iou_thresh=args.threshold,
                min_mask_region_area=200,
            )
            masks_list = predict_automatic(generator, image, args.threshold)
            binary_mask = masks_to_binary(masks_list, (H, W))
        else:
            predictor = SAM2ImagePredictor(sam2)
            cx, cy = W // 2, H // 2
            point_coords = np.array([[cx, cy]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int64)
            masks, scores, _ = predict_prompted(
                predictor, image, point_coords=point_coords,
                point_labels=point_labels, multimask_output=True
            )
            binary_mask = masks[0].astype(np.uint8)
            print(f"  Top mask IoU score: {scores[0]:.4f}")

        visualise_result(image, binary_mask, title=Path(args.image).name)

    # ── Batch ───────────────────────────────────────────────
    elif args.image_dir:
        run_batch(args.mode, args.image_dir, args.checkpoint, cfg,
                  args.output_dir, args.ext, args.threshold)
    else:
        parser.error("Provide --image or --image_dir")
