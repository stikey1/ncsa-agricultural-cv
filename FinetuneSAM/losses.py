"""
losses.py
─────────
Loss functions for binary segmentation fine-tuning.

  CombinedSegLoss  =  w_bce * BCE  +  w_dice * Dice  +  w_focal * Focal
                    + w_iou * IoU_pred_loss  (auxiliary head in SAM decoder)
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  Individual losses
# ─────────────────────────────────────────────────────────────

class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary segmentation.
    Down-weights easy negatives to focus training on hard examples.
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, 1, H, W) raw logits
        targets : (B, 1, H, W) binary float {0, 1}
        """
        prob = torch.sigmoid(logits)
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t  = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Robust to class imbalance (residues can be sparse).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, 1, H, W)
        targets : (B, 1, H, W)
        """
        prob = torch.sigmoid(logits)
        prob_flat    = prob.view(prob.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (prob_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / \
               (prob_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


# ─────────────────────────────────────────────────────────────
#  Combined loss
# ─────────────────────────────────────────────────────────────

class CombinedSegLoss(nn.Module):
    """
    Weighted combination of BCE + Dice + Focal + auxiliary IoU loss.

    Parameters
    ----------
    bce_weight   : weight for binary cross-entropy
    dice_weight  : weight for soft Dice
    focal_weight : weight for focal loss (set 0 to disable)
    iou_weight   : weight for auxiliary IoU prediction loss
    """

    def __init__(
        self,
        bce_weight:   float = 1.0,
        dice_weight:  float = 1.0,
        focal_weight: float = 0.5,
        iou_weight:   float = 1.0,
    ):
        super().__init__()
        self.bce_w   = bce_weight
        self.dice_w  = dice_weight
        self.focal_w = focal_weight
        self.iou_w   = iou_weight

        self.bce   = nn.BCEWithLogitsLoss()
        self.dice  = DiceLoss()
        self.focal = BinaryFocalLoss()

    def forward(
        self,
        pred_logits:     torch.Tensor,   # (B, 1, H, W) — upsampled mask logits
        gt_masks:        torch.Tensor,   # (B, H, W)    — binary int64
        iou_predictions: torch.Tensor,   # (B, 1)       — SAM's IoU head output
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with individual loss values and the combined total.
        """
        # Shape alignment
        target = gt_masks.unsqueeze(1).float()   # (B, 1, H, W)

        bce_loss   = self.bce(pred_logits, target)
        dice_loss  = self.dice(pred_logits, target)
        focal_loss = self.focal(pred_logits, target) if self.focal_w > 0 else torch.tensor(0.0)

        # Auxiliary IoU loss: MSE between predicted IoU and actual IoU
        with torch.no_grad():
            pred_bin = (pred_logits.sigmoid() > 0.5).float()
            inter = (pred_bin * target).sum(dim=(2, 3))
            union = (pred_bin + target).clamp(0, 1).sum(dim=(2, 3))
            true_iou = (inter / (union + 1e-6))                  # (B, 1)
        iou_loss = F.mse_loss(iou_predictions, true_iou)

        total = (self.bce_w   * bce_loss
               + self.dice_w  * dice_loss
               + self.focal_w * focal_loss
               + self.iou_w   * iou_loss)

        return {
            "loss":       total,
            "bce_loss":   bce_loss.detach(),
            "dice_loss":  dice_loss.detach(),
            "focal_loss": focal_loss.detach(),
            "iou_loss":   iou_loss.detach(),
        }


# ─────────────────────────────────────────────────────────────
#  Metrics (no gradients)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(
    pred_logits: torch.Tensor,    # (B, 1, H, W)
    gt_masks:    torch.Tensor,    # (B, H, W)
    threshold:   float = 0.5,
) -> Dict[str, float]:
    """
    Compute IoU, Dice, Precision, Recall, F1 for a batch.
    Returns mean values across the batch.
    """
    pred = (pred_logits.sigmoid() > threshold).float().squeeze(1)  # (B, H, W)
    gt   = gt_masks.float()

    pred_flat = pred.view(pred.size(0), -1)
    gt_flat   = gt.view(gt.size(0), -1)

    tp = (pred_flat * gt_flat).sum(dim=1)
    fp = (pred_flat * (1 - gt_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * gt_flat).sum(dim=1)

    iou       = (tp / (tp + fp + fn + 1e-6)).mean().item()
    dice      = (2 * tp / (2 * tp + fp + fn + 1e-6)).mean().item()
    precision = (tp / (tp + fp + 1e-6)).mean().item()
    recall    = (tp / (tp + fn + 1e-6)).mean().item()
    f1        = (2 * precision * recall / (precision + recall + 1e-6))

    return {
        "iou":       iou,
        "dice":      dice,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }
