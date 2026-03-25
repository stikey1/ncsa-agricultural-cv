"""
lora_sam2.py — Minimal LoRA injection for SAM 2 image encoder
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank adapter."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.linear  = linear
        self.scaling = alpha / rank
        device       = next(linear.parameters()).device  # match parent device

        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features,  device=device))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    def forward(self, x):
        return self.linear(x) + self.drop(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


def inject_lora(model: nn.Module, rank=16, alpha=32.0, dropout=0.05,
                targets=("qkv", "proj")) -> int:
    """
    Walk the model, find every nn.Linear whose attribute name is in
    `targets` (inside image_encoder), replace with LoRALinear.
    Returns the number of layers injected.
    """
    targets = set(targets)
    count   = 0

    def _set(parent, key, mod):
        setattr(parent, key, mod)

    def _walk(module, prefix=""):
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if "image_encoder" in full and isinstance(child, nn.Linear) and name in targets:
                _set(module, name, LoRALinear(child, rank, alpha, dropout))
                count_ref[0] += 1
            else:
                _walk(child, full)

    count_ref = [0]
    _walk(model)
    return count_ref[0]


def save_lora(model: nn.Module, path: str):
    """Save only trainable weights (LoRA + mask decoder)."""
    state = {k: v for k, v in model.state_dict().items()
             if any(x in k for x in ["lora_A", "lora_B", "sam_mask_decoder"])}
    torch.save(state, path)


def load_lora(model: nn.Module, path: str):
    """Load LoRA + mask decoder weights into an already-injected model."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)