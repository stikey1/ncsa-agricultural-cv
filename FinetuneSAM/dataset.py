"""
dataset.py — TillageDataset + transforms
"""
import random
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def get_transforms(image_size: int = 1024, aug: bool = True) -> A.Compose:
    if aug:
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size),
                                scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.HueSaturationValue(20, 30, 20, p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(std_range=(0.02, 0.15), p=0.2),
            A.CoarseDropout(num_holes_range=(1, 6),
                            hole_height_range=(8, 32),
                            hole_width_range=(8, 32),
                            fill=0, fill_mask=0, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


def _sample_points(mask: np.ndarray, num_pos: int, num_neg: int):
    """
    Sample exactly num_pos + num_neg points from mask.
    Uses replace=True when there aren't enough pixels of a class.
    Empty-class slots are filled with centre pixel, label=-1 (ignored).
    """
    h, w = mask.shape
    cx, cy = w // 2, h // 2
    pos_yx = np.argwhere(mask > 0)
    neg_yx = np.argwhere(mask == 0)

    coords, labels = [], []

    if len(pos_yx) > 0:
        idx = np.random.choice(len(pos_yx), num_pos,
                                replace=len(pos_yx) < num_pos)
        for y, x in pos_yx[idx]:
            coords.append([float(x), float(y)]); labels.append(1)
    else:
        for _ in range(num_pos):
            coords.append([float(cx), float(cy)]); labels.append(-1)

    if len(neg_yx) > 0:
        idx = np.random.choice(len(neg_yx), num_neg,
                                replace=len(neg_yx) < num_neg)
        for y, x in neg_yx[idx]:
            coords.append([float(x), float(y)]); labels.append(0)
    else:
        for _ in range(num_neg):
            coords.append([float(cx), float(cy)]); labels.append(-1)

    return (np.array(coords, dtype=np.float32),
            np.array(labels, dtype=np.int64))


def _mask_to_box(mask: np.ndarray, noise: int = 10):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x1 = max(0,   xs.min() - random.randint(0, noise))
    y1 = max(0,   ys.min() - random.randint(0, noise))
    x2 = min(w-1, xs.max() + random.randint(0, noise))
    y2 = min(h-1, ys.max() + random.randint(0, noise))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class TillageDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], transform,
                 num_pos: int = 5, num_neg: int = 5,
                 use_box: bool = True, box_noise: int = 10):
        self.pairs     = pairs
        self.transform = transform
        self.num_pos   = num_pos
        self.num_neg   = num_neg
        self.use_box   = use_box
        self.box_noise = box_noise

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        raw = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        msk = (raw > 127).astype(np.uint8)

        aug = self.transform(image=img, mask=msk)
        image = aug["image"]                          # (3,H,W) float32 tensor
        mask  = aug["mask"].numpy().astype(np.uint8)  # (H,W)

        coords, labels = _sample_points(mask, self.num_pos, self.num_neg)

        box = _mask_to_box(mask, self.box_noise) if self.use_box else None

        return {
            "image":        image,
            "mask":         torch.from_numpy(mask).long(),
            "point_coords": torch.from_numpy(coords),
            "point_labels": torch.from_numpy(labels),
            "box":          torch.from_numpy(box) if box is not None
                            else torch.zeros(4, dtype=torch.float32),
            "box_valid":    torch.tensor(1 if box is not None else 0),
        }