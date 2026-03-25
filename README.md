# SAM2 Fine-Tuning with LoRA for Agricultural Image Segmentation

Fine-tuning Meta's Segment Anything Model 2 (SAM2) using Low-Rank Adaptation (LoRA) for segmentation of agricultural street-view imagery.

---

## Overview

SAM2 is a powerful general-purpose segmentation model, but out-of-the-box performance on domain-specific agricultural imagery leaves room for improvement. This project fine-tunes SAM2 using LoRA — a parameter-efficient training technique that injects trainable low-rank matrices into the model's attention layers — allowing the model to adapt to agricultural scenes without retraining the full 81M parameter backbone.

---

## Approach

- **Base model:** SAM2.1 Hiera Base+ (`sam2.1_hiera_base_plus`)
- **Adaptation method:** LoRA injected into 51 attention layers
- **Trainable parameters:** 5,272,389 out of 81,907,458 (6.4%) — the rest of the backbone is frozen
- **Mask decoder:** fully trainable
- **Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU
- **Gradient checkpointing:** enabled to reduce memory usage

---

## Dataset

Agricultural street-view imagery split into:

| Split | Images |
|---|---|
| Train | 1,089 |
| Val | 203 |
| Test | 67 |

Images were manually labeled using Napari for ground-truth mask generation.

---

## Results

Model trained to segment agricultural features from street-level imagery. Best checkpoint saved based on validation performance during training.

---

## Project Structure

```
FinetuneSAM/
├── train.py          # Training loop with LoRA injection and gradient checkpointing
├── evaluate.py       # Evaluation metrics on validation and test sets
├── inference.py      # Run inference on new images
├── lora_sam2.py      # LoRA implementation and injection into SAM2 layers
├── losses.py         # Custom loss functions
├── dataset.py        # Dataset loading and preprocessing
├── sam2download.py   # SAM2 checkpoint download helper
└── config.yaml       # Training configuration

label_napari.py       # Napari-based GUI tool for manual mask annotation
visualizer.py         # Visualization utilities for predictions and ground truth
predict.py            # Batch prediction script
```

---

## Usage

**Train:**
```bash
python train.py --config config.yaml
```

**Resume from checkpoint:**
```bash
python train.py --config config.yaml --resume runs/YOUR_RUN/best.pt
```

**Run inference:**
```bash
python inference.py --config config.yaml --checkpoint runs/YOUR_RUN/best.pt
```

---

## Dependencies

See `requirements.txt`. Key libraries:

- PyTorch
- SAM2 (`sam2_install/`)
- Napari
- OpenCV
- NumPy

---

## Notes

This project was completed as part of the Students Pushing Innovation program at the **National Center for Supercomputing Applications (NCSA), University of Illinois at Urbana-Champaign**.
