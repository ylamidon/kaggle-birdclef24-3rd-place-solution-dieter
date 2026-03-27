# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Christof Henkel's (Dieter) contribution to the **BirdCLEF 2024 Kaggle competition** (3rd place, NVBird team). Multi-label bird species classification (182 species) from audio recordings using a two-stage training pipeline: base models → pseudo-labeled refinement.

## Setup & Commands

```bash
# Environment (Docker recommended)
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:24.03-py3
pip install -r requirements_ngc.txt

# Data preparation (run once)
python scripts/convert_np_first_last_10_v2.py   # OGG → numpy arrays
python scripts/create_train_folded_v3c.py        # Create k-fold splits
python scripts/create_test_fake_gt.py            # Test ground truth

# Training
python train.py -C cfg_1 --fold -1              # Single run (random seed)
python train.py -C cfg_1 --fold 0               # With validation fold 0
python train.py -C cfg_pl_1 --fold -1 --pl_df pl_blended_0.csv  # Pseudo-label round
```

No test suite or linter configured.

## Two-Stage Training Pipeline

**Round 1**: Train 3 configs × 5 seeds = 15 models; average predictions into `pl_blended_X.csv`

**Round 2**: Train `cfg_pl_1` using those pseudo-labels as `--pl_df` argument; repeat 5×

**Final submission**: Average predictions across all models.

## Architecture

### Configuration System

All configs inherit from `configs/default_config.py` (`basic_cfg`). Each specific config (e.g. `cfg_1.py`) imports it and overrides fields. CLI args (e.g. `--fold`, `--pl_df`, `--epochs`) override config values at runtime via `setattr(cfg, arg_key, val)`.

Key config fields: `cfg.model`, `cfg.dataset`, `cfg.metric`, `cfg.birds` (182-length list), data paths, training hyperparameters, Neptune project.

### Data Pipeline

- **Input format**: `.npy` files (pre-converted from OGG), first/last 10 seconds extracted separately
- **Sample rate**: 32kHz, clips to 5-second windows (~160k samples)
- **Augmentations**: mixup (blend two samples), random secondary-sample overlay, random cropping; defined in `configs/augmentations.py`
- **Multi-label**: primary labels always counted; secondary labels masked in loss
- `ds_pl_1.py` extends `ds_1.py` to load pseudo-label soft targets

### Model Architecture (mdl_1.py is primary)

1. `Preprocessor`: normalize → MelSpectrogram (288 bins, 90Hz–14kHz) → AmplitudeToDB → normalize (mean=40, std=80)
2. Backbone: `efficientvit_b1.r288_in1k` (timm) — 3-channel input (2× mel-spec + positional encoding), 182-logit output
3. Loss: BCEWithLogitsLoss with secondary-label masking

### Training Loop (train.py)

Mixed-precision (autocast + GradScaler), optional DDP multi-GPU, gradient accumulation + clipping, cosine LR with warmup. Metrics computed every N epochs. Neptune.ai tracking optional (`cfg.neptune_project`).

Key functions: `train()`, `run_eval()`, `get_data()`, `get_dataset()`, `get_model()`.

### Evaluation

`metrics/metric_1.py`: per-species binary AUROC (only species present in GT), averaged across species. Test predictions reshaped into 5-second windows; outputs `submission.csv` with `row_id` format.

## Data Paths

Expected at `/mount/birdclef24/data/` (Linux/Docker):
- `train_audio/` — original OGG files
- `numpy_arrays/` — pre-converted `.npy` files
- `train_folded_v3c.csv` — metadata with fold assignments
- `background/` — optional background noise augmentation data
- `birdaves-biox-base` — optional AVES pretrained weights
