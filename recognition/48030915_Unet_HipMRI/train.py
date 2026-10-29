
# train.py
# Training, validation, testing and saving. Imports the model from modules.py and the data
# from dataset.py (spec rule). Plots losses and metrics during training.

# Run: python train.py --dim 2d   |   python train.py --dim 3d

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (CLASS_NAMES, DATA_ROOT, N_CLASSES, PROSTATE, HipMRISlices, augment_2d, check_no_leakage,
                     prostate_sampler, split_cases)
from modules import DiceCELoss, SegMetrics, build_model, count_parameters

DIM_DEFAULTS = {
    "2d": {"batch_size": 16, "epochs": 60, "patience": 12},
    "3d": {"batch_size": 2, "epochs": 150, "patience": 30},
}
BASE_FILTERS = {"improved": 16, "unet": 32} 
TARGET_DICE = {"2d": 0.75, "3d": 0.70} 

# SHARED 

def get_args():
    ap = argparse.ArgumentParser(description="Train a U-Net on HipMRI")
    ap.add_argument("--dim", default="2d", choices=["2d", "3d"])
    ap.add_argument("--model", default="improved", choices=["improved", "unet"])
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default=None, help="output folder (default runs/<model><dim>)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--base-filters", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--prostate-weight", type=float, default=3.0,
                    help="2D: slices with prostate are drawn this many times more often (1 = off)")
    ap.add_argument("--patience", type=int, default=None, help="early stopping: epochs without improvement")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--max-scans", type=int, default=None, help="2D: only load N scans per split (quick tests)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true", help="turn off mixed precision")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # fill in the defaults that depend on --dim / --model
    for key, value in DIM_DEFAULTS[args.dim].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.base_filters is None:
        args.base_filters = BASE_FILTERS[args.model]
    if args.out is None:
        args.out = f"runs/{args.model}{args.dim}"
    return args


def set_seed(seed):
    """Same seed -> same initial weights, same shuffling, same augmentation: reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_grad_scaler(enabled):
    """GradScaler for mixed precision (the API moved in newer PyTorch versions)."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)

# train_one_epoch(model, loader, loss_fn, optimiser, scaler, device, augment_fn) -> mean loss
#   - for each batch: to GPU -> augment -> forward -> loss -> backward -> optimiser step
#   - mixed precision (torch.autocast + GradScaler): faster, about half the memory

# validate(model, loader, loss_fn) -> val loss, per-class Dice
#   - model.eval(), torch.no_grad(), argmax -> SegMetrics

# plot_curves(history, path)
#   - left: train vs validation loss; right: validation Dice per class (prostate in bold,
#     dashed line at the target). Save every epoch so you can look while it trains.

# main()
#   1. data: build datasets with the dataset.py functions; check_no_leakage; print summaries
#   2. model, loss (DiceCELoss), optimiser (AdamW), learning-rate schedule (cosine)
#   3. for each epoch: train_one_epoch -> validate -> scheduler step
#        - save last.pt every epoch; save best.pt when validation PROSTATE Dice improves
#          (store the settings inside the checkpoint, e.g. dim, model, base_filters, crop)
#        - log a row to history.csv: losses, Dice per class, lr, epoch time, peak VRAM
#        - plot_curves
#        - early stopping: stop after `patience` epochs without improvement
#   4. TEST: reload best.pt, run on the test split, print + save per-class Dice and IoU


# 2D Improved U-Net

#   - datasets: HipMRISlices("train"/"validate"/"test") from the provided folders
#   - DataLoader with the prostate oversampling sampler; batch 16
#   - augment_2d; target: test prostate Dice >= 0.75


# 3D Improved U-Net

#   - split_volumes(match2d) -> crop centre from TRAINING labels -> HipMRIVolumes
#     (train=True for random shifted crops; train=False for validation/test)
#   - batch 2 (memory); validation batch 1
#   - augment_3d; target: test prostate Dice >= 0.70
#   - --resume from last.pt (save optimiser/scheduler/history in it) for Rangpur time limits
#   - store crop centre, crop size and the split's volume IDs in the checkpoint for predict.py



