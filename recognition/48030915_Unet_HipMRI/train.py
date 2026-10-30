"""
train.py - Training, validation, testing and saving.

The model comes from modules.py and the data from dataset.py (spec rule). Losses and
per-class Dice are plotted after every epoch.

Run:
    python train.py --dim 2d --model improved          # 2D Improved U-Net
    python train.py --dim 2d --model unet              # plain 2D U-Net (tutorial style)
    python train.py --dim 2d --max-scans 4 --epochs 2  # quick test on a few scans
    python train.py --dim 2d --resume                  # continue after a Rangpur time-out
    python train.py --dim 3d                           # 3D Improved U-Net
    python train.py --dim 3d --max-scans 4 --epochs 2  # quick 3D test on a few volumes

Outputs in --out (default runs/<model><dim>, e.g. runs/improved2d):
    best.pt, last.pt     checkpoints (weights + all settings; last.pt also has the optimiser state)
    config.json          all settings
    history.csv          per epoch: losses, validation Dice per class, learning rate, time, peak VRAM
    curves.png           loss and validation Dice curves
    test_metrics.json    per-class Dice / IoU of best.pt on the test split

Adapted code from Lab demo 2 Unet and online resources
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (CLASS_NAMES, CROP_3D, DATA_ROOT, N_CLASSES, PROSTATE, SHIFT_3D, HipMRISlices, HipMRIVolumes,
                     augment_2d, augment_3d, check_no_leakage, crop_centre_from_labels, list_volume_pairs,
                     prostate_sampler, split_cases, split_volumes)
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
    ap.add_argument("--max-scans", type=int, default=None,
                    help="quick tests: only load N scans (2D) or N volumes (3D) per split")
    ap.add_argument("--cache-dir", default=None,
                    help="3D: save the cropped volumes here so later runs load in seconds (uses a few GB)")
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

def train_one_epoch(model, loader, loss_fn, optimiser, scaler, device, augment_fn=None):
    """One pass over the training data. Returns the mean training loss."""
    model.train()
    total_loss, n_samples = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if augment_fn is not None:
            images, labels = augment_fn(images, labels)

        optimiser.zero_grad(set_to_none=True)
        # mixed precision: most maths in float16 on the GPU -> faster and about half the memory
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            loss = loss_fn(model(images), labels)
        scaler.scale(loss).backward()  # scaling stops tiny float16 gradients becoming 0
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item() * len(images)
        n_samples += len(images)
    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp):
    """Loss and pooled per-class metrics on a validation or test loader. No augmentation, no gradients."""
    model.eval()
    metrics = SegMetrics(N_CLASSES)
    total_loss, n_samples = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        total_loss += loss_fn(logits.float(), labels).item() * len(images)
        n_samples += len(images)
        metrics.update(logits.argmax(dim=1), labels)
    return total_loss / n_samples, metrics


def plot_curves(history, path, target):
    """Left: train vs validation loss. Right: validation Dice per class (prostate in bold)."""
    import matplotlib
    matplotlib.use("Agg")                       # no screen on Rangpur: draw straight to a file
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, [row["train_loss"] for row in history], label="train")
    ax_loss.plot(epochs, [row["val_loss"] for row in history], label="validation")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("Dice + CE loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    for c in range(1, N_CLASSES):
        ax_dice.plot(epochs, [row[f"dice_{CLASS_NAMES[c]}"] for row in history], label=CLASS_NAMES[c],
                     lw=2.5 if c == PROSTATE else 1.2)
    ax_dice.axhline(target, color="grey", ls="--", lw=1, label=f"target {target}")
    ax_dice.set_ylim(0, 1)
    ax_dice.set_xlabel("epoch")
    ax_dice.set_ylabel("validation Dice")
    ax_dice.set_title("Validation Dice per class")
    ax_dice.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_history(history, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


# 2D Improved U-Net

def build_data_2d(args, pin_memory):
    """Datasets + loaders for the 2D slices, using the provided patient-level split."""
    split = split_cases(args.data_root)
    counts = check_no_leakage(split)                 # raises if any case is in two splits
    print(f"Patient-level split, no leakage: {counts} cases", flush=True)

    train_ds = HipMRISlices(args.data_root, "train", max_scans=args.max_scans)
    val_ds = HipMRISlices(args.data_root, "validate", max_scans=args.max_scans)
    print(train_ds.summary(), "\n" + val_ds.summary(), flush=True)

    sampler = prostate_sampler(train_ds, args.prostate_weight, args.seed) if args.prostate_weight != 1 else None
    train_loader = DataLoader(train_ds, args.batch_size, sampler=sampler, shuffle=sampler is None,
                              num_workers=args.workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory)

    def make_test_loader():                          # loaded only at the end, after training
        test_ds = HipMRISlices(args.data_root, "test", max_scans=args.max_scans)
        return test_ds, DataLoader(test_ds, args.batch_size * 2, shuffle=False, num_workers=args.workers)

    augment_fn = None if args.no_augment else augment_2d
    data_info = {"split_cases": split, "image_shape": list(train_ds.shape)}
    return train_loader, val_loader, make_test_loader, augment_fn, data_info


# 3D Improved U-Net
def build_data_3d(args, pin_memory):

    pairs = list_volume_pairs(args.data_root)
    by_split, groups, warnings = split_volumes(pairs, args.data_root)
    counts = check_no_leakage(groups) 
    print(f"Patient-level split matched to 2D, no leakage: {counts} patients, "
          f"{ {split: len(scans) for split, scans in by_split.items()} } volumes", flush=True)
    for warning in warnings:
        print("  note:", warning, flush=True)
    if args.max_scans:
        by_split = {split: scans[:args.max_scans] for split, scans in by_split.items()}

    centre, box = crop_centre_from_labels(by_split["train"], workers=args.workers)
    print(f"Crop centre {centre} (prostate of every training volume lies in {box})", flush=True)

    def volumes(split, train):
        return HipMRIVolumes(by_split[split], centre, train=train, cache_dir=args.cache_dir)

    train_ds, val_ds = volumes("train", True), volumes("validate", False)
    print(f"train: {train_ds.summary()}\nvalidate: {val_ds.summary()}", flush=True)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=pin_memory, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)

    def make_test_loader(): 
        test_ds = volumes("test", False)
        return test_ds, DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=args.workers)

    augment_fn = None if args.no_augment else augment_3d

    data_info = {"split_patients": groups, "split_scans": {split: [p["id"] for p in scans]
                                                           for split, scans in by_split.items()},
                 "split_warnings": warnings, "crop_centre": centre, "crop_size": list(CROP_3D),
                 "crop_shift": SHIFT_3D, "prostate_box": box, "image_shape": list(CROP_3D)}
    return train_loader, val_loader, make_test_loader, augment_fn, data_info


def main():
    args = get_args()
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    # data
    start_time = time.time()
    build_data = build_data_2d if args.dim == "2d" else build_data_3d
    train_loader, val_loader, make_test_loader, augment_fn, data_info = build_data(args, device.type == "cuda")
    print(f"Data ready in {time.time() - start_time:.0f}s", flush=True)

    # model, loss, optimiser, learning-rate schedule
    dims = 2 if args.dim == "2d" else 3
    model = build_model(args.model, dims, n_classes=N_CLASSES, base_filters=args.base_filters,
                        dropout=args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"Model: {args.model} {args.dim}, {n_params:,} parameters", flush=True)
    loss_fn = DiceCELoss(N_CLASSES, dice_weight=args.dice_weight, ce_weight=args.ce_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = make_grad_scaler(use_amp)

    config = {**vars(args), "dims": dims, "n_classes": N_CLASSES, "n_params": n_params, **data_info}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    history, best_dice, best_epoch, first_epoch = [], -1.0, 0, 1
    if args.resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best_dice, best_epoch = state["history"], state["best_dice"], state["best_epoch"]
        first_epoch = state["epoch"] + 1
        print(f"Resumed after epoch {state['epoch']} (best prostate Dice {best_dice:.4f} at epoch {best_epoch})",
              flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # training loop
    for epoch in range(first_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimiser, scaler, device, augment_fn)
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device, use_amp)
        scheduler.step()
        dice = val_metrics.dice().tolist()

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               **{f"dice_{CLASS_NAMES[c]}": dice[c] for c in range(N_CLASSES)},
               "mean_foreground_dice": sum(dice[1:]) / (N_CLASSES - 1),
               "lr": optimiser.param_groups[0]["lr"],
               "epoch_time_s": time.time() - epoch_start,
               "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0}
        history.append(row)

        improved = dice[PROSTATE] > best_dice
        checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch, "val_dice": dice}
        if improved:
            best_dice, best_epoch = dice[PROSTATE], epoch
            torch.save(checkpoint, out / "best.pt")
        torch.save({**checkpoint, "optimiser": optimiser.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "history": history, "best_dice": best_dice,
                    "best_epoch": best_epoch}, out / "last.pt")
        write_history(history, out / "history.csv")
        plot_curves(history, out / "curves.png", TARGET_DICE[args.dim])

        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | prostate Dice "
              f"{dice[PROSTATE]:.4f}{' *' if improved else '  '} | mean fg Dice {row['mean_foreground_dice']:.4f} | "
              f"{row['epoch_time_s']:.0f}s | {row['peak_vram_mb']:.0f} MB", flush=True)

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs", flush=True)
            break
    print(f"Best validation prostate Dice {best_dice:.4f} at epoch {best_epoch}. "
          f"Total time {(time.time() - start_time) / 60:.1f} min", flush=True)

    # test the best checkpoint
    test_ds, test_loader = make_test_loader()
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["model"])
    _, test_metrics = evaluate(model, test_loader, loss_fn, device, use_amp)
    dice, iou = test_metrics.dice().tolist(), test_metrics.iou().tolist()
    result = {"best_epoch": best_epoch, "test_set": test_ds.summary(),
              "dice": dict(zip(CLASS_NAMES, dice)), "iou": dict(zip(CLASS_NAMES, iou))}
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2))

    print(f"\nTEST ({test_ds.summary()})")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:10s} Dice {dice[c]:.4f}   IoU {iou[c]:.4f}")
    target = TARGET_DICE[args.dim]
    print(f"Prostate Dice {dice[PROSTATE]:.4f}: {'meets' if dice[PROSTATE] >= target else 'below'} "
          f"the {target} target. Saved to {out}", flush=True)


if __name__ == "__main__":
    main()