"""
utils.py - Evaluation metrics and figures shared by predict.py for the 2D and 3D models.

Assisted by AI to make code for neat visualisatin and categorisation of results so comaprison is easier
"""
import csv
import json

import numpy as np
from scipy import ndimage

from dataset import BLADDER, CLASS_NAMES, N_CLASSES, PROSTATE, RECTUM

BAND_NAMES = ["apex", "middle", "base"]
HEAD_FEET_AXIS = 2

# Metrics
def dice_iou(pred, true, n_classes=N_CLASSES):
    """Per-class Dice and IoU for two label maps of any shape. NaN when a class is in neither."""
    pred, true = np.asarray(pred).ravel(), np.asarray(true).ravel()
    dice, iou = np.full(n_classes, np.nan), np.full(n_classes, np.nan)
    for c in range(n_classes):
        p, t = pred == c, true == c
        overlap, n_pred, n_true = np.logical_and(p, t).sum(), p.sum(), t.sum()
        if n_pred + n_true > 0:
            dice[c] = 2 * overlap / (n_pred + n_true)
            iou[c] = overlap / (n_pred + n_true - overlap)
    return dice, iou


def surface_distances(pred_mask, true_mask, spacing):
    """HD95 and average symmetric surface distance (mm) between two boolean masks.

    surface = mask minus its erosion (the outermost layer of voxels)
    For every surface voxel of one mask: distance to the nearest surface voxel of the other
    (distance_transform_edt with the voxel spacing gives real mm).
    HD95 = the 95th percentile of those distances (robust version of the worst-case error).
    Returns (None, None) if either mask is empty.
    """
    if not pred_mask.any() or not true_mask.any():
        return None, None
    # work only on the region around both masks (much faster than the whole block)
    box = tuple(slice(max(a.min() - 2, 0), a.max() + 3) for a in np.nonzero(pred_mask | true_mask))
    pred_mask, true_mask = pred_mask[box], true_mask[box]
    structure = ndimage.generate_binary_structure(3, 1)
    pred_surface = pred_mask & ~ndimage.binary_erosion(pred_mask, structure)
    true_surface = true_mask & ~ndimage.binary_erosion(true_mask, structure)
    pred_to_true = ndimage.distance_transform_edt(~true_surface, sampling=spacing)[pred_surface]
    true_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[true_surface]
    hd95 = max(np.percentile(pred_to_true, 95), np.percentile(true_to_pred, 95))
    asd = np.concatenate([pred_to_true, true_to_pred]).mean()
    return float(hd95), float(asd)


def band_masks(true_labels, fraction=0.2, axis=HEAD_FEET_AXIS):
    """Split the head-feet axis into apex / middle / base bands using the TRUE prostate only.

    The true prostate spans indices z0..z1 along `axis`. The 20% at the end nearest the bladder
    is the base, the 20% at the other end is the apex, the rest is the middle. Everything beyond
    either end belongs to that end's band, so over-shooting predictions are counted there.
    Returns ([apex, middle, base] masks, base_is_high_end, (z0, z1)) or None without prostate.
    """
    other_axes = tuple(a for a in range(true_labels.ndim) if a != axis)
    prostate_z = np.nonzero((true_labels == PROSTATE).any(axis=other_axes))[0]
    if prostate_z.size == 0:
        return None
    z0, z1 = int(prostate_z[0]), int(prostate_z[-1])
    band = max(1, int(round(fraction * (z1 - z0 + 1))))
    coords = np.arange(true_labels.shape[axis])
    low_end, high_end = coords < z0 + band, coords > z1 - band
    middle = ~low_end & ~high_end

    bladder_z = np.nonzero((true_labels == BLADDER).any(axis=other_axes))[0]
    base_is_high = bool(prostate_z.mean() < bladder_z.mean()) if bladder_z.size else True
    apex, base = (low_end, high_end) if base_is_high else (high_end, low_end)

    shape = [1] * true_labels.ndim
    shape[axis] = -1                               # so the 1D masks broadcast over the block
    return [m.reshape(shape) for m in (apex, middle, base)], base_is_high, (z0, z1)


def evaluate_scan(pred, true, spacing, band_fraction=0.2):
    """All metrics for one scan/volume. Returns (row dict, slice profile list)."""
    voxel_ml = float(np.prod(spacing)) / 1000.0          # mm^3 -> ml
    dice, iou = dice_iou(pred, true)
    pred_pro, true_pro = pred == PROSTATE, true == PROSTATE
    hd95, asd = surface_distances(pred_pro, true_pro, spacing)
    row = {f"dice_{n}": float(dice[c]) for c, n in enumerate(CLASS_NAMES)}
    row.update({f"iou_{n}": float(iou[c]) for c, n in enumerate(CLASS_NAMES)})
    row.update(hd95_mm=hd95, asd_mm=asd,
               true_prostate_ml=float(true_pro.sum() * voxel_ml), pred_prostate_ml=float(pred_pro.sum() * voxel_ml),
               leak_rectum_ml=float((pred_pro & (true == RECTUM)).sum() * voxel_ml),
               leak_bladder_ml=float((pred_pro & (true == BLADDER)).sum() * voxel_ml))

    profile = []
    bands = band_masks(true, band_fraction)
    if bands:
        masks, base_is_high, (z0, z1) = bands
        for name, mask in zip(BAND_NAMES, masks):
            p, t = pred_pro & mask, true_pro & mask
            total = p.sum() + t.sum()
            row[f"dice_{name}"] = float(2 * (p & t).sum() / total) if total else float("nan")
        # Dice on every slice across the head-feet axis; position 0 = apex end, 1 = base end
        for z in range(z0, z1 + 1):
            p, t = pred_pro[:, :, z], true_pro[:, :, z]
            position = (z - z0) / max(z1 - z0, 1)
            profile.append((position if base_is_high else 1 - position,
                            float(2 * (p & t).sum() / max(p.sum() + t.sum(), 1))))
    return row, profile


def summarise(values):
    """mean / sd / median / min / max of a list, ignoring None and NaN."""
    a = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if a.size == 0:
        return None
    return {"mean": float(a.mean()), "sd": float(a.std()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def write_csv(rows, path):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=float)

# Figures
def _plt():
    import matplotlib
    matplotlib.use("Agg") 
    import matplotlib.pyplot as plt
    return plt


def overlay(ax, image, labels, title):
    """Grey image with coloured labels on top (labels=None -> image only)."""
    plt = _plt()
    ax.imshow(image, cmap="gray", origin="lower")
    if labels is not None:
        ax.imshow(np.ma.masked_equal(labels, 0), cmap=plt.get_cmap("tab10", 6), vmin=0, vmax=5,
                  alpha=0.45, origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def failure_figure(cases, path):
    """One row per case: image | ground truth | prediction | errors (red = missed, blue = false prostate).

    cases: list of dicts with keys title, image, true, pred (2D arrays, already oriented for display),
           dice, and optional band_lines (row positions of the apex/base band edges).
    """
    plt = _plt()
    fig, axes = plt.subplots(len(cases), 4, figsize=(12, 1.8 * len(cases) + 0.4), squeeze=False)
    for row_axes, case in zip(axes, cases):
        image, true, pred = case["image"], case["true"], case["pred"]
        overlay(row_axes[0], image, None, case["title"])
        overlay(row_axes[1], image, true, "ground truth")
        overlay(row_axes[2], image, pred, f"prediction (scan prostate Dice {case['dice']:.2f})")
        missed = (true == PROSTATE) & (pred != PROSTATE)
        false_pos = (pred == PROSTATE) & (true != PROSTATE)
        errors = np.zeros(true.shape + (4,))
        errors[missed] = (1, 0.1, 0.1, 0.85)
        errors[false_pos] = (0.1, 0.5, 1, 0.85)
        row_axes[3].imshow(image, cmap="gray", origin="lower")
        row_axes[3].imshow(errors, origin="lower", interpolation="nearest")
        for y in case.get("band_lines", []):       # dashed lines = edges of the apex/base bands
            row_axes[3].axhline(y, color="yellow", ls="--", lw=0.8)
        row_axes[3].set_title(f"red = missed ({missed.sum()} px), blue = false ({false_pos.sum()} px)", fontsize=8)
        row_axes[3].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def band_boxplot(rows, path, target):
    """Per-scan prostate Dice in each band and for the whole prostate, plus HD95."""
    plt = _plt()
    fig, (ax_dice, ax_hd) = plt.subplots(1, 2, figsize=(10, 3.8))
    data = [[r[f"dice_{b}"] for r in rows if not np.isnan(r.get(f"dice_{b}", np.nan))] for b in BAND_NAMES]
    data.append([r["dice_prostate"] for r in rows if not np.isnan(r["dice_prostate"])])
    ax_dice.boxplot(data)
    ax_dice.set_xticks(range(1, 5))
    ax_dice.set_xticklabels(BAND_NAMES + ["whole"])      # (two calls work on older Matplotlib too)
    for k, values in enumerate(data, 1):         # show every scan as a dot (few test scans)
        ax_dice.scatter(np.full(len(values), k) + np.random.uniform(-0.08, 0.08, len(values)), values, s=12, alpha=0.7)
    ax_dice.axhline(target, color="grey", ls="--", lw=1)
    ax_dice.set_ylim(0, 1)
    ax_dice.set_ylabel("prostate Dice per scan")
    ax_dice.set_title("Prostate Dice by band")
    hd = [r["hd95_mm"] for r in rows if r["hd95_mm"] is not None]
    ax_hd.boxplot([hd])
    ax_hd.set_xticks([1])
    ax_hd.set_xticklabels(["prostate"])
    ax_hd.scatter(np.ones(len(hd)) + np.random.uniform(-0.08, 0.08, len(hd)), hd, s=12, alpha=0.7)
    ax_hd.set_ylabel("HD95 (mm)")
    ax_hd.set_title("Prostate HD95 per scan")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def profile_plot(profiles, path, n_bins=10):
    """Mean +- sd of slice-wise prostate Dice against position from apex (0) to base (1)."""
    plt = _plt()
    edges = np.linspace(0, 1, n_bins + 1)
    per_scan = []
    for profile in profiles:
        position = np.array([p for p, _ in profile])
        dice = np.array([d for _, d in profile])
        bin_of = np.clip(np.digitize(position, edges) - 1, 0, n_bins - 1)
        per_scan.append([dice[bin_of == k].mean() if (bin_of == k).any() else np.nan for k in range(n_bins)])
    per_scan = np.array(per_scan)
    mean, sd = np.nanmean(per_scan, 0), np.nanstd(per_scan, 0)
    centres = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.axvspan(0, 0.2, color="#f9ece7")
    ax.axvspan(0.8, 1, color="#e7f1fb")
    ax.fill_between(centres, mean - sd, mean + sd, alpha=0.25)
    ax.plot(centres, mean, marker="o")
    ax.text(0.1, 0.03, "apex", ha="center")
    ax.text(0.9, 0.03, "base", ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("position along the prostate (0 = apex, 1 = base)")
    ax.set_ylabel("slice prostate Dice")
    ax.set_title(f"Where along the prostate the model fails ({len(profiles)} scans, mean ± sd)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"position": centres.tolist(), "mean_dice": mean.tolist(), "sd": sd.tolist()}