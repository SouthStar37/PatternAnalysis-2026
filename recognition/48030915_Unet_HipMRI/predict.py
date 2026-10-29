"""
predict.py - Load a trained model, run it on the test cases, print the results and save figures.

Run:
    python predict.py --checkpoint runs/improved2d/best.pt [--split validate]
    python predict.py --compare A/results.json B/results.json     # README comparison table

Outputs in <checkpoint folder>/eval_<split>: results.json, per_scan.csv, failure_cases.png,
best_case.png, band_boxplot.png, slice_profile.png.

2D: each scan's predicted slices are stacked back into a (slices, 256, 128) block, so the 2D
model is scored per scan with the same metrics as the 3D model (axis 2 = head-feet).
"""
import argparse
import json
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from dataset import CLASS_NAMES, IMAGE_DIR_3D, PROSTATE, HipMRISlices
from modules import SegMetrics, build_model, count_parameters
from utils import BAND_NAMES, band_boxplot, band_masks, evaluate_scan, failure_figure, profile_plot, save_json, \
    summarise, write_csv

TARGET_DICE = {2: 0.75, 3: 0.70}
METRICS = ["dice_prostate", "iou_prostate", "dice_apex", "dice_middle", "dice_base", "hd95_mm", "asd_mm",
           "leak_rectum_ml", "leak_bladder_ml", "dice_bladder", "dice_rectum"]
PATIENT_METRICS = ["dice_prostate", "hd95_mm", "dice_apex", "dice_middle", "dice_base"]


def get_args():
    ap = argparse.ArgumentParser(description="Evaluate a trained HipMRI model")
    ap.add_argument("--checkpoint", help="path to best.pt")
    ap.add_argument("--split", default="test", choices=["train", "validate", "test"])
    ap.add_argument("--data-root", help="default: the one used for training")
    ap.add_argument("--out")
    ap.add_argument("--n-failures", type=int, default=4, help="how many worst scans to plot")
    ap.add_argument("--band-fraction", type=float, default=0.2, help="size of the apex and base bands")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-scans", type=int, help="quick test on a few scans")
    ap.add_argument("--compare", nargs=2, metavar=("RESULTS_A", "RESULTS_B"), help="compare two results.json")
    args = ap.parse_args()
    if not (args.compare or args.checkpoint):
        ap.error("--checkpoint is required (or use --compare)")
    return args


def load_model(path, device):
    """Rebuild the network from the settings saved in the checkpoint and load its weights."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], cfg["dims"], n_classes=cfg["n_classes"], base_filters=cfg["base_filters"],
                        dropout=cfg["dropout"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt["epoch"]


def spacing_lookup(root):
    """(patient number, week) -> voxel spacing (mm) from the 3D headers (case_006 week 0 <-> B006_Week0)."""
    lookup = {}
    for path in (Path(root) / IMAGE_DIR_3D).glob("*.nii*"):
        m = re.search(r"(\d+)_week_?(\d+)", path.name, re.I)
        if m:
            lookup[(int(m[1]), int(m[2]))] = tuple(float(z) for z in nib.load(str(path)).header.get_zooms()[:3])
    return lookup


def mean_sd(stats, fmt=".3f"):
    return f"{stats['mean']:{fmt}} ± {stats['sd']:{fmt}}" if stats else "n/a"


def print_results(s):
    ps = s["per_scan"]
    print(f"\n=== {s['model']} on {s['split']}: {s['n_scans']} scans, {s['n_patients']} patients ===")
    print("Pooled Dice: " + ", ".join(f"{k} {v:.3f}" for k, v in s["pooled_dice"].items()))
    print("Pooled IoU:  " + ", ".join(f"{k} {v:.3f}" for k, v in s["pooled_iou"].items()))
    print(f"Prostate Dice {s['pooled_dice']['prostate']:.3f} ({'meets' if s['meets_target'] else 'below'} the "
          f"{s['target_dice']} target); per scan {mean_sd(ps['dice_prostate'])}")
    for band in BAND_NAMES:
        print(f"  {band:6s} band Dice {mean_sd(ps[f'dice_{band}'])}")
    if ps["hd95_mm"]:
        print(f"HD95 median {ps['hd95_mm']['median']:.1f} mm (range {ps['hd95_mm']['min']:.1f}-{ps['hd95_mm']['max']:.1f})")
    print(f"Leakage per scan: rectum {mean_sd(ps['leak_rectum_ml'], '.2f')} ml, "
          f"bladder {mean_sd(ps['leak_bladder_ml'], '.2f')} ml")
    print(f"Inference {s['inference_ms_per_slice']:.2f} ms/slice, {s['inference_s_per_scan']:.2f} s/scan, "
          f"peak VRAM {s['peak_inference_vram_mb']:.0f} MB, {s['n_parameters']:,} parameters")


def compare(path_a, path_b):
    """Markdown table of two results.json files (e.g. 2D vs 3D on the same test patients)."""
    cols = [("Per scan", "dice_prostate", "mean", ".3f"), ("Apex", "dice_apex", "mean", ".3f"),
            ("Middle", "dice_middle", "mean", ".3f"), ("Base", "dice_base", "mean", ".3f"),
            ("HD95 median (mm)", "hd95_mm", "median", ".1f"), ("Rectum leak (ml)", "leak_rectum_ml", "mean", ".2f"),
            ("Bladder leak (ml)", "leak_bladder_ml", "mean", ".2f")]
    lines = ["| Model | Prostate Dice (pooled) | " + " | ".join(c[0] for c in cols) + " | Parameters | Inference / scan (s) |",
             "| --- |" + " --- |" * (len(cols) + 3)]
    for r in (json.load(open(p)) for p in (path_a, path_b)):
        cells = [format(r["per_scan"][k][stat], f) if r["per_scan"].get(k) else "n/a" for _, k, stat, f in cols]
        lines.append(f"| {r['model']} | {r['pooled_dice']['prostate']:.3f} | " + " | ".join(cells) +
                     f" | {r['n_parameters']:,} | {r['inference_s_per_scan']:.2f} |")
    table = "\n".join(lines)
    out = Path(path_a).parent / "comparison.md"
    out.write_text(table + "\n")
    print(f"{table}\n\nSaved to {out}")

# 2D model
@torch.no_grad()
def evaluate_2d(model, cfg, args, device):
    """Predict every slice, then score each scan (its slices stacked into one block)."""
    root = args.data_root or cfg["data_root"]
    ds = HipMRISlices(root, args.split, shape=tuple(cfg["image_shape"]), max_scans=args.max_scans)
    print(f"Loaded {ds.summary()}", flush=True)

    gpu = device.type == "cuda"
    if gpu:                                   # warm-up, so the first batch doesn't distort the timing
        model(torch.from_numpy(ds.images[:2]).unsqueeze(1).to(device))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    preds = np.zeros(ds.labels.shape, dtype=np.uint8)
    start = time.perf_counter()
    for i in range(0, len(ds), args.batch_size):
        x = torch.from_numpy(ds.images[i:i + args.batch_size]).unsqueeze(1).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=gpu):
            preds[i:i + args.batch_size] = model(x).argmax(1).cpu().numpy()
    if gpu:
        torch.cuda.synchronize()              # wait for the GPU before stopping the clock
    seconds = time.perf_counter() - start

    spacings, rows, profiles = spacing_lookup(root), [], []
    for (case, week), idx in ds.scan_index.items():
        spacing = spacings.get((case, week), (1.0, 1.0, 1.0))
        row, profile = evaluate_scan(preds[idx], ds.labels[idx], spacing, args.band_fraction)
        rows.append({"case": case, "week": week, "n_slices": len(idx), "spacing_mm": "x".join(f"{s:.3f}" for s in spacing),
                     "spacing_source": "3D header" if (case, week) in spacings else "not found: voxels", **row})
        profiles += [profile] if profile else []
    timing = {"inference_ms_per_slice": 1000 * seconds / len(ds), "inference_s_per_scan": seconds / len(rows),
              "peak_inference_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if gpu else 0.0}
    return ds, preds, rows, profiles, timing


def figure_cases_2d(ds, preds, picked, band_fraction):
    """For each picked scan: its slice with the most prostate errors, head-up, with apex/base band edges."""
    cases = []
    for r in picked:
        idx = ds.scan_index[(r["case"], r["week"])]
        i = idx[int(np.argmax([((preds[j] == PROSTATE) != (ds.labels[j] == PROSTATE)).sum() for j in idx]))]
        bands, lines = band_masks(ds.labels[idx], band_fraction), []
        if bands:
            z0, z1 = bands[2]
            width = max(1, round(band_fraction * (z1 - z0 + 1)))
            lines = [z0 + width - 0.5, z1 - width + 0.5]
        cases.append({"title": f"case {r['case']} week {r['week']} slice {ds.keys[i][2]}", "image": ds.images[i].T,
                      "true": ds.labels[i].T, "pred": preds[i].T, "dice": r["dice_prostate"], "band_lines": lines})
    return cases

# main
def main():
    args = get_args()
    if args.compare:
        return compare(*args.compare)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, epoch = load_model(args.checkpoint, device)
    dims = cfg["dims"]
    out = Path(args.out or Path(args.checkpoint).parent / f"eval_{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Model {cfg['model']} {dims}D from epoch {epoch} ({count_parameters(model):,} parameters)", flush=True)
    if dims != 2:
        raise NotImplementedError("3D evaluation is added after the 3D model is trained")

    ds, preds, rows, profiles, timing = evaluate_2d(model, cfg, args, device)
    pooled = SegMetrics()                                   # Dice / IoU over every test pixel
    pooled.update(torch.from_numpy(preds).long(), torch.from_numpy(ds.labels).long())
    pooled_dice = pooled.dice().tolist()

    summary = {
        "model": f"{cfg['model']} {dims}D", "checkpoint": str(args.checkpoint), "epoch": epoch, "split": args.split,
        "n_parameters": count_parameters(model), "n_scans": len(rows), "n_patients": len({r["case"] for r in rows}),
        "pooled_dice": {n: round(d, 4) for n, d in zip(CLASS_NAMES, pooled_dice)},
        "pooled_iou": {n: round(v, 4) for n, v in zip(CLASS_NAMES, pooled.iou().tolist())},
        "per_scan": {k: summarise([r.get(k) for r in rows]) for k in METRICS},
        "per_patient": {str(c): {k: (summarise([r.get(k) for r in rows if r["case"] == c]) or {}).get("mean")
                                 for k in PATIENT_METRICS} for c in sorted({r["case"] for r in rows})},
        "slice_profile": profile_plot(profiles, out / "slice_profile.png") if profiles else None,
        **timing, "device": str(device),
        "target_dice": TARGET_DICE[dims], "meets_target": pooled_dice[PROSTATE] >= TARGET_DICE[dims],
    }
    save_json(summary, out / "results.json")
    write_csv(rows, out / "per_scan.csv")

    ranked = sorted((r for r in rows if not np.isnan(r["dice_prostate"])), key=lambda r: r["dice_prostate"])
    if ranked:
        failure_figure(figure_cases_2d(ds, preds, ranked[:args.n_failures], args.band_fraction), out / "failure_cases.png")
        failure_figure(figure_cases_2d(ds, preds, ranked[-1:], args.band_fraction), out / "best_case.png")
    band_boxplot(rows, out / "band_boxplot.png", TARGET_DICE[dims])
    print_results(summary)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

# 3D Model specifics
#   - default: run on the central 128^3 crop the model was trained on
#   - --full-volume: Gaussian-weighted sliding window over the whole 256x256x128 scan
#       (needed for a fair comparison: the 2D model sees whole slices)
#   - --keep-largest: keep only the largest connected prostate region (removes stray blobs)
#   - timing per volume, peak inference VRAM

