"""
predict.py - Load a trained model, run it on the test cases, print the results and save figures.

Run:
    python predict.py --checkpoint runs/improved2d/best.pt [--split validate]
    python predict.py --compare A/results.json B/results.json 
    python predict.py --checkpoint runs/improved3d/best.pt [--split validate]

Outputs in <checkpoint folder>/eval_<split>: results.json, per_scan.csv, failure_cases.png,
best_case.png, band_boxplot.png, slice_profile.png.
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from dataset import CLASS_NAMES, IMAGE_DIR_3D, PROSTATE, HipMRISlices, HipMRIVolumes, list_volume_pairs, load_volume
from modules import SegMetrics, build_model, count_parameters
from utils import BAND_NAMES, band_boxplot, band_masks, evaluate_scan, failure_figure, profile_comparison_plot, \
    profile_plot, save_json, summarise, write_csv

TARGET_DICE = {2: 0.75, 3: 0.70}
METRICS = ["dice_prostate", "iou_prostate", "dice_apex", "dice_middle", "dice_base", "hd95_mm", "asd_mm",
           "leak_rectum_ml", "leak_bladder_ml", "dice_bladder", "dice_rectum", "prostate_pieces", "stray_prostate_ml"]
PATIENT_METRICS = ["dice_prostate", "hd95_mm", "dice_apex", "dice_middle", "dice_base"]

SETTINGS = {
    "data_root": None,      # None = the data folder saved in the checkpoint
    "n_failures": 4,        # how many worst scans failure_cases.png shows
    "band_fraction": 0.2,   # apex and base bands: 20% of the prostate's head-feet length each
    "batch_size": 32,       # 2D slices per forward pass
}

def get_args():
    ap = argparse.ArgumentParser(description="Evaluate a trained HipMRI model")
    ap.add_argument("--checkpoint", help="path to best.pt")
    ap.add_argument("--split", default="test", choices=["train", "validate", "test"])
    ap.add_argument("--max-scans", type=int, help="quick test on a few scans")
    ap.add_argument("--compare", nargs=2, metavar=("RESULTS_A", "RESULTS_B"), help="compare two results.json")
    args = ap.parse_args()
    if not (args.compare or args.checkpoint):
        ap.error("--checkpoint is required (or use --compare)")
    vars(args).update(SETTINGS)
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
    if ps.get("prostate_pieces"):
        print(f"Prostate pieces per scan {mean_sd(ps['prostate_pieces'], '.1f')} (truth: 1), "
              f"stray prostate outside the main piece {mean_sd(ps['stray_prostate_ml'], '.2f')} ml")
    per_slice = f"{s['inference_ms_per_slice']:.2f} ms/slice, " if s.get("inference_ms_per_slice") else ""
    print(f"Inference {per_slice}{s['inference_s_per_scan']:.2f} s/scan, "
          f"peak VRAM {s['peak_inference_vram_mb']:.0f} MB, {s['n_parameters']:,} parameters")
    if s.get("prostate_voxels_outside_crop"):
        print(f"WARNING: {s['prostate_voxels_outside_crop']} true prostate voxels lie outside the 3D crop")

def read_per_scan(results_path):
    with open(Path(results_path).parent / "per_scan.csv", newline="") as f:
        return {(int(r["case"]), int(r["week"])): r for r in csv.DictReader(f)}


def as_number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def compare(path_a, path_b):
    results = [json.load(open(p)) for p in (path_a, path_b)]
    names = [r["model"] for r in results]
    short = [n.split()[-1] for n in names] if names[0] != names[1] else ["A", "B"]   # e.g. "2D", "3D"

    # summary
    cols = [("Per scan", "dice_prostate", "mean", ".3f"), ("Apex", "dice_apex", "mean", ".3f"),
            ("Middle", "dice_middle", "mean", ".3f"), ("Base", "dice_base", "mean", ".3f"),
            ("HD95 median (mm)", "hd95_mm", "median", ".1f"), ("Rectum leak (ml)", "leak_rectum_ml", "mean", ".2f"),
            ("Bladder leak (ml)", "leak_bladder_ml", "mean", ".2f"), ("Pieces", "prostate_pieces", "mean", ".1f"),
            ("Stray (ml)", "stray_prostate_ml", "mean", ".2f")]
    lines = ["## Summary", "",
             "| Model | Prostate Dice (pooled) | " + " | ".join(c[0] for c in cols) + " | Parameters | Inference / scan (s) |",
             "| --- |" + " --- |" * (len(cols) + 3)]
    for r in results:
        cells = [format(r["per_scan"][k][stat], f) if r["per_scan"].get(k) else "n/a" for _, k, stat, f in cols]
        lines.append(f"| {r['model']} | {r['pooled_dice']['prostate']:.3f} | " + " | ".join(cells) +
                     f" | {r['n_parameters']:,} | {r['inference_s_per_scan']:.2f} |")

    # paired per-scan table (only scans both models were evaluated on)
    scans = [read_per_scan(p) for p in (path_a, path_b)]
    shared = sorted(set(scans[0]) & set(scans[1]))
    metrics = [("Dice", "dice_prostate", ".3f"), ("Apex", "dice_apex", ".3f"), ("Base", "dice_base", ".3f"),
               ("HD95 (mm)", "hd95_mm", ".1f"), ("Pieces", "prostate_pieces", ".0f")]
    lines += ["", f"## Paired per-scan comparison ({len(shared)} scans evaluated by both models)", "",
              "| Case | Week | " + " | ".join(f"{m} {s}" for m, _, _ in metrics for s in short) +
              f" | Dice {short[1]} - {short[0]} |",
              "| --- | --- |" + " --- |" * (2 * len(metrics) + 1)]
    differences = []
    for case, week in shared:
        a, b = scans[0][(case, week)], scans[1][(case, week)]
        cells = [format(as_number(row.get(key)), f) for _, key, f in metrics for row in (a, b)]
        difference = as_number(b["dice_prostate"]) - as_number(a["dice_prostate"])
        differences.append(difference)
        lines.append(f"| {case} | {week} | " + " | ".join(cells).replace("nan", "-") + f" | {difference:+.3f} |")
    if shared:
        valid = [d for d in differences if d == d]
        lines += ["", f"{short[1]} has the higher prostate Dice on {sum(d > 0 for d in valid)} of {len(valid)} scans "
                      f"(mean difference {sum(valid) / max(len(valid), 1):+.3f})."]
    missing = (len(scans[0]) - len(shared), len(scans[1]) - len(shared))
    if any(missing):
        lines += ["", f"Note: {missing[0]} scans only in {names[0]}, {missing[1]} only in {names[1]} (left out above)."]

    # per patient
    patients = sorted(set(results[0]["per_patient"]) & set(results[1]["per_patient"]), key=int)
    lines += ["", "## Per patient (mean over that patient's scans)", "",
              "| Case | " + " | ".join(f"{m} {s}" for m, _, _ in metrics[:3] for s in short) + " |",
              "| --- |" + " --- |" * (2 * 3)]
    for case in patients:
        cells = [results[k]["per_patient"][case].get(key) for _, key, _ in metrics[:3] for k in (0, 1)]
        lines.append(f"| {case} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in cells) + " |")

    out = Path(path_a).parent
    table = "\n".join(lines)
    (out / "comparison.md").write_text(table + "\n")
    profiles = {n: r["slice_profile"] for n, r in zip(names, results) if r.get("slice_profile")}
    if profiles:
        profile_comparison_plot(profiles, out / "profile_comparison.png")
    print(f"{table}\n\nSaved comparison.md and profile_comparison.png to {out}")

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

def band_lines(true_block, band_fraction):
    """Row positions of the apex/base band edges, drawn as dashed lines on the error panel."""
    bands = band_masks(true_block, band_fraction)
    if not bands:
        return []
    z0, z1 = bands[2]
    width = max(1, round(band_fraction * (z1 - z0 + 1)))
    return [z0 + width - 0.5, z1 - width + 0.5]

def figure_cases_2d(ds, preds, picked, band_fraction, show="errors"):
    cases = []
    for r in picked:
        idx = ds.scan_index[(r["case"], r["week"])]
        if show == "errors":
            scores = [((preds[j] == PROSTATE) != (ds.labels[j] == PROSTATE)).sum() for j in idx]
        else:
            scores = [(ds.labels[j] == PROSTATE).sum() for j in idx]
        i = idx[int(np.argmax(scores))]
        cases.append({"title": f"case {r['case']} week {r['week']} slice {ds.keys[i][2]}", "image": ds.images[i].T,
                      "true": ds.labels[i].T, "pred": preds[i].T, "dice": r["dice_prostate"],
                      "band_lines": band_lines(ds.labels[idx], band_fraction)})
    return cases

# 3D model
@torch.no_grad()
def evaluate_3d(model, cfg, args, device):
    root = args.data_root or cfg["data_root"]
    wanted = set(cfg["split_scans"][args.split])
    pairs = [p for p in list_volume_pairs(root) if p["id"] in wanted]
    if args.max_scans:
        pairs = pairs[:args.max_scans]
    cache = None if args.max_scans else cfg.get("cache_dir")
    ds = HipMRIVolumes(pairs, cfg["crop_centre"], crop=tuple(cfg["crop_size"]), shift=cfg["crop_shift"],
                       cache_dir=cache)
    print(f"Loaded {args.split}: {ds.summary()}", flush=True)

    gpu = device.type == "cuda"
    if gpu:
        model(ds[0][0].unsqueeze(0).to(device))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    images, labels, preds, rows, profiles, seconds = [], [], [], [], [], 0.0
    for i, pair in enumerate(pairs):
        image, label = ds[i]
        start = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.float16, enabled=gpu):
            pred = model(image.unsqueeze(0).to(device)).argmax(1)[0]
        if gpu:
            torch.cuda.synchronize()
        seconds += time.perf_counter() - start
        pred, label = pred.cpu().numpy().astype(np.uint8), label.numpy().astype(np.uint8)

        spacing = ds.meta[i]["spacing"]
        row, profile = evaluate_scan(pred, label, spacing, args.band_fraction)
        full_label, _ = load_volume(pair["label"], is_label=True)
        outside = int((full_label == PROSTATE).sum() - (label == PROSTATE).sum())
        rows.append({"case": pair["patient_num"], "week": pair["week"], "scan": pair["id"],
                     "spacing_mm": "x".join(f"{s:.3f}" for s in spacing), "prostate_voxels_outside_crop": outside,
                     **row})
        profiles += [profile] if profile else []
        images.append(image[0].numpy())
        labels.append(label)
        preds.append(pred)
    timing = {"inference_ms_per_slice": None, "inference_s_per_scan": seconds / len(rows),
              "peak_inference_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if gpu else 0.0}
    return images, labels, preds, rows, profiles, timing


def figure_cases_3d(images, labels, preds, rows, picked, band_fraction, show="errors"):
    cases = []
    for r in picked:
        k = rows.index(r)
        image, true, pred = images[k], labels[k], preds[k]
        if show == "errors":
            scores = ((pred == PROSTATE) != (true == PROSTATE)).sum(axis=(0, 2))
        else:
            scores = (true == PROSTATE).sum(axis=(0, 2))
        y = int(np.argmax(scores))
        cases.append({"title": f"{r['scan']} coronal slice {y}", "image": image[:, y, :].T,
                      "true": true[:, y, :].T, "pred": pred[:, y, :].T, "dice": r["dice_prostate"],
                      "band_lines": band_lines(true, band_fraction)})
    return cases

# main
def main():
    args = get_args()
    if args.compare:
        return compare(*args.compare)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, epoch = load_model(args.checkpoint, device)
    dims = cfg["dims"]
    out = Path(args.checkpoint).parent / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Model {cfg['model']} {dims}D from epoch {epoch} ({count_parameters(model):,} parameters)", flush=True)
    if dims == 2:
        ds, preds, rows, profiles, timing = evaluate_2d(model, cfg, args, device)
        all_preds, all_labels = preds, ds.labels

        def make_cases(picked, show="errors"):
            return figure_cases_2d(ds, preds, picked, args.band_fraction, show)
    else:
        images, labels, preds, rows, profiles, timing = evaluate_3d(model, cfg, args, device)
        all_preds, all_labels = np.stack(preds), np.stack(labels)

        def make_cases(picked, show="errors"):
            return figure_cases_3d(images, labels, preds, rows, picked, args.band_fraction, show)

    pooled = SegMetrics()  
    pooled.update(torch.from_numpy(all_preds).long(), torch.from_numpy(all_labels).long())
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
        "evaluated_on": "whole 2D slices" if dims == 2 else f"central {tuple(cfg['crop_size'])} crop",
        "prostate_voxels_outside_crop": sum(r.get("prostate_voxels_outside_crop", 0) for r in rows),
    }
    save_json(summary, out / "results.json")
    write_csv(rows, out / "per_scan.csv")

    ranked = sorted((r for r in rows if not np.isnan(r["dice_prostate"])), key=lambda r: r["dice_prostate"])
    if ranked:
        failure_figure(make_cases(ranked[:args.n_failures]), out / "failure_cases.png")
        failure_figure(make_cases(ranked[-1:], show="prostate"), out / "best_case.png")
    band_boxplot(rows, out / "band_boxplot.png", TARGET_DICE[dims])
    print_results(summary)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
