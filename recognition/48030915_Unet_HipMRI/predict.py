# predict.py
# Example usage of the trained model: load saved weights, run inference on test cases,
# print results, save visualisations. NumPy is allowed here (spec)


# SHARED

# get_args(): --checkpoint, --split test, --out, --n-failures 4,
#             3D only: --full-volume, --keep-largest

# load_model(checkpoint) -> model, settings
#   - torch.load -> rebuild with build_model(settings) -> load_state_dict -> model.eval()

# metrics per scan/volume
#   - dice_iou(pred, true) per class
#   - surface_distances(pred_prostate, true_prostate, spacing) -> HD95 and average distance in mm
#       (scipy.ndimage: surface = mask minus eroded mask; distance_transform_edt with voxel spacing)
#   - band masks along the head-feet axis (axis 2): from the TRUE prostate's extent,
#       lowest 20% = apex, top 20% = base (the end next to the bladder), rest = middle
#       -> prostate Dice per band
#   - slice profile: Dice on each slice vs position from apex (0) to base (1)
#   - leakage: predicted prostate inside the true rectum / bladder, in ml (voxel volume from spacing)

# print_results(): pooled Dice/IoU per class; per scan mean +- sd; band Dice; HD95 median;
#                  leakage; inference time; peak VRAM; parameters; "meets target?"

# save: results.json (all numbers), per_scan.csv (one row per scan)

# visualisations
#   - failure_cases.png: worst scans by prostate Dice, at the slice with the most errors:
#       image | ground truth | prediction | errors (red = missed, blue = false prostate)
#   - best_case.png
#   - band_boxplot.png: per-scan Dice for apex / middle / base / whole, and HD95
#   - slice_profile.png: where along the prostate the model fails

# compare(results_2d, results_3d) -> the 2D-vs-3D table for the README
#   (same test patients by design)


# 2D Model specifics
#   - run the model on every test slice (batches), timing per slice
#   - STACK each scan's 60 predicted slices back into a (60, 256, 128) block, so the
#     2D model is scored per scan with the same 3D metrics as the 3D model
#   - voxel spacing: from the matching 3D volume header (case_006 week 0 -> B006_Week0)

# 3D Model specifics
#   - default: run on the central 128^3 crop the model was trained on
#   - --full-volume: Gaussian-weighted sliding window over the whole 256x256x128 scan
#       (needed for a fair comparison: the 2D model sees whole slices)
#   - --keep-largest: keep only the largest connected prostate region (removes stray blobs)
#   - timing per volume, peak inference VRAM

