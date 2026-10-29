# dataset.py

# Labels: 0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate

# SHARED (used by both 2D and 3D)

# constants
#   - DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open"
#   - folder names for the 2D slices and the 3D volumes
#   - CLASS_NAMES, N_CLASSES = 6, PROSTATE = 5, BLADDER = 3, RECTUM = 4

# load_nifti(path) -> numpy array
#   - nibabel.load(path), np.asanyarray(img.dataobj), np.squeeze (drops extra size-1 dims)
#   - labels: np.rint(...).astype(uint8) (stored as float in some files)

# check_no_leakage({split: patient_ids}) -> raises an error if a patient is in 2+ splits


# 2D HipMRI dataset (keras_slices_data)

# ---- data loader -----
# parse_key(filename) -> (case, week, slice)
#   - regex on "case_004_week_0_slice_12.nii.gz" / "seg_004_week_0_slice_12.nii.gz"

# list_pairs(root, split) -> sorted [(key, image_path, label_path)]
#   - pairs image and label by key; raise if any file has no partner

# class HipMRISlices(torch.utils.data.Dataset)
#   __init__(root, split, shape=(256, 128), max_scans=None)
#       - load every slice of the split into memory once 
#       - keep keys and a scan index: (case, week) -> list of slice indices 
#       - has_prostate[i]: whether slice i contains prostate (for oversampling)

#   __len__()
#   __getitem__(i) -> image tensor (1, H, W) float32, label tensor (H, W) int64
#   summary() -> one-line description: slices, scans, cases, % slices with prostate

# ---- preprocessing -----

# crop_or_pad(arr, shape) -> centre-crop or pad to (256, 128)
#   - needed because one scan's slices are 256 x 144 (fro data audit)

# per-scan z-score: for each (case, week), (x - mean) / std over all 60 slices together
#   - intensity ranges differ a lot between scans (data audit: 99th percentile 161-713)
#   - after normalisation: mean about 0, std about 1

# ---- leakage-free splitting -----

# split_cases(root) -> {split: [case ids]} read from the provided folders
#   - the provided 2D split is already by patient (31 / 4 / 3 cases)
#   - check_no_leakage(split_cases(root)) passes

# patient_level_split(case_ids, val_frac, test_frac, seed) -> new random split by patient

# prostate_sampler(weight) -> WeightedRandomSampler drawing prostate slices `weight` x more often
#   - only ~36% of slices contain prostate

# ---- data augmentation -----

# augment_2d(x, y)
#   - random rotation and scaling
#   - brightness / contrast / small Gaussian noise 
#   - NO flips: the slices are not axial, so a flip would put the rectum in front of the prostate




# 3D HipMRI dataset (semantic_MRs + semantic_labels_only)

# ---- data loader -----

# parse_volume_key(filename) -> (patient "B006", patient number 6, week 0)

# list_volume_pairs(root) -> [{id, patient, patient_num, week, image path, label path}]
#   - 211 pairs, 38 patients

# load_volume(path) -> array + voxel spacing (mm) from the header

# class HipMRIVolumes(Dataset)
#   __init__(pairs, crop_centre, crop=(128,128,128), shift=16, train=False, cache_dir=None)
#       - load each volume, cut a window around the crop centre, normalise, cache to disk
#   __getitem__(i) -> image (1, X, Y, Z), label (X, Y, Z)
#       - train=True: random crop inside the window (shift augmentation)
#       - train=False: the fixed central crop (validation / test / predict)

# ---- preprocessing -----

# crop_centre_from_labels(train_pairs) -> centre of the union prostate box over TRAINING labels only
#   - never use test labels for anything except scoring

# extract_window(arr, start, size) -> crop with padding where it leaves the volume
#   - handles the one 256 x 256 x 144 volume

# per-volume z-score using the statistics of the central crop
#   - store mean/std

# ---- leakage-free splitting -----

# split_from_2d(root) -> {split: {case number: set of weeks}} from the 2D filenames

# split_volumes(pairs, mode="match2d") -> {split: [pairs]}
#   - each 3D patient goes in the same split as its 2D case (case_006 -> B006),
#     so the 2D and 3D models are tested on the same patients
#   - compare weeks per patient between 2D and 3D: confirms the mapping is right
#   - mode="random": seeded split by patient instead

# ---- data augmentation -----
# augment_3d(x, y)
#   - left-right flip (valid here: axis 0 is left-right, pelvis nearly symmetric)
#   - rotation about the head-feet axis (axis 2) only, + scaling
#   - brightness / contrast / noise
