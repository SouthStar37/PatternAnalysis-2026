"""
dataset.py - Data loading, preprocessing, leakage-free splitting and augmentation for HipMRI.

Labels: 0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate.
"""
import math
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

import hashlib
import json
import math
import re
from collections import OrderedDict, defaultdict

# SHARED (used by both 2D and 3D)

DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open"
SLICE_DIR = "keras_slices_data" # 2D slices
IMAGE_DIR_3D, LABEL_DIR_3D = "semantic_MRs", "semantic_labels_only" # 3D volumes
SPLITS_2D = { # split -> (image folder, label folder)
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "validate": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
N_CLASSES = len(CLASS_NAMES)
BLADDER, RECTUM, PROSTATE = 3, 4, 5
SLICE_SHAPE = (256, 128) # every 2D slice is cropped/padded to this (H, W)


def load_nifti(path, is_label=False):
    """Read a NIfTI file into a numpy array.

    np.squeeze removes extra size-1 dimensions some files have (e.g. 256x128x1).
    Labels are rounded to integers because some label files are stored as floats.
    """
    arr = np.squeeze(np.asanyarray(nib.load(str(path)).dataobj))
    if is_label:
        return np.rint(arr).astype(np.uint8)
    return arr.astype(np.float32)


def check_no_leakage(groups):
    """Raise an error if any patient appears in more than one split.

    groups: {split name: iterable of patient/case IDs}. Returns patients per split.
    """
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(groups[a]) & set(groups[b])
            if shared:
                raise RuntimeError(f"Patient leakage: {sorted(shared)} are in both {a} and {b}")
    return {k: len(set(v)) for k, v in groups.items()}

# 2D HipMRI dataset (keras_slices_data)

# ---- data loader -----
_KEY_2D = re.compile(r"(\d+)_week_?(\d+)_slice_?(\d+)", re.I) # case_004_week_0_slice_12


def parse_key(filename):
    """'case_004_week_0_slice_12.nii.gz' (or 'seg_...') -> (4, 0, 12)."""
    m = _KEY_2D.search(filename)
    if m is None:
        raise ValueError(f"Cannot parse case/week/slice from '{filename}'")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def list_pairs(root, split):
    """Sorted list of (key, image_path, label_path) for one provided split."""
    if split not in SPLITS_2D:
        raise ValueError(f"split must be one of {list(SPLITS_2D)}")
    
    img_dir, seg_dir = (Path(root) / SLICE_DIR / d for d in SPLITS_2D[split])
    images = {parse_key(p.name): p for p in img_dir.glob("*.nii*")}
    labels = {parse_key(p.name): p for p in seg_dir.glob("*.nii*")}

    if set(images) != set(labels):
        raise RuntimeError(f"{split}: {len(set(images) - set(labels))} images without labels, "
                           f"{len(set(labels) - set(images))} labels without images")
    
    return [(k, images[k], labels[k]) for k in sorted(images)]


def list_all_pairs(root):
    """Pairs from all three provided folders (used only when re-splitting by patient)."""
    return sorted((p for split in SPLITS_2D for p in list_pairs(root, split)), key=lambda p: p[0])

class HipMRISlices(Dataset):
    
    def __init__(self, root=DATA_ROOT, split="train", shape=SLICE_SHAPE, max_scans=None, cases=None, workers=8):
        self.split, self.shape = split, tuple(shape)
        if cases is None:
            pairs = list_pairs(root, split)
        else:
            keep = set(cases)
            pairs = [p for p in list_all_pairs(root) if p[0][0] in keep]
        if max_scans:
            first = list(OrderedDict.fromkeys(k[:2] for k, _, _ in pairs))[:max_scans]
            pairs = [p for p in pairs if p[0][:2] in set(first)]
        if not pairs:
            raise RuntimeError(f"No slices found for split '{split}' under {root}")

        with ThreadPoolExecutor(max_workers=workers) as ex:      # reading 11k files in parallel
            loaded = list(ex.map(self._load_pair, pairs))
        self.images = np.stack([img for img, _ in loaded])       # (N, H, W) float32
        self.labels = np.stack([lab for _, lab in loaded])       # (N, H, W) uint8
        bad = np.setdiff1d(np.unique(self.labels), np.arange(N_CLASSES))
        if bad.size:
            raise ValueError(f"Unexpected label values {bad.tolist()} in split '{split}'")

        self.keys = [k for k, _, _ in pairs]                     # (case, week, slice) per item
        self.scan_index = OrderedDict()                           # (case, week) -> slice indices
        for i, (case, week, _) in enumerate(self.keys):
            self.scan_index.setdefault((case, week), []).append(i)
        self.cases = sorted({case for case, _ in self.scan_index})
        self._normalise_per_scan()
        self.has_prostate = (self.labels == PROSTATE).reshape(len(self.labels), -1).any(axis=1)

    def _load_pair(self, pair):
        _, img_path, lab_path = pair
        img = crop_or_pad(load_nifti(img_path), self.shape)
        lab = crop_or_pad(load_nifti(lab_path, is_label=True), self.shape)
        return img, lab

    def _normalise_per_scan(self):
        """z-score each scan using all its slices together (intensity ranges differ between scans)."""
        for idx in self.scan_index.values():
            block = self.images[idx]
            self.images[idx] = (block - block.mean()) / max(float(block.std()), 1e-6)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).unsqueeze(0) # add channel dim -> (1, H, W)
        lab = torch.from_numpy(self.labels[i].astype(np.int64)) # class numbers -> (H, W)
        return img, lab

    def class_fractions(self):
        counts = np.bincount(self.labels.ravel(), minlength=N_CLASSES)
        return counts / counts.sum()

    def summary(self):
        return (f"{self.split}: {len(self)} slices, {len(self.scan_index)} scans, {len(self.cases)} cases, "
                f"{self.has_prostate.mean():.1%} slices with prostate, "
                f"prostate {self.class_fractions()[PROSTATE]:.3%} of pixels")

# ---- preprocessing -----
def crop_or_pad(arr, shape, fill=0):
    """Centre-crop or pad a 2D array to `shape`. Needed because one scan is 256 x 144."""
    out = np.full(shape, fill, dtype=arr.dtype)
    src, dst = [], []
    for n_in, n_out in zip(arr.shape, shape):
        if n_in >= n_out:                       # crop: take the middle n_out
            start = (n_in - n_out) // 2
            src.append(slice(start, start + n_out))
            dst.append(slice(0, n_out))
        else:                                   # pad: put the array in the middle
            start = (n_out - n_in) // 2
            src.append(slice(0, n_in))
            dst.append(slice(start, start + n_in))
    out[tuple(dst)] = arr[tuple(src)]
    return out

# ---- leakage-free splitting -----

def split_cases(root=DATA_ROOT):
    """Case (patient) IDs in each provided 2D split, read from the filenames."""
    out = {}
    for split, (img_dir, _) in SPLITS_2D.items():
        files = (Path(root) / SLICE_DIR / img_dir).glob("*.nii*")
        out[split] = sorted({parse_key(p.name)[0] for p in files})
    return out


def patient_level_split(case_ids, val_frac=0.15, test_frac=0.15, seed=42):
    """Randomly assign WHOLE patients to train/validate/test (optional alternative split)."""

    cases = sorted(set(case_ids))
    np.random.default_rng(seed).shuffle(cases)
    n_test = max(1, round(test_frac * len(cases)))
    n_val = max(1, round(val_frac * len(cases)))
    split = {"test": sorted(cases[:n_test]),
             "validate": sorted(cases[n_test:n_test + n_val]),
             "train": sorted(cases[n_test + n_val:])}
    check_no_leakage(split)

    return split


def prostate_sampler(dataset, weight=3.0, seed=0):
    """Sampler that draws slices containing prostate `weight` times more often.
    Only ~36% of slices contain prostate, so without this most batches barely show it."""
    w = np.where(dataset.has_prostate, weight, 1.0)
    g = torch.Generator().manual_seed(seed)

    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True, generator=g)

# ---- data augmentation -----

def augment_2d(images, labels, max_rotation_degrees=10.0, scale_range=(0.9, 1.1), spatial_probability=0.5,
               max_brightness_shift=0.1, contrast_range=(0.9, 1.1), noise_strength=0.03):

    batch_size, _, height, width = images.shape
    device = images.device

    apply_spatial = torch.rand(batch_size, device=device) < spatial_probability

    angle = (torch.rand(batch_size, device=device) * 2 - 1) * math.radians(max_rotation_degrees)
    angle = angle * apply_spatial

    random_scale = torch.empty(batch_size, device=device).uniform_(*scale_range)
    scale = torch.where(apply_spatial, random_scale, torch.ones(batch_size, device=device))

    cos_term = torch.cos(angle) / scale
    sin_term = torch.sin(angle) / scale

    transform = torch.zeros(batch_size, 2, 3, device=device)   # one 2x3 matrix per image
    transform[:, 0, 0] = cos_term
    transform[:, 0, 1] = -sin_term * height / width
    transform[:, 1, 0] = sin_term * width / height
    transform[:, 1, 1] = cos_term

    sampling_grid = F.affine_grid(transform, images.shape, align_corners=False)
    images = F.grid_sample(images, sampling_grid, mode="bilinear", padding_mode="border", align_corners=False)
    labels = F.grid_sample(labels.unsqueeze(1).float(), sampling_grid, mode="nearest", padding_mode="border",
                           align_corners=False).squeeze(1).long()

    brightness_shift = (torch.rand(batch_size, 1, 1, 1, device=device) * 2 - 1) * max_brightness_shift
    contrast_factor = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(*contrast_range)
    noise = torch.randn_like(images) * noise_strength
    images = images * contrast_factor + brightness_shift + noise

    return images, labels



# 3D HipMRI dataset (semantic_MRs + semantic_labels_only)

CROP_3D = (128, 128, 128) 
SHIFT_3D = 16  # training crops are randomly shifted by up to this many voxels
_KEY_3D = re.compile(r"([A-Za-z]*)(\d+)_week_?(\d+)", re.I)


# ---- data loader -----

def parse_volume_key(filename):
    m = _KEY_3D.search(filename)
    if m is None:
        raise ValueError(f"Cannot parse patient/week from '{filename}'")
    return f"{m.group(1).upper()}{m.group(2)}", int(m.group(2)), int(m.group(3))


def list_volume_pairs(root=DATA_ROOT):
    root = Path(root)
    images = {parse_volume_key(p.name): p for p in (root / IMAGE_DIR_3D).glob("*.nii*")}
    labels = {parse_volume_key(p.name): p for p in (root / LABEL_DIR_3D).glob("*.nii*")}
    if not images:
        raise FileNotFoundError(f"No volumes found in {root / IMAGE_DIR_3D}")
    if set(images) != set(labels):
        raise RuntimeError(f"{len(set(images) - set(labels))} volumes without labels, "
                           f"{len(set(labels) - set(images))} labels without volumes")
    pairs = []
    for patient, number, week in sorted(images):
        key = (patient, number, week)
        pairs.append({"id": f"{patient}_Week{week}", "patient": patient, "patient_num": number, "week": week,
                      "image": str(images[key]), "label": str(labels[key])})
    return pairs


def load_volume(path, is_label=False):
    spacing = tuple(float(z) for z in nib.load(str(path)).header.get_zooms()[:3])
    return load_nifti(path, is_label), spacing

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
