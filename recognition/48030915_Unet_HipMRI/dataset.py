"""
dataset.py - Data loading, preprocessing, leakage-free splitting and augmentation for HipMRI.

2D: the provided keras_slices_data (coronal slices, 256 x 128) with its patient-level split.
3D: the semantic_MRs / semantic_labels_only volumes (256 x 256 x 128), split so that every patient
    is in the same split as in 2D, and cropped to 128^3 around the prostate.

Labels: 0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate.
Axes of the 3D volumes (LPS): 0 = left-right, 1 = front-back, 2 = head-feet.
"""
import hashlib
import json
import math
import os
import re
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

# --- Shared by 2D and 3D ---
DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open"
SLICE_DIR = "keras_slices_data"
IMAGE_DIR_3D, LABEL_DIR_3D = "semantic_MRs", "semantic_labels_only"
SPLITS_2D = { # (image folder, label folder)
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "validate": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
N_CLASSES = len(CLASS_NAMES)
BLADDER, RECTUM, PROSTATE = 3, 4, 5


def load_nifti(path, is_label=False):
    """Read a NIfTI file as float32 (images) or uint8 class numbers (labels)"""
    array = np.squeeze(np.asanyarray(nib.load(str(path)).dataobj))
    return np.rint(array).astype(np.uint8) if is_label else array.astype(np.float32)


def check_no_leakage(groups):
    """Raise RuntimeError if any patient is in more than one split

    groups: {split: patient IDs}. Returns {split: number of patients}.
    """
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(groups[a]) & set(groups[b])
            if shared:
                raise RuntimeError(f"Patient leakage: {sorted(shared)} are in both {a} and {b}")
    return {split: len(set(ids)) for split, ids in groups.items()}


def _augment_intensity(images, max_brightness_shift, contrast_range, noise_strength):
    """Random brightness shift, contrast scaling and Gaussian noise, per image (2D or 3D batch)"""
    shape = (images.shape[0],) + (1,) * (images.ndim - 1)
    brightness_shift = (torch.rand(shape, device=images.device) * 2 - 1) * max_brightness_shift
    contrast_factor = torch.empty(shape, device=images.device).uniform_(*contrast_range)
    noise = torch.randn_like(images) * noise_strength
    return images * contrast_factor + brightness_shift + noise



# --- 2D slices (keras_slices_data) ---
SLICE_SHAPE = (256, 128) # every slice is centre-cropped / padded to this
_KEY_2D = re.compile(r"(\d+)_week_?(\d+)_slice_?(\d+)", re.I)

# data loader
def parse_key(filename):
    """'case_004_week_0_slice_12.nii.gz' (or 'seg_...') -> (case 4, week 0, slice 12)"""
    match = _KEY_2D.search(filename)
    if match is None:
        raise ValueError(f"Cannot parse case/week/slice from '{filename}'")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))

def list_pairs(root, split):
    """Sorted [(key, image path, label path)] for one provided split; images and labels matched by key"""
    if split not in SPLITS_2D:
        raise ValueError(f"split must be one of {list(SPLITS_2D)}")
    image_dir, label_dir = (Path(root) / SLICE_DIR / folder for folder in SPLITS_2D[split])
    images = {parse_key(p.name): p for p in image_dir.glob("*.nii*")}
    labels = {parse_key(p.name): p for p in label_dir.glob("*.nii*")}
    if set(images) != set(labels):
        raise RuntimeError(f"{split}: {len(set(images) - set(labels))} images without labels, "
                           f"{len(set(labels) - set(images))} labels without images")
    return [(key, images[key], labels[key]) for key in sorted(images)]

def list_all_pairs(root):
    """Pairs from all three provided splits (used by check_patient_match.py)"""
    return sorted((p for split in SPLITS_2D for p in list_pairs(root, split)), key=lambda p: p[0])

# preprocessing
def crop_or_pad(array, shape, fill=0):
    """Centre-crop or zero-pad a 2D array to `shape` (one scan's slices are 256 x 144)"""
    out = np.full(shape, fill, dtype=array.dtype)
    source, target = [], []
    for size_in, size_out in zip(array.shape, shape):
        if size_in >= size_out:
            start = (size_in - size_out) // 2
            source.append(slice(start, start + size_out))
            target.append(slice(0, size_out))
        else:
            start = (size_out - size_in) // 2
            source.append(slice(0, size_in))
            target.append(slice(start, start + size_in))
    out[tuple(target)] = array[tuple(source)]
    return out


class HipMRISlices(Dataset):
    """All slices of one provided 2D split, held in memory and z-scored per scan

    Item i: image (1, H, W) float32, label (H, W) int64.
    keys[i] is the slice's (case, week, slice); scan_index maps (case, week) -> its slice indices,
    which predict.py uses to stack the slices back into one block per scan.
    """

    def __init__(self, root=DATA_ROOT, split="train", shape=SLICE_SHAPE, max_scans=None, workers=8):
        self.split, self.shape = split, tuple(shape)
        pairs = list_pairs(root, split)
        if max_scans:                             # quick tests: only the first N scans
            first = list(OrderedDict.fromkeys(key[:2] for key, _, _ in pairs))[:max_scans]
            pairs = [p for p in pairs if p[0][:2] in set(first)]
        if not pairs:
            raise RuntimeError(f"No slices found for split '{split}' under {root}")

        with ThreadPoolExecutor(max_workers=workers) as pool:    # ~11k small files: read in parallel
            loaded = list(pool.map(self._load_pair, pairs))
        self.images = np.stack([image for image, _ in loaded])   # (N, H, W) float32
        self.labels = np.stack([label for _, label in loaded])   # (N, H, W) uint8
        unexpected = np.setdiff1d(np.unique(self.labels), np.arange(N_CLASSES))
        if unexpected.size:
            raise ValueError(f"Unexpected label values {unexpected.tolist()} in split '{split}'")

        self.keys = [key for key, _, _ in pairs]
        self.scan_index = OrderedDict()
        for i, (case, week, _) in enumerate(self.keys):
            self.scan_index.setdefault((case, week), []).append(i)
        self.cases = sorted({case for case, _ in self.scan_index})
        self._normalise_per_scan()
        self.has_prostate = (self.labels == PROSTATE).reshape(len(self.labels), -1).any(axis=1)

    def _load_pair(self, pair):
        _, image_path, label_path = pair
        return (crop_or_pad(load_nifti(image_path), self.shape),
                crop_or_pad(load_nifti(label_path, is_label=True), self.shape))

    def _normalise_per_scan(self):
        """z-score each scan over all its slices (intensity ranges differ between scans)"""
        for indices in self.scan_index.values():
            block = self.images[indices]
            self.images[indices] = (block - block.mean()) / max(float(block.std()), 1e-6)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return (torch.from_numpy(self.images[i]).unsqueeze(0),
                torch.from_numpy(self.labels[i].astype(np.int64)))

    def class_fractions(self):
        """Fraction of all pixels in each class."""
        counts = np.bincount(self.labels.ravel(), minlength=N_CLASSES)
        return counts / counts.sum()

    def summary(self):
        return (f"{self.split}: {len(self)} slices, {len(self.scan_index)} scans, {len(self.cases)} cases, "
                f"{self.has_prostate.mean():.1%} slices with prostate, "
                f"prostate {self.class_fractions()[PROSTATE]:.3%} of pixels")

def prostate_sampler(dataset, weight=3.0, seed=0):
    """Sample slices that contain prostate `weight` times more often (only ~36% of slices do)"""
    weights = np.where(dataset.has_prostate, weight, 1.0)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True,
                                 generator=generator)

# leakage-free splitting
def split_cases(root=DATA_ROOT):
    """{split: case IDs} of the provided 2D split, read from the filenames"""
    return {split: sorted({parse_key(p.name)[0] for p in (Path(root) / SLICE_DIR / image_dir).glob("*.nii*")})
            for split, (image_dir, _) in SPLITS_2D.items()}

# data augmentation
def augment_2d(images, labels, max_rotation_degrees=10.0, scale_range=(0.9, 1.1), spatial_probability=0.5,
               max_brightness_shift=0.1, contrast_range=(0.9, 1.1), noise_strength=0.03):
    """Random in-plane rotation + scaling (image and label together), then intensity changes

    images: (B, 1, H, W) float; labels: (B, H, W) class numbers. Runs on the GPU, per batch.
    """
    batch_size, _, height, width = images.shape
    device = images.device
    apply_spatial = torch.rand(batch_size, device=device) < spatial_probability
    angle = (torch.rand(batch_size, device=device) * 2 - 1) * math.radians(max_rotation_degrees) * apply_spatial
    random_scale = torch.empty(batch_size, device=device).uniform_(*scale_range)
    scale = torch.where(apply_spatial, random_scale, torch.ones(batch_size, device=device))
    cos_term, sin_term = torch.cos(angle) / scale, torch.sin(angle) / scale

    # affine_grid works in [-1, 1] coordinates, so the off-diagonal terms are corrected for H != W
    transform = torch.zeros(batch_size, 2, 3, device=device)
    transform[:, 0, 0], transform[:, 0, 1] = cos_term, -sin_term * height / width
    transform[:, 1, 0], transform[:, 1, 1] = sin_term * width / height, cos_term
    grid = F.affine_grid(transform, images.shape, align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
    labels = F.grid_sample(labels.unsqueeze(1).float(), grid, mode="nearest", padding_mode="border",
                           align_corners=False).squeeze(1).long()
    return _augment_intensity(images, max_brightness_shift, contrast_range, noise_strength), labels

# --- 3D volumes (semantic_MRs + semantic_labels_only) ---
CROP_3D = (128, 128, 128) # network input; multiples of 16 because the network halves the size 4 times
SHIFT_3D = 16 # training crops are randomly shifted by up to this many voxels
_KEY_3D = re.compile(r"([A-Za-z]*)(\d+)_week_?(\d+)", re.I)

# data loader
def parse_volume_key(filename):
    """'B006_Week0_LFOV.nii.gz' -> ('B006', patient number 6, week 0)"""
    match = _KEY_3D.search(filename)
    if match is None:
        raise ValueError(f"Cannot parse patient/week from '{filename}'")
    return f"{match.group(1).upper()}{match.group(2)}", int(match.group(2)), int(match.group(3))


def list_volume_pairs(root=DATA_ROOT):
    """One dict per usable scan: id, patient, patient_num, week, image path, label path

    Images and labels are matched by (patient, week). Unusable scans (unreadable name, duplicate,
    missing image or label, or image/label shapes that differ) are reported and skipped, so one
    bad file does not stop a training job.
    """
    root = Path(root)
    skipped = []

    def index(folder):
        found = {}
        for path in sorted((root / folder).glob("*.nii*")):
            try:
                key = parse_volume_key(path.name)
            except ValueError:
                skipped.append(f"{folder}/{path.name}: cannot read patient/week from the name")
                continue
            if key in found:
                skipped.append(f"{folder}/{path.name}: duplicate of {found[key].name}")
                continue
            found[key] = path
        return found

    images, labels = index(IMAGE_DIR_3D), index(LABEL_DIR_3D)
    if not images:
        raise FileNotFoundError(f"No volumes found in {root / IMAGE_DIR_3D}")
    skipped += [f"{patient}_Week{week}: image has no label" for patient, _, week in sorted(set(images) - set(labels))]
    skipped += [f"{patient}_Week{week}: label has no image" for patient, _, week in sorted(set(labels) - set(images))]

    pairs = []
    for key in sorted(set(images) & set(labels)):
        patient, number, week = key
        scan_id = f"{patient}_Week{week}"
        image_shape = nib.load(str(images[key])).shape[:3]       # header only: fast
        label_shape = nib.load(str(labels[key])).shape[:3]
        if image_shape != label_shape:
            skipped.append(f"{scan_id}: image {image_shape} and label {label_shape} differ")
            continue
        pairs.append({"id": scan_id, "patient": patient, "patient_num": number, "week": week,
                      "image": str(images[key]), "label": str(labels[key])})

    print(f"Found {len(pairs)} usable 3D scans, skipped {len(skipped)}", flush=True)
    for reason in skipped:
        print("  skipped:", reason, flush=True)
    if not pairs:
        raise RuntimeError("No usable 3D scans")
    return pairs


def load_volume(path, is_label=False):
    """A volume and its voxel spacing in mm"""
    spacing = tuple(float(z) for z in nib.load(str(path)).header.get_zooms()[:3])
    return load_nifti(path, is_label), spacing

def load_data_3D(image_names, norm_image=False, dtype=np.float32, get_affines=False, early_stop=False,
                 centre=None, crop=CROP_3D, shift=SHIFT_3D):
    """Load NIfTI volumes into one pre-allocated array (num, X, Y, Z)

    Adpated and improved with new features from load_data_3D from the task sheet (Appendix B, Listing 2)
    """
    is_label = dtype == np.uint8
    first_case = load_nifti(image_names[0], is_label)
    if centre is None:
        window_size = None
        rows, cols, depth = first_case.shape
    else:
        _, _, window_size = crop_windows(first_case.shape, centre, crop, shift)
        rows, cols, depth = window_size
    images = np.zeros((len(image_names), rows, cols, depth), dtype=dtype)
    affines = []

    for i, name in enumerate(image_names):
        nifti_image = nib.load(str(name))
        image = load_nifti(name, is_label)
        info = {"affine": nifti_image.affine.tolist(), "shape": list(image.shape),
                "spacing": [float(z) for z in nifti_image.header.get_zooms()[:3]]}
        if centre is None:
            image = image[:rows, :cols, :depth]
            offset, crop_here = [0, 0, 0], image.shape
        else:
            eval_start, window_start, _ = crop_windows(image.shape, centre, crop, shift, window_size)
            image = extract_window(image, window_start, window_size, fill=0 if is_label else float(image.min()))
            offset, crop_here = [e - w for e, w in zip(eval_start, window_start)], crop
            info.update(eval_start=eval_start, eval_offset=offset)
        image = image.astype(dtype)
        if norm_image:
            central = image[tuple(slice(o, o + n) for o, n in zip(offset, crop_here))]
            mean, std = float(central.mean()), float(central.std())
            image = (image - mean) / (std + 1e-8)
            info.update(norm_mean=mean, norm_std=std)
        images[i, :image.shape[0], :image.shape[1], :image.shape[2]] = image
        affines.append(info)
        if i > 20 and early_stop:
            break

    return (images, affines) if get_affines else images

# leakage-free splitting
def split_volumes(pairs, root=DATA_ROOT):
    """Split the 3D scans by patient, putting each patient in the same split as its 2D case

    The 3D patient number matches the 2D case number (B006 <-> case_006; verified by
    data audit and manual inspection), so the 2D and 3D models are tested on the same patients. Only the
    2D filenames are read. Returns ({split: [pairs]}, {split: [patients]}, warnings), where the
    warnings list patients with no 2D case (put in train) or with different weeks in 2D and 3D.
    """
    cases_2d = split_cases(root)
    check_no_leakage(cases_2d)
    split_of_case = {case: split for split, cases in cases_2d.items() for case in cases}
    weeks_2d = defaultdict(set)
    for image_dir, _ in SPLITS_2D.values():
        for path in (Path(root) / SLICE_DIR / image_dir).glob("*.nii*"):
            case, week, _ = parse_key(path.name)
            weeks_2d[case].add(week)

    groups, warnings = {split: [] for split in SPLITS_2D}, []
    for patient in sorted({p["patient"] for p in pairs}):
        scans = [p for p in pairs if p["patient"] == patient]
        case = scans[0]["patient_num"]
        if case not in split_of_case:
            groups["train"].append(patient)
            warnings.append(f"{patient}: no 2D case {case}, put in train")
            continue
        groups[split_of_case[case]].append(patient)
        weeks_3d = {p["week"] for p in scans}
        if weeks_3d != weeks_2d[case]:
            warnings.append(f"{patient}: 3D weeks {sorted(weeks_3d)} vs 2D weeks {sorted(weeks_2d[case])}")

    check_no_leakage(groups)
    by_split = {split: [p for p in pairs if p["patient"] in groups[split]] for split in groups}
    return by_split, groups, warnings

# preprocessing
def crop_centre_from_labels(train_pairs, workers=4):
    """Crop centre = centre of the box containing the prostate of every TRAINING volume"""
    def prostate_box(pair):
        labels, _ = load_volume(pair["label"], is_label=True)
        voxels = np.nonzero(labels == PROSTATE)
        return [(int(v.min()), int(v.max())) for v in voxels] if voxels[0].size else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        boxes = [box for box in pool.map(prostate_box, train_pairs) if box]
    if not boxes:
        raise RuntimeError("No prostate found in the training labels")
    low = [min(box[axis][0] for box in boxes) for axis in range(3)]
    high = [max(box[axis][1] for box in boxes) for axis in range(3)]
    return [int(round((lo + hi) / 2)) for lo, hi in zip(low, high)], {"min": low, "max": high}


def crop_windows(volume_shape, centre, crop=CROP_3D, shift=SHIFT_3D, window_size=None):
    """Where to cut one volume. Returns (eval_start, window_start, window_size)

    eval_start: the fixed crop around `centre`, kept inside the volume (validation, test, predict)
    window: a larger region (crop + `shift` voxels each side where the volume allows) kept in memory,
            from which training takes randomly shifted crops. window_size=None sizes it for this
            volume; load_data_3D passes the first volume's size so all windows are the same shape
    """
    eval_start = [int(min(max(c - n // 2, 0), max(length - n, 0)))
                  for c, n, length in zip(centre, crop, volume_shape)]
    if window_size is None:
        window_size = [min(n + 2 * shift, max(length, n)) for n, length in zip(crop, volume_shape)]
    window_start = []
    for e, w, n, length in zip(eval_start, window_size, crop, volume_shape):
        start = e - (w - n) // 2 # centre the window on the crop,
        start = min(max(start, 0), max(length - w, 0))
        start = min(max(start, e + n - w), e)
        window_start.append(int(start))
    return eval_start, window_start, list(window_size)


def extract_window(volume, start, size, fill=0):
    """volume[start : start + size] on every axis, padded with `fill` where it leaves the volume"""
    out = np.full(size, fill, dtype=volume.dtype)
    source, target = [], []
    for s, n, length in zip(start, size, volume.shape):
        a, b = max(s, 0), min(s + n, length)
        source.append(slice(a, b))
        target.append(slice(a - s, b - s))
    out[tuple(target)] = volume[tuple(source)]
    return out


class HipMRIVolumes(Dataset):
    """Prostate-centred 3D crops of one split, held in memory

    Each volume's window (crop + SHIFT_3D margin) is loaded once with load_data_3D, keyed by the scan IDs and crop settings
    train=True:  every access returns a crop at a random position inside the window (shift augmentation)
    train=False: always the fixed central crop (validation, test, predict.py)
    Item i: image (1, X, Y, Z) float32, label (X, Y, Z) int64. meta[i]: spacing, crop position, normalisation
    """
    def __init__(self, pairs, centre, crop=CROP_3D, shift=SHIFT_3D, train=False, cache_dir=None):
        self.pairs, self.train = pairs, train
        self.centre, self.crop, self.shift = [int(c) for c in centre], tuple(crop), shift
        self.ids = [p["id"] for p in pairs]
        cache = self._cache_path(cache_dir)
        if cache is not None and cache.exists():
            saved = np.load(cache)
            self.images, self.labels = saved["images"], saved["labels"]
            self.meta = json.loads(str(saved["meta"]))
            return

        settings = {"centre": self.centre, "crop": self.crop, "shift": self.shift}
        self.images, self.meta = load_data_3D([p["image"] for p in pairs], norm_image=True, get_affines=True,
                                              **settings)
        self.labels, label_meta = load_data_3D([p["label"] for p in pairs], dtype=np.uint8, get_affines=True,
                                               **settings)
        for pair, image_info, label_info in zip(pairs, self.meta, label_meta):
            if image_info["shape"] != label_info["shape"]:
                raise RuntimeError(f"{pair['id']}: image {image_info['shape']} and label {label_info['shape']} differ")
        if cache is not None:
            temp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
            np.savez(temp, images=self.images, labels=self.labels, meta=json.dumps(self.meta))
            temp.replace(cache)

    def _cache_path(self, cache_dir):
        if not cache_dir:
            return None
        folder = Path(cache_dir).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        settings = json.dumps([self.ids, self.centre, list(self.crop), self.shift])
        return folder / f"volumes_{len(self.ids)}_{hashlib.md5(settings.encode()).hexdigest()[:10]}.npz"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image, label = self.images[i], self.labels[i]
        if self.train:
            offset = [int(torch.randint(0, w - n + 1, (1,))) for w, n in zip(image.shape, self.crop)]
        else:
            offset = self.meta[i]["eval_offset"]
        region = tuple(slice(o, o + n) for o, n in zip(offset, self.crop))
        return (torch.from_numpy(np.ascontiguousarray(image[region])).unsqueeze(0),
                torch.from_numpy(label[region].astype(np.int64)))

    def summary(self):
        n_patients = len({p["patient"] for p in self.pairs})
        counts = np.bincount(self.labels.ravel(), minlength=N_CLASSES)
        return (f"{len(self)} volumes from {n_patients} patients, stored {self.images.shape[1:]}, "
                f"crop {self.crop}, prostate {counts[PROSTATE] / counts.sum():.2%} of stored voxels")

# data augmentation
def augment_3d(images, labels, flip_probability=0.5, max_rotation_degrees=10.0, scale_range=(0.9, 1.1),
               spatial_probability=0.5, max_brightness_shift=0.1, contrast_range=(0.9, 1.1), noise_strength=0.03):
    """Random left-right flip, rotation about the head-feet axis + scaling, then intensity changes

    images: (B, 1, X, Y, Z) float; labels: (B, X, Y, Z) class numbers
    """
    batch_size, device = images.shape[0], images.device

    flip = torch.rand(batch_size, device=device) < flip_probability
    images = torch.where(flip.view(-1, 1, 1, 1, 1), images.flip(2), images)
    labels = torch.where(flip.view(-1, 1, 1, 1), labels.flip(1), labels)

    apply_spatial = torch.rand(batch_size, device=device) < spatial_probability
    angle = (torch.rand(batch_size, device=device) * 2 - 1) * math.radians(max_rotation_degrees) * apply_spatial
    random_scale = torch.empty(batch_size, device=device).uniform_(*scale_range)
    scale = torch.where(apply_spatial, random_scale, torch.ones(batch_size, device=device))
    cos_term, sin_term = torch.cos(angle) / scale, torch.sin(angle) / scale
    # affine_grid orders coordinates (Z, Y, X): rows 1-2 rotate the X-Y plane, row 0 only scales Z
    transform = torch.zeros(batch_size, 3, 4, device=device)
    transform[:, 0, 0] = 1 / scale
    transform[:, 1, 1], transform[:, 1, 2] = cos_term, -sin_term
    transform[:, 2, 1], transform[:, 2, 2] = sin_term, cos_term
    grid = F.affine_grid(transform, images.shape, align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
    labels = F.grid_sample(labels.unsqueeze(1).float(), grid, mode="nearest", padding_mode="border",
                           align_corners=False).squeeze(1).long()
    return _augment_intensity(images, max_brightness_shift, contrast_range, noise_strength), labels