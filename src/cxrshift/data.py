"""Data loading, binary label mapping, transforms and histogram matching.

Every dataset is reduced to the same in-memory form: a uint8 array of grayscale images
with shape (N, H, W) and a binary label vector (1 = pneumonia, 0 = negative class).

Three loaders are supported, chosen by ``loader`` in the dataset config:

* ``medmnist``: any MedMNIST flag (e.g. pneumoniamnist, chestmnist). Label names are read
  from ``medmnist.INFO`` at runtime and mapped by name, never by hardcoded index.
* ``folder``: an ImageFolder style tree ``root/<split>/<class_name>/*.png`` such as the
  Kaggle chest X-ray pneumonia dataset. ``label_map`` maps class folder names.
* ``synthetic``: random images with a weak class signal, used for tests only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger("cxrshift.data")

NO_FINDING = "no_finding"
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class SplitData:
    images: np.ndarray  # uint8, (N, H, W)
    labels: np.ndarray  # int64, (N,), values in {0, 1}
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, idx: np.ndarray, name: str | None = None) -> "SplitData":
        return SplitData(self.images[idx], self.labels[idx], name or self.name, dict(self.meta))

    def with_images(self, images: np.ndarray, name: str) -> "SplitData":
        return SplitData(images, self.labels, name, dict(self.meta))


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


def _as_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def resolve_label_indices(label_names: dict[str, str], wanted: list[str]) -> list[int]:
    """Map label names to indices using a ``{index_str: name}`` dict such as medmnist INFO.

    Matching is case insensitive. Raises if a name is not present, so a wrong config fails
    loudly instead of silently using the wrong column.
    """
    lookup = {name.lower(): int(idx) for idx, name in label_names.items()}
    out = []
    for name in wanted:
        if name == NO_FINDING:
            continue
        key = name.lower()
        if key not in lookup:
            raise KeyError(
                f"Label {name!r} not found. Available labels: {sorted(lookup)}"
            )
        out.append(lookup[key])
    return out


def map_to_binary(
    raw_labels: np.ndarray,
    label_names: dict[str, str],
    task: str,
    positive: list[str] | str,
    negative: list[str] | str,
    exclusive_positive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dataset labels to a binary pneumonia label.

    Returns ``(keep_mask, binary_labels)`` where ``binary_labels`` is only meaningful on
    rows where ``keep_mask`` is True. Rows that are neither positive nor negative are dropped.

    For single-label tasks (binary-class, multi-class) ``raw_labels`` holds class indices.
    For multi-label tasks it holds a 0/1 matrix with one column per label name. The special
    negative ``"no_finding"`` means "all label columns are zero".
    """
    positive = _as_list(positive)
    negative = _as_list(negative)
    pos_idx = resolve_label_indices(label_names, positive)
    neg_idx = resolve_label_indices(label_names, negative)
    if not pos_idx:
        raise ValueError("At least one positive label name is required")
    if set(pos_idx) & set(neg_idx):
        raise ValueError("A label cannot be both positive and negative")

    raw = np.asarray(raw_labels)
    if task.startswith("multi-label"):
        if raw.ndim != 2 or raw.shape[1] != len(label_names):
            raise ValueError(
                f"Multi-label data should have {len(label_names)} columns, got shape {raw.shape}"
            )
        is_pos = raw[:, pos_idx].max(axis=1) > 0
        if exclusive_positive:
            others = [i for i in range(raw.shape[1]) if i not in pos_idx]
            is_pos &= raw[:, others].sum(axis=1) == 0
        is_neg = np.zeros(len(raw), dtype=bool)
        if NO_FINDING in negative:
            is_neg |= raw.sum(axis=1) == 0
        if neg_idx:
            is_neg |= (raw[:, neg_idx].max(axis=1) > 0) & (raw[:, pos_idx].max(axis=1) == 0)
        is_neg &= ~is_pos
    else:
        if NO_FINDING in negative:
            raise ValueError("'no_finding' is only valid for multi-label datasets")
        flat = raw.reshape(len(raw), -1)[:, 0]
        is_pos = np.isin(flat, pos_idx)
        is_neg = np.isin(flat, neg_idx)
    keep = is_pos | is_neg
    return keep, is_pos.astype(np.int64)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _to_gray_uint8(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 4:  # (N, H, W, C)
        images = images.mean(axis=-1)
    if images.dtype != np.uint8:
        images = np.clip(images, 0, 255).astype(np.uint8)
    return images


def load_medmnist_split(ds_cfg: dict, split: str) -> SplitData:
    import medmnist

    flag = ds_cfg["name"].lower()
    if flag not in medmnist.INFO:
        raise KeyError(f"Unknown medmnist flag {flag!r}")
    info = medmnist.INFO[flag]
    data_class = getattr(medmnist, info["python_class"])
    root = Path(ds_cfg.get("root", "data")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    size = int(ds_cfg.get("size", 224))
    ds = data_class(
        split=split,
        size=size,
        root=str(root),
        download=bool(ds_cfg.get("download", True)),
    )
    label_map = ds_cfg["label_map"]
    keep, y = map_to_binary(
        ds.labels,
        info["label"],
        info["task"],
        label_map["positive"],
        label_map["negative"],
        bool(label_map.get("exclusive_positive", False)),
    )
    images = _to_gray_uint8(ds.imgs)[keep]
    return SplitData(
        images,
        y[keep],
        name=f"{flag}/{split}",
        meta={"label_names": info["label"], "task": info["task"], "dropped": int((~keep).sum())},
    )


def load_folder_split(ds_cfg: dict, split: str) -> SplitData:
    from PIL import Image

    root = Path(ds_cfg["root"]).expanduser()
    split_dir = root / ds_cfg.get("split_dirs", {}).get(split, split)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split folder {split_dir}")
    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    label_names = {str(i): p.name for i, p in enumerate(class_dirs)}
    label_map = ds_cfg["label_map"]
    pos_idx = set(resolve_label_indices(label_names, _as_list(label_map["positive"])))
    neg_idx = set(resolve_label_indices(label_names, _as_list(label_map["negative"])))
    size = int(ds_cfg.get("size", 224))
    images, labels = [], []
    for i, cdir in enumerate(class_dirs):
        if i not in pos_idx and i not in neg_idx:
            continue
        for fp in sorted(cdir.iterdir()):
            if fp.suffix.lower() not in IMG_EXTENSIONS:
                continue
            with Image.open(fp) as im:
                im = im.convert("L").resize((size, size), Image.BILINEAR)
                images.append(np.asarray(im, dtype=np.uint8))
            labels.append(1 if i in pos_idx else 0)
    if not images:
        raise RuntimeError(f"No images found under {split_dir}")
    return SplitData(
        np.stack(images),
        np.asarray(labels, dtype=np.int64),
        name=f"{root.name}/{split}",
        meta={"label_names": label_names, "task": "folder"},
    )


def make_synthetic_split(ds_cfg: dict, split: str) -> SplitData:
    """Random grayscale images. Positives get a brighter blob in one lung field.

    ``shift`` darkens and adds contrast change plus noise to imitate a different scanner.
    ``corner_marker`` puts a bright square in the top left corner of positives, which gives
    the shortcut check something to find in tests.
    """
    split_offsets = {"train": 0, "val": 1, "test": 2}
    seed = int(ds_cfg.get("seed", 0)) * 10 + split_offsets.get(split, 3)
    rng = np.random.default_rng(seed)
    n = int(ds_cfg.get("n_per_split", {}).get(split, 64))
    size = int(ds_cfg.get("size", 32))
    prevalence = float(ds_cfg.get("prevalence", 0.5))
    labels = (rng.random(n) < prevalence).astype(np.int64)
    yy, xx = np.mgrid[0:size, 0:size] / size
    base = 0.35 + 0.25 * np.exp(-((xx - 0.5) ** 2) / 0.08)  # bright mediastinum
    imgs = np.empty((n, size, size), dtype=np.float32)
    for i in range(n):
        img = base + rng.normal(0, 0.08, (size, size))
        if labels[i] == 1:
            cx = 0.3 if rng.random() < 0.5 else 0.7
            img += 0.35 * np.exp(-((xx - cx) ** 2 + (yy - 0.55) ** 2) / 0.02)
            if ds_cfg.get("corner_marker", False):
                k = max(2, size // 10)
                img[:k, :k] = 1.0
        imgs[i] = img
    shift = float(ds_cfg.get("shift", 0.0))
    if shift:
        imgs = 0.8 * imgs ** (1 + shift) + rng.normal(0, 0.05 * shift, imgs.shape)
    images = (np.clip(imgs, 0, 1) * 255).astype(np.uint8)
    return SplitData(images, labels, name=f"synthetic/{split}", meta={"task": "synthetic"})


LOADERS = {
    "medmnist": load_medmnist_split,
    "folder": load_folder_split,
    "synthetic": make_synthetic_split,
}


def subsample(split: SplitData, max_samples: int | None, seed: int) -> SplitData:
    """Stratified random subsample, keeping class proportions."""
    if not max_samples or len(split) <= max_samples:
        return split
    rng = np.random.default_rng(seed)
    idx = []
    for c in (0, 1):
        cidx = np.flatnonzero(split.labels == c)
        k = int(round(max_samples * len(cidx) / len(split)))
        idx.append(rng.choice(cidx, size=min(k, len(cidx)), replace=False))
    idx = np.sort(np.concatenate(idx))
    return split.subset(idx)


def load_split(ds_cfg: dict, split: str, seed: int = 0) -> SplitData:
    loader = ds_cfg.get("loader", "medmnist")
    if loader not in LOADERS:
        raise KeyError(f"Unknown loader {loader!r}. Options: {sorted(LOADERS)}")
    data = LOADERS[loader](ds_cfg, split)
    max_samples = (ds_cfg.get("max_samples") or {}).get(split)
    data = subsample(data, max_samples, seed)
    logger.info(
        "Loaded %s: n=%d, positives=%d (%.1f%%)",
        data.name, len(data), int(data.labels.sum()), 100 * data.labels.mean(),
    )
    return data


def split_for_calibration(val: SplitData, cal_fraction: float, seed: int) -> tuple[SplitData, SplitData]:
    """Split the source validation set into a part for early stopping, temperature and threshold
    fitting, and a disjoint part used only as the conformal calibration set."""
    rng = np.random.default_rng(seed)
    fit_idx, cal_idx = [], []
    for c in (0, 1):
        cidx = rng.permutation(np.flatnonzero(val.labels == c))
        k = int(round(len(cidx) * cal_fraction))
        cal_idx.append(cidx[:k])
        fit_idx.append(cidx[k:])
    fit_idx = np.sort(np.concatenate(fit_idx))
    cal_idx = np.sort(np.concatenate(cal_idx))
    return val.subset(fit_idx, val.name + "[fit]"), val.subset(cal_idx, val.name + "[cal]")


def load_all(cfg: dict) -> dict[str, SplitData]:
    """Load every split the pipeline needs.

    Returns keys ``source_train``, ``source_val`` (early stopping, temperature, threshold),
    ``source_cal`` (conformal calibration), ``source_test`` and ``target_test``.
    """
    seed = int(cfg.get("seed", 0))
    src, tgt = cfg["data"]["source"], cfg["data"]["target"]
    out = {
        "source_train": load_split(src, "train", seed),
        "source_test": load_split(src, "test", seed),
        "target_test": load_split(tgt, tgt.get("eval_split", "test"), seed),
    }
    val = load_split(src, "val", seed)
    cal_fraction = float(cfg.get("conformal", {}).get("cal_fraction", 0.5))
    out["source_val"], out["source_cal"] = split_for_calibration(val, cal_fraction, seed)
    for key in out:
        out[key].name = key
    return out


# ---------------------------------------------------------------------------
# Torch dataset and transforms
# ---------------------------------------------------------------------------


def build_augmentation(strength: str):
    """Augmentations applied to float tensors of shape (1, H, W) in [0, 1]."""
    from torchvision.transforms import v2 as T

    if strength in (None, "none"):
        return None
    if strength == "basic":
        return T.RandomAffine(degrees=5, translate=(0.04, 0.04), scale=(0.95, 1.05))
    if strength == "strong":
        # Aimed at acquisition differences: field of view, exposure, contrast, blur, noise.
        return T.Compose([
            T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.85, 1.15)),
            T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.8),
            RandomGamma((0.7, 1.5), p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
            GaussianNoise(0.03, p=0.3),
        ])
    raise ValueError(f"Unknown augmentation strength {strength!r}")


class RandomGamma(torch.nn.Module):
    def __init__(self, gamma_range=(0.7, 1.5), p=0.5):
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p

    def forward(self, x):
        if torch.rand(()) >= self.p:
            return x
        lo, hi = self.gamma_range
        gamma = float(torch.empty(()).uniform_(lo, hi))
        return x.clamp(0, 1) ** gamma


class GaussianNoise(torch.nn.Module):
    def __init__(self, std=0.03, p=0.3):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, x):
        if torch.rand(()) >= self.p:
            return x
        return (x + torch.randn_like(x) * self.std).clamp(0, 1)


class CXRDataset(Dataset):
    """Yields ``(image, label)`` with image a float tensor (1, S, S) in [0, 1]."""

    def __init__(self, split: SplitData, image_size: int, augment=None):
        self.images = split.images
        self.labels = split.labels
        self.image_size = int(image_size)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(np.ascontiguousarray(self.images[i])).float().div_(255.0)[None]
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x[None], size=(self.image_size, self.image_size), mode="bilinear",
                align_corners=False, antialias=True,
            )[0]
        if self.augment is not None:
            x = self.augment(x).clamp(0, 1)
        return x, int(self.labels[i])


# ---------------------------------------------------------------------------
# Histogram matching (test-time shift mitigation)
# ---------------------------------------------------------------------------


def reference_histogram(images: np.ndarray, max_images: int = 2000, seed: int = 0) -> np.ndarray:
    """Pooled 256-bin intensity CDF of (a sample of) source training images."""
    rng = np.random.default_rng(seed)
    if len(images) > max_images:
        images = images[rng.choice(len(images), max_images, replace=False)]
    hist = np.bincount(images.ravel(), minlength=256).astype(np.float64)
    return np.cumsum(hist) / hist.sum()


def match_histograms(images: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    """Map each image's intensity CDF onto ``ref_cdf`` (per-image quantile mapping)."""
    out = np.empty_like(images)
    levels = np.arange(256, dtype=np.float64)
    for i, img in enumerate(images):
        hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
        cdf = np.cumsum(hist) / hist.sum()
        mapping = np.interp(cdf, ref_cdf, levels)
        out[i] = np.clip(np.round(mapping[img]), 0, 255).astype(np.uint8)
    return out
