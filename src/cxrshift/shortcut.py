"""Shortcut checks: does the model rely on non-anatomical image regions?

Two tests:

1. Masking: rerun the trained model on images whose border or corners are replaced by the
   image's mean intensity, and on images where only the corners are kept. If AUROC on
   corner-only images is well above 0.5, the model can tell the classes apart from regions
   that hold text markers, collimation edges and padding rather than lungs.
2. Corner-only baseline: a logistic regression trained on downsampled corner patches of the
   source training images. If this simple model separates the classes, the dataset itself
   contains a shortcut that any classifier could pick up.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def region_mask(h: int, w: int, kind: str, border_frac: float = 0.1, corner_frac: float = 0.2) -> np.ndarray:
    """Boolean (H, W) mask, True where pixels are kept."""
    keep = np.ones((h, w), dtype=bool)
    if kind == "full":
        return keep
    if kind == "border_masked":
        bh, bw = int(round(h * border_frac)), int(round(w * border_frac))
        keep[:] = False
        keep[bh:h - bh, bw:w - bw] = True
        return keep
    ch, cw = int(round(h * corner_frac)), int(round(w * corner_frac))
    corners = np.zeros((h, w), dtype=bool)
    corners[:ch, :cw] = corners[:ch, -cw:] = corners[-ch:, :cw] = corners[-ch:, -cw:] = True
    if kind == "corners_masked":
        return ~corners
    if kind == "corners_only":
        return corners
    raise ValueError(f"Unknown mask kind {kind!r}")


def apply_mask(images: np.ndarray, kind: str, border_frac: float = 0.1, corner_frac: float = 0.2) -> np.ndarray:
    """Replace pixels outside the kept region with each image's mean over the kept region."""
    if kind == "full":
        return images
    keep = region_mask(images.shape[1], images.shape[2], kind, border_frac, corner_frac)
    out = images.copy()
    fill = images[:, keep].mean(axis=1).round().astype(np.uint8)
    out[:, ~keep] = fill[:, None]
    return out


def corner_features(images: np.ndarray, corner_frac: float = 0.2, patch: int = 8) -> np.ndarray:
    """Downsample each of the four corners to ``patch`` x ``patch`` and add simple statistics."""
    n, h, w = images.shape
    ch, cw = int(round(h * corner_frac)), int(round(w * corner_frac))
    x = torch.from_numpy(images).float().div(255.0)[:, None]
    feats = []
    for sl in [(slice(0, ch), slice(0, cw)), (slice(0, ch), slice(w - cw, w)),
               (slice(h - ch, h), slice(0, cw)), (slice(h - ch, h), slice(w - cw, w))]:
        c = x[:, :, sl[0], sl[1]]
        feats.append(F.adaptive_avg_pool2d(c, patch).flatten(1))
        feats.append(torch.stack([c.mean((1, 2, 3)), c.std((1, 2, 3)), c.amax((1, 2, 3))], dim=1))
    return torch.cat(feats, dim=1).numpy()


def corner_baseline(train_images, train_labels, eval_sets: dict, corner_frac: float = 0.2, seed: int = 0) -> dict:
    """Fit logistic regression on corner features; report AUROC on each eval set.

    ``eval_sets`` maps a name to ``(images, labels)``.
    """
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, random_state=seed))
    clf.fit(corner_features(train_images, corner_frac), train_labels)
    out = {}
    for name, (imgs, y) in eval_sets.items():
        p = clf.predict_proba(corner_features(imgs, corner_frac))[:, 1]
        out[name] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return out
