"""Discrimination metrics, threshold selection and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def choose_threshold(y: np.ndarray, p: np.ndarray, method: str = "youden", target_sensitivity: float = 0.9) -> float:
    """Pick a decision threshold on validation data.

    ``youden`` maximises sensitivity + specificity - 1. ``sensitivity`` returns the highest
    threshold whose sensitivity is at least ``target_sensitivity``.
    """
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.clip(thr, 0.0, 1.0)  # sklearn sets thr[0] = inf
    if method == "youden":
        return float(thr[np.argmax(tpr - fpr)])
    if method == "sensitivity":
        ok = np.flatnonzero(tpr >= target_sensitivity)
        return float(thr[ok[0]]) if len(ok) else float(thr[-1])
    if method == "fixed":
        return 0.5
    raise ValueError(f"Unknown threshold method {method!r}")


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "auprc": float(average_precision_score(y, p)) if both else float("nan"),
        "accuracy": (tp + tn) / max(1, len(y)),
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "ppv": tp / max(1, tp + fp),
        "threshold": float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric_fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for ``metric_fn(y, p)``. Resamples that miss a class are skipped."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals.append(metric_fn(yb, p[idx]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
