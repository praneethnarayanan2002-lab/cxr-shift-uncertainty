"""Split conformal prediction for classification.

Scores:
* ``lac``: 1 - p(true class). Gives the smallest average sets.
* ``aps``: adaptive prediction sets (cumulative probability mass up to the true class),
  non-randomised version.

``mondrian=True`` calibrates a separate quantile per true class, which targets coverage within
each class rather than only on average.

The guarantee only holds when calibration and test data are exchangeable. Calibrating on the
source and testing on the target breaks that assumption, which is exactly what we measure.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    n = len(scores)
    if n == 0:
        raise ValueError("Empty calibration set")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def _all_scores(probs: np.ndarray, score: str) -> np.ndarray:
    """Nonconformity score of every candidate label, shape (N, C)."""
    if score == "lac":
        return 1.0 - probs
    if score == "aps":
        order = np.argsort(-probs, axis=1)
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        out = np.empty_like(probs)
        np.put_along_axis(out, order, cum, axis=1)
        return out
    raise ValueError(f"Unknown conformal score {score!r}")


class SplitConformal:
    def __init__(self, alpha: float = 0.1, score: str = "lac", mondrian: bool = False):
        self.alpha = float(alpha)
        self.score = score
        self.mondrian = mondrian
        self.qhat: np.ndarray | None = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "SplitConformal":
        s = _all_scores(probs, self.score)[np.arange(len(labels)), labels]
        n_classes = probs.shape[1]
        if self.mondrian:
            self.qhat = np.array([conformal_quantile(s[labels == c], self.alpha) for c in range(n_classes)])
        else:
            self.qhat = np.full(n_classes, conformal_quantile(s, self.alpha))
        return self

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        """Boolean matrix (N, C): True where the label is in the prediction set."""
        if self.qhat is None:
            raise RuntimeError("Call fit first")
        return _all_scores(probs, self.score) <= self.qhat[None, :]


def set_metrics(sets: np.ndarray, labels: np.ndarray) -> dict:
    covered = sets[np.arange(len(labels)), labels]
    size = sets.sum(axis=1)
    out = {
        "n": int(len(labels)),
        "coverage": float(covered.mean()),
        "avg_set_size": float(size.mean()),
        "frac_empty": float((size == 0).mean()),
        "frac_singleton": float((size == 1).mean()),
        "frac_full": float((size == sets.shape[1]).mean()),
        "singleton_accuracy": float(covered[size == 1].mean()) if (size == 1).any() else float("nan"),
    }
    for c in range(sets.shape[1]):
        m = labels == c
        out[f"coverage_class{c}"] = float(covered[m].mean()) if m.any() else float("nan")
        out[f"avg_set_size_class{c}"] = float(size[m].mean()) if m.any() else float("nan")
    return out


def coverage_by_confidence(sets: np.ndarray, labels: np.ndarray, probs: np.ndarray, n_bins: int = 5) -> list[dict]:
    """Coverage within quantile bins of top-class confidence. Shows where coverage breaks."""
    conf = probs.max(axis=1)
    covered = sets[np.arange(len(labels)), labels]
    edges = np.unique(np.quantile(conf, np.linspace(0, 1, n_bins + 1)))
    ids = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = ids == b
        if not m.any():
            continue
        rows.append({
            "bin": b,
            "conf_lo": float(edges[b]),
            "conf_hi": float(edges[b + 1]),
            "n": int(m.sum()),
            "coverage": float(covered[m].mean()),
            "avg_set_size": float(sets[m].sum(axis=1).mean()),
        })
    return rows


def coverage_curve(cal_probs, cal_labels, test_sets: dict, alphas, score="lac", mondrian=False) -> dict:
    """Empirical coverage and set size for several alphas on several test sets.

    ``test_sets`` maps a name to ``(probs, labels)``.
    """
    out = {name: {"alpha": [], "coverage": [], "avg_set_size": []} for name in test_sets}
    for a in alphas:
        cp = SplitConformal(a, score, mondrian).fit(cal_probs, cal_labels)
        for name, (p, y) in test_sets.items():
            m = set_metrics(cp.predict_sets(p), y)
            out[name]["alpha"].append(float(a))
            out[name]["coverage"].append(m["coverage"])
            out[name]["avg_set_size"].append(m["avg_set_size"])
    return out


def plot_coverage_curve(ax_cov, ax_size, curve: dict) -> None:
    for name, d in curve.items():
        target = 1 - np.asarray(d["alpha"])
        ax_cov.plot(target, d["coverage"], marker="o", ms=3, label=name)
        ax_size.plot(target, d["avg_set_size"], marker="o", ms=3, label=name)
    lims = [min(min(1 - np.asarray(d["alpha"])) for d in curve.values()), 1.0]
    ax_cov.plot(lims, lims, "k--", lw=1, label="nominal")
    ax_cov.set_xlabel("target coverage (1 - alpha)")
    ax_cov.set_ylabel("empirical coverage")
    ax_cov.legend(fontsize=7)
    ax_size.set_xlabel("target coverage (1 - alpha)")
    ax_size.set_ylabel("average set size")
    ax_size.legend(fontsize=7)
