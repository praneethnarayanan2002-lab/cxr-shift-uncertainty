"""Calibration metrics, reliability diagrams and temperature scaling."""

from __future__ import annotations

import numpy as np
import torch


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _bin_stats(conf: np.ndarray, correct: np.ndarray, n_bins: int):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    counts = np.bincount(ids, minlength=n_bins)
    acc = np.bincount(ids, weights=correct, minlength=n_bins) / np.maximum(counts, 1)
    avg_conf = np.bincount(ids, weights=conf, minlength=n_bins) / np.maximum(counts, 1)
    return edges, counts, acc, avg_conf


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins. ``probs`` has shape (N, C)."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    _, counts, acc, avg_conf = _bin_stats(conf, correct, n_bins)
    return float(np.sum(counts / max(1, len(labels)) * np.abs(acc - avg_conf)))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score of the positive-class probability."""
    return float(np.mean((probs[:, 1] - labels) ** 2))


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Fit a single temperature by minimising NLL with L-BFGS. Optimises log T so T > 0."""
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def calibration_summary(logits: np.ndarray, labels: np.ndarray, temperature: float, n_bins: int = 15) -> dict:
    raw = softmax(logits)
    cal = softmax(logits, temperature)
    return {
        "ece_raw": expected_calibration_error(raw, labels, n_bins),
        "ece_ts": expected_calibration_error(cal, labels, n_bins),
        "brier_raw": brier_score(raw, labels),
        "brier_ts": brier_score(cal, labels),
        "nll_raw": nll(raw, labels),
        "nll_ts": nll(cal, labels),
    }


def plot_reliability(ax, probs: np.ndarray, labels: np.ndarray, n_bins: int = 15, title: str = "") -> None:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges, counts, acc, avg_conf = _bin_stats(conf, correct, n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    mask = counts > 0
    ax.bar(centers[mask], acc[mask], width=width * 0.9, color="#4C72B0", alpha=0.8, label="accuracy")
    ax.bar(centers[mask], avg_conf[mask] - acc[mask], bottom=acc[mask], width=width * 0.9,
           color="#DD8452", alpha=0.5, label="gap")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ece = expected_calibration_error(probs, labels, n_bins)
    ax.set_title(f"{title}\nECE = {ece:.3f}", fontsize=9)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left")
