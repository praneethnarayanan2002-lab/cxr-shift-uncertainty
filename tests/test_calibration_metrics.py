import numpy as np

from cxrshift.calibration import calibration_summary, expected_calibration_error, fit_temperature, softmax
from cxrshift.metrics import binary_metrics, bootstrap_ci, choose_threshold


def test_ece_zero_for_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(0)
    n = 20000
    p1 = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p1).astype(int)
    probs = np.stack([1 - p1, p1], axis=1)
    assert expected_calibration_error(probs, y) < 0.02
    sharp = np.stack([(p1 < 0.5).astype(float), (p1 >= 0.5).astype(float)], axis=1)
    sharp = np.clip(sharp, 0.001, 0.999)
    assert expected_calibration_error(sharp, y) > 0.15


def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(1)
    n = 20000
    true_logits = rng.normal(0, 2, (n, 1)) * np.array([[-1, 1]]) / 2
    p = softmax(true_logits)[:, 1]
    y = (rng.uniform(size=n) < p).astype(int)
    overconfident = true_logits * 3.0
    t = fit_temperature(overconfident, y)
    assert 2.6 < t < 3.4
    s = calibration_summary(overconfident, y, t)
    assert s["ece_ts"] < s["ece_raw"]
    assert s["nll_ts"] < s["nll_raw"]


def test_binary_metrics_counts():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.7])
    m = binary_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert abs(m["sensitivity"] - 2 / 3) < 1e-9
    assert abs(m["specificity"] - 0.5) < 1e-9


def test_threshold_selection():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)])
    t = choose_threshold(y, p, "youden")
    assert 0.4 < t <= 0.6
    t_sens = choose_threshold(y, p, "sensitivity", 0.9)
    assert ((p >= t_sens)[y == 1]).mean() >= 0.9


def test_bootstrap_ci_contains_point_estimate():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 300), 0, 1)
    lo, hi = bootstrap_ci(y, p, roc_auc_score, 200)
    assert lo <= roc_auc_score(y, p) <= hi
