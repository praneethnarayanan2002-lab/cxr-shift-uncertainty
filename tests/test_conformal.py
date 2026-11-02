import numpy as np
import pytest

from cxrshift.conformal import SplitConformal, conformal_quantile, coverage_by_confidence, set_metrics


def _sample(rng, n, shift=0.0):
    y = rng.integers(0, 2, n)
    logit = (2 * y - 1) * 1.5 + rng.normal(0, 1.5, n) + shift
    p1 = 1 / (1 + np.exp(-logit))
    return np.stack([1 - p1, p1], axis=1), y


def test_quantile_finite_sample_rule():
    s = np.arange(1, 10, dtype=float)  # n = 9
    assert conformal_quantile(s, 0.1) == 9.0  # ceil(10 * 0.9) = 9
    assert conformal_quantile(s, 0.5) == 5.0  # ceil(10 * 0.5) = 5
    assert conformal_quantile(s[:3], 0.1) == np.inf


@pytest.mark.parametrize("score", ["lac", "aps"])
@pytest.mark.parametrize("mondrian", [False, True])
def test_coverage_holds_on_exchangeable_data(score, mondrian):
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(30):
        cp_p, cp_y = _sample(rng, 500)
        te_p, te_y = _sample(rng, 2000)
        cp = SplitConformal(0.1, score, mondrian).fit(cp_p, cp_y)
        m = set_metrics(cp.predict_sets(te_p), te_y)
        covs.append(m["coverage"])
        if mondrian:
            covs_c = [m["coverage_class0"], m["coverage_class1"]]
    assert np.mean(covs) >= 0.89
    if mondrian:
        assert min(covs_c) > 0.85


def test_coverage_can_break_under_shift():
    rng = np.random.default_rng(0)
    cal_p, cal_y = _sample(rng, 1000)
    shifted_p, shifted_y = _sample(rng, 4000, shift=-3.0)  # scores pushed towards class 0
    cp = SplitConformal(0.1).fit(cal_p, cal_y)
    m = set_metrics(cp.predict_sets(shifted_p), shifted_y)
    assert m["coverage_class1"] < 0.8


def test_set_metrics_and_confidence_bins():
    sets = np.array([[True, False], [True, True], [False, False], [False, True]])
    y = np.array([0, 1, 0, 0])
    m = set_metrics(sets, y)
    assert m["coverage"] == 0.5
    assert m["avg_set_size"] == 1.0
    assert m["frac_empty"] == 0.25
    probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.6, 0.4], [0.2, 0.8]])
    rows = coverage_by_confidence(sets, y, probs, n_bins=2)
    assert sum(r["n"] for r in rows) == 4
