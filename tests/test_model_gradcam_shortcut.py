import numpy as np
import torch

from cxrshift.gradcam import GradCAM, center_mass, select_examples
from cxrshift.model import build_model
from cxrshift.shortcut import apply_mask, corner_baseline, region_mask


def _tiny_cfg(size=32):
    return {"model": {"backbone": "tiny", "num_labels": 2}, "data": {"image_size": size}}


def test_tiny_model_forward():
    model = build_model(_tiny_cfg())
    out = model(torch.rand(3, 1, 32, 32))
    assert out.shape == (3, 2)


def test_gradcam_shape_and_range():
    torch.manual_seed(0)
    model = build_model(_tiny_cfg())
    cam = GradCAM(model, "backbone.convnext.encoder.stages.-1")
    maps = cam(torch.rand(2, 1, 32, 32))
    cam.remove()
    assert maps.shape == (2, 32, 32)
    assert maps.min() >= 0 and maps.max() <= 1 + 1e-6


def test_select_examples_groups():
    y = np.array([1, 1, 0, 0, 1, 0])
    pred = np.array([1, 0, 0, 1, 1, 0])
    conf = np.array([0.9, 0.6, 0.8, 0.7, 0.95, 0.55])
    g = select_examples(y, pred, conf, k=1)
    assert g == {"TP": [4], "TN": [2], "FP": [3], "FN": [1]}


def test_center_mass():
    cams = np.zeros((1, 20, 20))
    cams[0, 8:12, 8:12] = 1
    assert abs(center_mass(cams)[0] - 1.0) < 1e-6
    cams = np.zeros((1, 20, 20))
    cams[0, 0, 0] = 1
    assert abs(center_mass(cams)[0]) < 1e-6


def test_masks():
    imgs = np.full((2, 20, 20), 200, dtype=np.uint8)
    imgs[:, 0, 0] = 0
    out = apply_mask(imgs, "corners_masked", corner_frac=0.2)
    assert (out[:, :4, :4] == 200).all()  # corner replaced by the mean of kept pixels
    keep = region_mask(20, 20, "corners_only", corner_frac=0.2)
    assert keep.sum() == 4 * 16
    assert np.array_equal(apply_mask(imgs, "full"), imgs)


def test_corner_baseline_finds_a_corner_marker():
    rng = np.random.default_rng(0)

    def make(n, marker):
        x = rng.integers(60, 120, (n, 32, 32)).astype(np.uint8)
        y = rng.integers(0, 2, n)
        if marker:
            x[y == 1, :4, :4] = 255
        return x, y

    tr = make(200, True)
    res = corner_baseline(*tr, {"with_marker": make(200, True), "no_marker": make(200, False)})
    assert res["with_marker"] > 0.95
    assert 0.3 < res["no_marker"] < 0.7
