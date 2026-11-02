import numpy as np
import pytest
import torch
from PIL import Image

from cxrshift.data import (
    CXRDataset, build_augmentation, load_split, map_to_binary, match_histograms,
    reference_histogram, split_for_calibration, SplitData,
)

medmnist = pytest.importorskip("medmnist")


def test_medmnist_label_names_are_what_the_config_expects():
    pneu = medmnist.INFO["pneumoniamnist"]["label"]
    chest = medmnist.INFO["chestmnist"]["label"]
    assert {v.lower() for v in pneu.values()} == {"normal", "pneumonia"}
    assert "pneumonia" in {v.lower() for v in chest.values()}
    assert medmnist.INFO["chestmnist"]["task"].startswith("multi-label")


def test_binary_mapping_single_label():
    names = medmnist.INFO["pneumoniamnist"]["label"]
    pneu_idx = int([k for k, v in names.items() if v == "pneumonia"][0])
    raw = np.array([[pneu_idx], [1 - pneu_idx], [pneu_idx]])
    keep, y = map_to_binary(raw, names, "binary-class", ["pneumonia"], ["normal"])
    assert keep.all()
    assert y.tolist() == [1, 0, 1]


def test_binary_mapping_multilabel_no_finding():
    info = medmnist.INFO["chestmnist"]
    names = info["label"]
    n = len(names)
    pidx = int([k for k, v in names.items() if v == "pneumonia"][0])
    other = (pidx + 1) % n
    raw = np.zeros((4, n), dtype=int)
    raw[0, pidx] = 1                      # pneumonia only -> positive
    raw[1, [pidx, other]] = 1             # pneumonia + other -> positive (non-exclusive)
    raw[2, other] = 1                     # other finding only -> dropped
    # raw[3] all zero -> no finding -> negative
    keep, y = map_to_binary(raw, names, info["task"], ["pneumonia"], "no_finding")
    assert keep.tolist() == [True, True, False, True]
    assert y[keep].tolist() == [1, 1, 0]
    keep_ex, _ = map_to_binary(raw, names, info["task"], ["pneumonia"], "no_finding", exclusive_positive=True)
    assert keep_ex.tolist() == [True, False, False, True]


def test_unknown_label_name_fails_loudly():
    names = medmnist.INFO["chestmnist"]["label"]
    with pytest.raises(KeyError):
        map_to_binary(np.zeros((1, len(names))), names, "multi-label, binary-class", ["pneumonitis"], "no_finding")


def test_medmnist_loader_reads_local_npz(tmp_path):
    """Write a fake chestmnist_64.npz and load it through the real medmnist class."""
    names = medmnist.INFO["chestmnist"]["label"]
    pidx = int([k for k, v in names.items() if v == "pneumonia"][0])
    rng = np.random.default_rng(0)
    arrays = {}
    for split in ("train", "val", "test"):
        labels = np.zeros((10, len(names)), dtype=np.uint8)
        labels[:3, pidx] = 1
        labels[3:5, (pidx + 2) % len(names)] = 1
        arrays[f"{split}_images"] = rng.integers(0, 255, (10, 64, 64), dtype=np.uint8)
        arrays[f"{split}_labels"] = labels
    np.savez(tmp_path / "chestmnist_64.npz", **arrays)
    cfg = {"loader": "medmnist", "name": "chestmnist", "size": 64, "root": str(tmp_path), "download": False,
           "label_map": {"positive": ["pneumonia"], "negative": "no_finding"}}
    d = load_split(cfg, "test")
    assert len(d) == 8 and d.labels.sum() == 3
    assert d.images.shape == (8, 64, 64) and d.images.dtype == np.uint8


def test_folder_loader(tmp_path):
    for split in ("train", "test"):
        for cls, n in (("NORMAL", 3), ("PNEUMONIA", 2), ("OTHER", 1)):
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(n):
                Image.fromarray(np.full((50, 40, 3), 20 * i, dtype=np.uint8)).save(d / f"{i}.png")
    cfg = {"loader": "folder", "root": str(tmp_path), "size": 32,
           "label_map": {"positive": ["PNEUMONIA"], "negative": ["NORMAL"]}}
    d = load_split(cfg, "train")
    assert d.images.shape == (5, 32, 32)
    assert d.labels.sum() == 2


def test_calibration_split_is_disjoint_and_stratified():
    labels = np.array([0] * 40 + [1] * 60)
    val = SplitData(np.arange(100)[:, None, None].astype(np.uint8), labels)
    fit, cal = split_for_calibration(val, 0.5, seed=0)
    assert len(fit) + len(cal) == 100
    assert not set(fit.images.ravel()) & set(cal.images.ravel())
    assert cal.labels.sum() == 30


def test_dataset_resizes_and_augments():
    split = SplitData(np.random.default_rng(0).integers(0, 255, (4, 40, 40), dtype=np.uint8), np.array([0, 1, 0, 1]))
    for strength in ("none", "basic", "strong"):
        ds = CXRDataset(split, 32, build_augmentation(strength))
        x, y = ds[1]
        assert x.shape == (1, 32, 32) and y == 1
        assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0


def test_histogram_matching_moves_intensity_towards_reference():
    rng = np.random.default_rng(0)
    src = rng.normal(150, 20, (20, 32, 32)).clip(0, 255).astype(np.uint8)
    tgt = rng.normal(80, 10, (5, 32, 32)).clip(0, 255).astype(np.uint8)
    matched = match_histograms(tgt, reference_histogram(src))
    assert abs(matched.mean() - src.mean()) < 5
    assert abs(tgt.mean() - src.mean()) > 50
    # rank order of pixels within an image is preserved
    a, b = tgt[0].ravel().astype(int), matched[0].ravel().astype(int)
    order = np.argsort(a, kind="stable")
    assert np.all(np.diff(b[order]) >= 0)
