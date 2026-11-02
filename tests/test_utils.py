import torch

from cxrshift.utils import apply_overrides, deep_merge, get_module_by_path, load_config
from conftest import ROOT


def test_deep_merge_keeps_unrelated_keys():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"b": 10}})
    assert out == {"a": {"b": 10, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1


def test_overrides_parse_types():
    cfg = apply_overrides({"train": {"epochs": 1}}, ["train.epochs=5", "train.lr=1e-3", "x.y=null", "z=[1, 2]"])
    assert cfg["train"]["epochs"] == 5
    assert abs(cfg["train"]["lr"] - 1e-3) < 1e-12
    assert cfg["x"]["y"] is None
    assert cfg["z"] == [1, 2]


def test_config_inheritance():
    cfg = load_config(ROOT / "configs" / "strong_aug.yaml")
    assert cfg["train"]["augment"] == "strong"
    assert cfg["data"]["source"]["name"] == "pneumoniamnist"
    assert cfg["model"]["name"] == "facebook/convnext-tiny-224"
    assert "base" not in cfg


def test_get_module_by_path_negative_index():
    m = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1))
    assert get_module_by_path(m, "-1") is m[2]
