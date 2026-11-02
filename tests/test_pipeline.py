"""End-to-end run on synthetic data with the tiny model. Checks plumbing, not performance."""

import json

import pandas as pd

from cxrshift.pipeline import run


def test_smoke_pipeline_writes_all_outputs(smoke_cfg, tmp_path):
    results = run(smoke_cfg)
    out = tmp_path / "smoke"
    for f in ["metrics.csv", "conformal.csv", "shortcut.csv", "results.json", "results_table.md",
              "train_history.json", "best_model.pt", "config_used.json"]:
        assert (out / f).exists(), f
    for f in ["reliability.png", "conformal_coverage.png", "gradcam_source_test.png",
              "gradcam_target_test.png", "shortcut_check.png"]:
        assert (out / "figures" / f).exists(), f

    metrics = pd.read_csv(out / "metrics.csv")
    assert set(metrics["dataset"]) == {"source_test", "target_test", "target_test_histmatch"}
    assert results["temperature"] > 0
    assert 0 <= results["threshold"] <= 1
    conformal = pd.read_csv(out / "conformal.csv")
    assert set(conformal["score"]) == {"lac", "aps"}
    assert conformal["coverage"].between(0, 1).all()

    # The synthetic source puts a bright marker in the corner of positive images only.
    # The corner-only logistic regression should detect it on the source but not the target.
    shortcut = pd.read_csv(out / "shortcut.csv")
    lr = shortcut[shortcut["test"] == "logistic regression on corners"].set_index("dataset")["auroc"]
    assert lr["source_test"] > 0.9
    assert lr["target_test"] < 0.8

    saved = json.loads((out / "results.json").read_text())
    assert saved["train"]["best_epoch"] >= 1


def test_evaluate_from_checkpoint(smoke_cfg, tmp_path):
    run(smoke_cfg)
    ckpt = tmp_path / "smoke" / "best_model.pt"
    smoke_cfg["run_name"] = "from_ckpt"
    smoke_cfg["gradcam"]["enabled"] = False
    smoke_cfg["shortcut"]["enabled"] = False
    res = run(smoke_cfg, checkpoint=str(ckpt))
    assert res["train"] is None
    assert (tmp_path / "from_ckpt" / "metrics.csv").exists()
