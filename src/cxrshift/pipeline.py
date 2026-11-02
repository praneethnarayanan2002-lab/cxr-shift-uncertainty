"""End-to-end run: load data, fine-tune (or load a checkpoint), evaluate, write results."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .data import load_all
from .evaluate import run_evaluation
from .model import build_model, count_parameters
from .train import train_model
from .utils import ensure_dir, get_device, save_json, set_seed

logger = logging.getLogger("cxrshift.pipeline")


def run(cfg: dict, checkpoint: str | None = None, skip_train: bool = False) -> dict:
    set_seed(int(cfg.get("seed", 0)))
    out_dir = ensure_dir(Path(cfg.get("output_dir", "results")) / cfg.get("run_name", "run"))
    save_json(cfg, out_dir / "config_used.json")
    device = get_device(cfg.get("device", "auto"))
    logger.info("Device: %s, output: %s", device, out_dir)

    data = load_all(cfg)
    model = build_model(cfg).to(device)
    logger.info("Model parameters: %.2fM", count_parameters(model) / 1e6)

    train_info = None
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    elif not skip_train:
        train_info = train_model(model, data["source_train"], data["source_val"], cfg, device, out_dir)
        save_json(train_info, out_dir / "train_history.json")

    results = run_evaluation(cfg, model, data, device, out_dir)
    results["train"] = train_info
    save_json(results, out_dir / "results.json")
    return results
