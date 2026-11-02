"""Config handling, seeding, device selection and small helpers."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger("cxrshift")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_scalar(text: str) -> Any:
    """Parse a command line override value using YAML rules (ints, floats, bools, null, lists)."""
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``key.sub=value`` overrides to a config dict."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(value)
    return cfg


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. A ``base`` key points to another YAML file that is merged first."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is not None:
        base_path = (path.parent / base).resolve()
        cfg = deep_merge(load_config(base_path), cfg)
    return apply_overrides(cfg, overrides)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass


def get_device(preference: str = "auto"):
    import torch

    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def get_module_by_path(root, path: str):
    """Return a submodule from a dotted path. Integer parts index into lists, negatives allowed.

    Example: ``"backbone.convnext.encoder.stages.-1"``.
    """
    node = root
    for part in path.split("."):
        if part.lstrip("-").isdigit():
            node = node[int(part)]
        else:
            node = getattr(node, part)
    return node


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)


def ensure_dir(path: str | os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
