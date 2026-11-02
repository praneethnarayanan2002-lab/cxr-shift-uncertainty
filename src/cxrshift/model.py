"""Model construction.

``model.backbone: hf`` loads a pretrained Hugging Face image classifier (default
facebook/convnext-tiny-224) and replaces its head with a 2-class head.
``model.backbone: tiny`` builds a small randomly initialised ConvNeXt from a config, with no
download. It shares the code path of the real model, so tests exercise the same Grad-CAM
target layer and forward pass.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("cxrshift.model")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CXRClassifier(nn.Module):
    """Wraps a Hugging Face classifier.

    Input: grayscale batch (B, 1, H, W) in [0, 1]. The wrapper repeats it to 3 channels and
    applies the normalisation the backbone was pretrained with. Output: logits (B, 2).
    Keeping inputs in [0, 1] makes masking and histogram matching easy to reason about.
    """

    def __init__(self, hf_model: nn.Module, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.backbone = hf_model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        return self.backbone(pixel_values=x).logits


def _tiny_convnext(num_labels: int, image_size: int):
    from transformers import ConvNextConfig, ConvNextForImageClassification

    config = ConvNextConfig(
        num_channels=3,
        patch_size=4,
        num_stages=2,
        hidden_sizes=[8, 16],
        depths=[1, 1],
        image_size=image_size,
        num_labels=num_labels,
    )
    return ConvNextForImageClassification(config)


def build_model(cfg: dict) -> CXRClassifier:
    mcfg = cfg["model"]
    num_labels = int(mcfg.get("num_labels", 2))
    backbone = mcfg.get("backbone", "hf")
    if backbone == "tiny":
        hf_model = _tiny_convnext(num_labels, int(cfg["data"]["image_size"]))
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif backbone == "hf":
        from transformers import AutoModelForImageClassification

        name = mcfg["name"]
        logger.info("Loading pretrained model %s", name)
        hf_model = AutoModelForImageClassification.from_pretrained(
            name,
            num_labels=num_labels,
            id2label={0: "negative", 1: "pneumonia"},
            label2id={"negative": 0, "pneumonia": 1},
            ignore_mismatched_sizes=True,
        )
        mean = tuple(mcfg.get("mean", IMAGENET_MEAN))
        std = tuple(mcfg.get("std", IMAGENET_STD))
    else:
        raise ValueError(f"Unknown model.backbone {backbone!r}, use 'hf' or 'tiny'")
    return CXRClassifier(hf_model, mean, std)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
