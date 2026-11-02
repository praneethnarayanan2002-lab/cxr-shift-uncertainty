"""Grad-CAM for convolutional backbones.

The target layer is set by ``gradcam.target_layer`` as a dotted path inside the model, for
example ``backbone.convnext.encoder.stages.-1`` for Hugging Face ConvNeXt. The layer must
output a (B, C, H, W) feature map. Transformer backbones would need a reshape step that is
not implemented here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .utils import get_module_by_path


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.layer = get_module_by_path(model, target_layer)
        self._acts = None
        self._grads = None
        self._handles = [
            self.layer.register_forward_hook(self._save_acts),
            self.layer.register_full_backward_hook(self._save_grads),
        ]

    def _save_acts(self, _module, _inp, out):
        out = out[0] if isinstance(out, (tuple, list)) else out
        self._acts = out

    def _save_grads(self, _module, _grad_in, grad_out):
        self._grads = grad_out[0]

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | torch.Tensor | None = None) -> np.ndarray:
        """Return heatmaps in [0, 1] with shape (B, H, W) at the input resolution.

        ``class_idx`` defaults to the predicted class of each image.
        """
        self.model.eval()
        x = x.clone().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(dim=1)
            elif isinstance(class_idx, int):
                class_idx = torch.full((len(x),), class_idx, device=x.device, dtype=torch.long)
            selected = logits.gather(1, class_idx.view(-1, 1)).sum()
            selected.backward()
        acts, grads = self._acts.detach(), self._grads.detach()
        if acts.dim() != 4:
            raise ValueError(f"Grad-CAM needs a (B, C, H, W) feature map, got shape {tuple(acts.shape)}")
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.view(len(cam), -1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        cam = ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(cam)
        return cam.cpu().numpy()


def select_examples(labels: np.ndarray, preds: np.ndarray, conf: np.ndarray, k: int, seed: int = 0) -> dict:
    """Pick up to ``k`` indices for each outcome (TP, TN, FP, FN).

    Uses the most confident cases in each group, which are the most informative for errors
    (confident mistakes) and for checking what a confident correct call looks at.
    """
    groups = {
        "TP": (labels == 1) & (preds == 1),
        "TN": (labels == 0) & (preds == 0),
        "FP": (labels == 0) & (preds == 1),
        "FN": (labels == 1) & (preds == 0),
    }
    out = {}
    for name, mask in groups.items():
        idx = np.flatnonzero(mask)
        out[name] = idx[np.argsort(-conf[idx])][:k].tolist()
    return out


def center_mass(cams: np.ndarray, border_frac: float = 0.15) -> np.ndarray:
    """Fraction of heatmap mass inside the central region (excluding a border of ``border_frac``).

    A low value means the explanation sits on the image border, where labels and markers are.
    """
    h, w = cams.shape[-2:]
    bh, bw = int(round(h * border_frac)), int(round(w * border_frac))
    inner = cams[:, bh:h - bh, bw:w - bw].sum(axis=(1, 2))
    total = cams.sum(axis=(1, 2)) + 1e-8
    return inner / total


def plot_gradcam_grid(fig, images: np.ndarray, cams: np.ndarray, titles: list[str], ncols: int) -> None:
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    for i in range(n):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        ax.imshow(cams[i], cmap="jet", alpha=0.35, vmin=0, vmax=1)
        ax.set_title(titles[i], fontsize=7)
        ax.axis("off")
