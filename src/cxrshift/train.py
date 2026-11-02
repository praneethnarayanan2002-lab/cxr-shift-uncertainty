"""Fine-tuning with early stopping on the source validation split, and batched inference."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .data import CXRDataset, SplitData, build_augmentation

logger = logging.getLogger("cxrshift.train")


def make_loader(split: SplitData, cfg: dict, train: bool) -> DataLoader:
    tcfg = cfg["train"]
    augment = build_augmentation(tcfg.get("augment", "basic")) if train else None
    ds = CXRDataset(split, cfg["data"]["image_size"], augment=augment)
    return DataLoader(
        ds,
        batch_size=int(tcfg.get("batch_size", 32)) if train else int(tcfg.get("eval_batch_size", 64)),
        shuffle=train,
        num_workers=int(tcfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=torch.Generator().manual_seed(int(cfg.get("seed", 0))) if train else None,
    )


@torch.no_grad()
def predict_logits(model: nn.Module, split: SplitData, cfg: dict, device) -> np.ndarray:
    """Return raw logits (N, 2) for a split, in order."""
    model.eval()
    loader = make_loader(split, cfg, train=False)
    out = []
    use_amp = device.type == "cuda" and cfg["train"].get("amp", True)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        out.append(logits.float().cpu())
    return torch.cat(out).numpy()


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    w = counts.sum() / (2 * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def _val_auroc(model, val: SplitData, cfg, device) -> tuple[float, float]:
    logits = predict_logits(model, val, cfg, device)
    loss = nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(val.labels)).item()
    probs = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()
    if len(np.unique(val.labels)) < 2:
        return float("nan"), loss
    return float(roc_auc_score(val.labels, probs)), loss


def train_model(model: nn.Module, train: SplitData, val: SplitData, cfg: dict, device, out_dir) -> dict:
    """Fine-tune ``model`` in place. Keeps the weights with the best validation AUROC.

    Returns a history dict and writes ``best_model.pt`` into ``out_dir``.
    """
    tcfg = cfg["train"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    loader = make_loader(train, cfg, train=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg.get("lr", 1e-4)), weight_decay=float(tcfg.get("weight_decay", 0.05))
    )
    epochs = int(tcfg.get("epochs", 10))
    steps = max(1, epochs * len(loader))
    warmup = int(float(tcfg.get("warmup_frac", 0.05)) * steps)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    weight = _class_weights(train.labels).to(device) if tcfg.get("class_weighted", True) else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    use_amp = device.type == "cuda" and tcfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    patience = int(tcfg.get("patience", 3))
    max_steps = tcfg.get("max_steps_per_epoch")

    best_auc, best_epoch, bad_epochs = -np.inf, -1, 0
    history = []
    ckpt = out_dir / "best_model.pt"
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running, n_seen = 0.0, 0
        for step, (x, y) in enumerate(loader):
            if max_steps is not None and step >= int(max_steps):
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(y)
            n_seen += len(y)
        val_auc, val_loss = _val_auroc(model, val, cfg, device)
        rec = {
            "epoch": epoch + 1,
            "train_loss": running / max(1, n_seen),
            "val_loss": val_loss,
            "val_auroc": val_auc,
            "seconds": time.time() - t0,
        }
        history.append(rec)
        logger.info(
            "epoch %d  train_loss %.4f  val_loss %.4f  val_auroc %.4f  (%.0fs)",
            rec["epoch"], rec["train_loss"], val_loss, val_auc, rec["seconds"],
        )
        score = val_auc if not np.isnan(val_auc) else -val_loss
        if score > best_auc:
            best_auc, best_epoch, bad_epochs = score, epoch + 1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("Early stopping after epoch %d (best epoch %d)", epoch + 1, best_epoch)
                break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return {"history": history, "best_epoch": best_epoch, "best_val_auroc": best_auc, "checkpoint": str(ckpt)}
