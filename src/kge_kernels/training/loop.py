"""Lean KGE training loop and its helpers.

This module owns the shared primitives that any pure-KGE training
pipeline needs:

  - ``TripleDataset``               — thin tensor-backed torch Dataset
  - ``set_seed``                    — reproducible Python + torch RNG
  - ``make_cosine_warmup_scheduler`` — LambdaLR factory
  - ``wrap_model_for_training``     — optional DataParallel + compile wrap
  - ``train_kge``                   — one-function KGE training loop

``train_kge`` is deliberately lean: it runs the forward/loss/backward/
step inner loop and lets the caller handle validation, early stopping,
metric logging, and checkpointing via a single ``on_epoch_end`` callback.
Consumers that want a richer loop (DpRL, future pipelines) can compose
the other primitives directly.
"""
from __future__ import annotations

import math
import random
import time
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from .checkpoints import unwrap_model as _unwrap
from .config import KGETrainConfig


class TripleDataset(Dataset):
    """Tensor-backed ``Dataset`` over integer triples.

    Stores triples as a ``[N, 3]`` long tensor and returns one row per
    ``__getitem__`` call. Triple column order is whatever the caller
    provides; KGE convention in this package is ``(r, h, t)``.
    """

    def __init__(self, triples: List[Tuple[int, int, int]]) -> None:
        self.triples = torch.tensor(triples, dtype=torch.long)

    def __len__(self) -> int:
        return self.triples.size(0)

    def __getitem__(self, idx: int) -> Tensor:
        return self.triples[idx]


def set_seed(seed: int) -> None:
    """Seed Python, torch, and CUDA RNGs for reproducible training."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    """Linear warmup → cosine decay ``LambdaLR`` scheduler.

    Args:
        optimizer: Optimizer to schedule.
        total_steps: Total number of optimizer steps over training.
        warmup_ratio: Fraction of ``total_steps`` used for linear warmup.

    The resulting learning-rate multiplier is ``step/warmup`` during
    warmup and ``0.5 * (1 + cos(pi * progress))`` afterwards.
    """
    total_steps = max(1, total_steps)
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def wrap_model_for_training(
    model: nn.Module,
    device: torch.device,
    cfg: KGETrainConfig,
) -> nn.Module:
    """Apply optional DataParallel and ``torch.compile`` wrappers.

    The unwrapped model is always moved to ``device`` first. If
    ``cfg.multi_gpu`` is set and at least 2 CUDA devices are available,
    the model is wrapped in ``nn.DataParallel``. If ``cfg.compile`` is
    set, the result is then passed through ``torch.compile`` with
    ``cfg.compile_mode`` and ``cfg.compile_fullgraph``.

    Both transformations match DpRL's ``build_training_model`` semantics
    so existing checkpoints produced by that pipeline remain loadable.
    """
    model = model.to(device)

    if cfg.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    elif cfg.multi_gpu:
        # Requested but unavailable — proceed single-device silently.
        pass

    if cfg.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")
        if cfg.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
            raise RuntimeError(
                "torch.compile with DataParallel is not supported in this path"
            )
        model = torch.compile(
            model, mode=cfg.compile_mode, fullgraph=cfg.compile_fullgraph
        )
    return model


# ═══════════════════════════════════════════════════════════════════════
# train_kge — the lean training loop
# ═══════════════════════════════════════════════════════════════════════


OnEpochEnd = Callable[[int, float, nn.Module, float], bool]
"""Callback invoked after each epoch.

Args:
    epoch: 1-based epoch index.
    avg_loss: Mean training loss across the epoch.
    model: The training model (still in training mode).
    epoch_time: Wall-clock seconds for the epoch.

Return ``True`` to stop training early (e.g. for early stopping),
``False`` to continue. ``None`` is treated as ``False``.
"""


def train_kge(
    cfg: KGETrainConfig,
    model: nn.Module,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    sample_negatives: Callable[[Tensor], Tensor],
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: Optional[torch.device] = None,
    on_epoch_end: Optional[OnEpochEnd] = None,
) -> List[float]:
    """Lean KGE training loop.

    Runs ``cfg.epochs`` passes over ``dataloader``. For each batch:

      1. Draw ``cfg.neg_ratio`` corruptions per positive via
         ``sample_negatives(batch)`` (consumer supplies the sampler).
      2. Compute positive and negative scores via ``model(h, r, t)``.
      3. Compute loss via ``loss_fn(pos_scores, neg_scores)``.
      4. Backward + optimizer step with optional AMP, grad clipping,
         and scheduler step.
      5. Call model's ``project_entity_modulus_()`` if present (RotatE).

    After each epoch, ``on_epoch_end(epoch, avg_loss, model)`` is called
    if provided; returning ``True`` from the callback stops training
    early. The callback is where consumers plug in validation,
    early stopping, checkpoint saving, and metric logging.

    Args:
        cfg: ``KGETrainConfig`` (uses ``epochs``, ``amp``, ``grad_clip``).
        model: Unwrapped KGE model with a ``forward(h, r, t)`` method.
            If DataParallel- or compile-wrapped, the loop still works.
        dataloader: Yields ``[B, 3]`` long tensors in ``(r, h, t)`` order.
        optimizer: Any torch optimizer.
        loss_fn: ``(pos_scores, neg_scores) -> scalar loss``.
        sample_negatives: ``batch -> [B * neg_ratio, 3]`` negatives,
            in the same ``(r, h, t)`` column order as ``dataloader``.
        scheduler: Optional LR scheduler (stepped per batch).
        scaler: Optional ``GradScaler`` (if ``cfg.amp``, a default
            ``torch.amp.GradScaler("cuda", enabled=cfg.amp)`` is created).
        device: Inference device. Defaults to the first parameter's
            device.
        on_epoch_end: Optional callback ``(epoch, avg_loss, model, epoch_time)``
            for validation / early stopping.

    Returns:
        ``List[float]`` of per-epoch mean losses (partial list if the
        callback triggered early stopping).
    """
    if device is None:
        device = next(model.parameters()).device
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    model.train()
    epoch_losses: List[float] = []

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        running = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            negatives = sample_negatives(batch)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                pos_scores = model(batch[:, 1], batch[:, 0], batch[:, 2])
                neg_scores = model(
                    negatives[:, 1], negatives[:, 0], negatives[:, 2]
                )
                loss = loss_fn(pos_scores, neg_scores)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            # RotatE-style entity modulus projection, if the model exposes it.
            inner = _unwrap(model)
            if hasattr(inner, "project_entity_modulus_"):
                inner.project_entity_modulus_()

            running += loss.item()
            n_batches += 1

        avg_loss = running / max(1, n_batches)
        epoch_losses.append(avg_loss)
        epoch_time = time.perf_counter() - epoch_start

        if on_epoch_end is not None:
            should_stop = on_epoch_end(epoch, avg_loss, model, epoch_time)
            if should_stop:
                break

    return epoch_losses


__all__ = [
    "OnEpochEnd",
    "TripleDataset",
    "make_cosine_warmup_scheduler",
    "set_seed",
    "train_kge",
    "wrap_model_for_training",
]
