"""tkk training-loop body — extracted from the monolithic ``pipeline()``.

Calls :func:`kge_kernels.training.epoch.train_epoch` once per epoch
and fires the validation/early-stop hook on
:class:`~kge_kernels.training.builder.Callbacks`. Loss switch (BCE
default vs NSSA) is driven by ``cfg.loss`` via the ``train_step``
override on ``train_epoch``.

The function exists as its own file because the body is long enough
that putting it in ``builder.py`` would dilute the build_* factories
there. ``experiment.pipeline`` calls this directly.
"""
from __future__ import annotations

import time
from functools import partial
from typing import Any, Dict

import torch

from .builder import Callbacks, DataBundle, OptimBundle
from .checkpoints import save_latest_weights, unwrap_model
from .config import TrainConfig
from .epoch import train_epoch
from .loss import nssa_train_step


def train(
    model: torch.nn.Module,
    data: DataBundle,
    optim: OptimBundle,
    callbacks: Callbacks,
    evaluator: Any,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    """Run ``cfg.epochs`` of training.

    Per-epoch:
      1. ``train_epoch`` (static-buffer compiled step over all batches)
      2. ``save_latest_weights`` if ``cfg.save_dir`` is set
      3. ``ReduceLROnPlateau.step(avg_loss)`` if a plateau scheduler is set
      4. ``callbacks.on_epoch_end(epoch, model, avg_loss)`` — runs validation,
         updates best, saves best checkpoint, returns True to early-stop

    Mutates ``callbacks`` (best_*, validation_history, epoch_durations,
    epochs_completed, stopped_early). ``evaluator`` is unused inside
    ``train`` — it is passed for parity with the DpRL builder API and
    is consulted by ``callbacks.on_epoch_end`` directly.

    Returns a small metrics dict for the caller to merge into
    ``TrainArtifacts.metrics``. The richer per-epoch trace lives on
    ``callbacks`` for the post-training eval to read.
    """
    train_step_override = None
    if cfg.loss == "nssa":
        train_step_override = partial(nssa_train_step, adv_temp=cfg.adv_temp)

    train_start = time.perf_counter()
    model.train()
    last_loss = 0.0

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        losses = train_epoch(
            model, data.sampler, optim.optimizer, data.train_pos,
            batch_size=cfg.batch_size,
            num_negatives=cfg.neg_ratio,
            corrupt_modes=data.train_corrupt_modes,
            grad_clip=cfg.grad_clip,
            scaler=optim.scaler,
            filter_negatives=True, unique_negatives=False,
            compile=getattr(cfg, "compile", False),
            train_step=train_step_override,
        )
        avg_loss = losses["loss"]
        last_loss = avg_loss
        callbacks.epochs_completed = epoch
        epoch_time = time.perf_counter() - epoch_start
        callbacks.epoch_durations.append(epoch_time)
        print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | time={epoch_time:.2f}s", end="\r")

        if cfg.save_dir is not None:
            save_latest_weights(cfg.save_dir, unwrap_model(model).state_dict())

        if optim.plateau_scheduler is not None:
            optim.plateau_scheduler.step(avg_loss)

        if callbacks.on_epoch_end(epoch, model, avg_loss):
            break

    total_train_time = time.perf_counter() - train_start
    durations = callbacks.epoch_durations
    if durations:
        print(f"\nTraining time | epochs={len(durations)} | "
              f"total={total_train_time:.2f}s | "
              f"avg_per_epoch={sum(durations) / len(durations):.2f}s")

    return {"train_loss": last_loss, "epochs_completed": callbacks.epochs_completed}


__all__ = ["train"]
