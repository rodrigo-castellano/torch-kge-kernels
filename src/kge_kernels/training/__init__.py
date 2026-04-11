"""Shared KGE training primitives: config, dataset, seeding, scheduler,
model wrapping, and a lean ``train_kge`` loop.

Use this subpackage to stand up a pure KGE training pipeline in any
consumer without reimplementing the optimizer/scheduler/compile glue.
Validation, early stopping, checkpointing, and metric logging are left
to the caller via the ``on_epoch_end`` callback in ``train_kge``.
"""
from __future__ import annotations

from .config import KGETrainConfig
from .loop import (
    OnEpochEnd,
    TripleDataset,
    make_cosine_warmup_scheduler,
    set_seed,
    train_kge,
    wrap_model_for_training,
)

__all__ = [
    "KGETrainConfig",
    "OnEpochEnd",
    "TripleDataset",
    "make_cosine_warmup_scheduler",
    "set_seed",
    "train_kge",
    "wrap_model_for_training",
]
