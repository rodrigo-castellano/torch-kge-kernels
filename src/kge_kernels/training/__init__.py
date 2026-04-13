"""Shared KGE training primitives: config, dataset, seeding, scheduler,
model wrapping, checkpoints, and a lean ``train_kge`` loop.

Use this subpackage to stand up a pure KGE training pipeline in any
consumer without reimplementing the optimizer/scheduler/compile glue.
Validation, early stopping, checkpointing, and metric logging are left
to the caller via the ``on_epoch_end`` callback in ``train_kge``.
"""
from __future__ import annotations

from .checkpoints import (
    load_checkpoint,
    model_state_dict,
    normalize_loaded_state_dict,
    save_checkpoint,
    save_state_dict,
    unwrap_model,
    write_json_payload,
)
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
    "load_checkpoint",
    "make_cosine_warmup_scheduler",
    "model_state_dict",
    "normalize_loaded_state_dict",
    "save_checkpoint",
    "save_state_dict",
    "set_seed",
    "train_kge",
    "unwrap_model",
    "wrap_model_for_training",
    "write_json_payload",
]
