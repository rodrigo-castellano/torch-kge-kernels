"""KGE training: lean inner loop, full pipeline, config, and checkpoints.

Two levels of abstraction:

  - ``train_kge``   — lean inner loop; caller handles validation, early
    stopping, checkpointing via ``on_epoch_end``.
  - ``train_model`` — full pipeline: data loading, model construction,
    training, validation, early stopping, final evaluation, checkpoint
    saving.

Both share ``TrainConfig`` (all hyperparameters) and the checkpoint /
utility helpers.
"""
from __future__ import annotations

from .checkpoints import (
    build_config_payload,
    config_from_payload,
    load_checkpoint,
    load_checkpoint_payload,
    model_state_dict,
    normalize_loaded_state_dict,
    save_best_checkpoint,
    save_checkpoint,
    save_final_checkpoint,
    save_latest_weights,
    save_state_dict,
    unwrap_model,
    write_json_payload,
)
from .config import KGETrainConfig, TrainArtifacts, TrainConfig
from .loop import (
    OnEpochEnd,
    TripleDataset,
    make_cosine_warmup_scheduler,
    set_seed,
    train_kge,
    wrap_model_for_training,
)
from .pipeline import train_model

__all__ = [
    # Config
    "KGETrainConfig",
    "TrainArtifacts",
    "TrainConfig",
    # Inner loop
    "OnEpochEnd",
    "TripleDataset",
    "make_cosine_warmup_scheduler",
    "set_seed",
    "train_kge",
    "wrap_model_for_training",
    # Full pipeline
    "train_model",
    # Checkpoints
    "build_config_payload",
    "config_from_payload",
    "load_checkpoint",
    "load_checkpoint_payload",
    "model_state_dict",
    "normalize_loaded_state_dict",
    "save_best_checkpoint",
    "save_checkpoint",
    "save_final_checkpoint",
    "save_latest_weights",
    "save_state_dict",
    "unwrap_model",
    "write_json_payload",
]
