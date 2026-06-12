"""KGE training: full pipeline, config, primitives, and checkpoints.

Public surface:

  - ``pipeline``  — full training pipeline (data → model → train → eval → save)
  - ``train_epoch`` / ``train_step`` / ``nssa_train_step`` — per-epoch + per-step primitives
  - ``iterate_epoch_batches`` / ``pick_query_batch`` — batching helpers
  - ``set_seed`` — reproducibility
  - ``StreamingRankingMetrics`` — training-time observability
  - ``TrainConfig`` (alias ``KGETrainConfig``) — hyperparameter dataclass
  - checkpoint save/load helpers

The ``pipeline`` entry is wired by :mod:`kge_kernels.training.cli` for
``python -m kge_kernels.training.cli``. ``train_epoch`` is the shared
per-epoch primitive used by both tkk's ``pipeline`` and torch-ns's
training path.
"""
from __future__ import annotations

from .builder import (
    Callbacks,
    DataBundle,
    OptimBundle,
    build_callbacks,
    build_data,
    build_evaluator,
    build_model,
    build_optimizer,
    run_evaluation,
)
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
from .epoch import (
    clear_train_cache,
    iterate_epoch_batches,
    pick_query_batch,
    set_seed,
    train_epoch,
)
from .experiment import pipeline
from .loss import nssa_train_step, train_step
from .metrics import StreamingRankingMetrics
from .train import train

__all__ = [
    # Config
    "KGETrainConfig",
    "TrainArtifacts",
    "TrainConfig",
    # Per-epoch + per-step primitives
    "clear_train_cache",
    "iterate_epoch_batches",
    "nssa_train_step",
    "pick_query_batch",
    "set_seed",
    "train_epoch",
    "train_step",
    # Streaming metrics for training-time observability
    "StreamingRankingMetrics",
    # Full pipeline
    "pipeline",
    # Builder + train (DpRL-aligned factory layer)
    "Callbacks",
    "DataBundle",
    "OptimBundle",
    "build_callbacks",
    "build_data",
    "build_evaluator",
    "build_model",
    "build_optimizer",
    "run_evaluation",
    "train",
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
