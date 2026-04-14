"""KGE training configuration.

``TrainConfig`` holds every hyperparameter needed for a complete
KGE training pipeline: data paths, model specification, optimiser
settings, evaluation schedule, early stopping, and checkpointing.

``KGETrainConfig`` is a backwards-compatibility alias for ``TrainConfig``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


def _default_data_root() -> str:
    """Shared data repo, overridable via ``DATA_ROOT`` env var."""
    return os.environ.get(
        "DATA_ROOT",
        os.path.expanduser("~/repos/data-swarm/main"),
    )


@dataclass
class TrainConfig:
    """Full KGE training configuration.

    Covers everything from data paths to optimiser settings to
    evaluation schedule. All fields have defaults so the lean
    ``train_kge`` inner loop can still be called with just
    ``TrainConfig(epochs=N)``.
    """

    # -- Data paths --
    save_dir: str = "checkpoints"
    run_signature: str | None = None
    train_path: str | None = None
    dataset: str | None = None
    data_root: str = field(default_factory=_default_data_root)
    train_split: str = "train.txt"
    valid_path: str | None = None
    test_path: str | None = None
    valid_split: str = "valid.txt"
    test_split: str = "test.txt"

    # -- Model --
    model: str = "RotatE"
    dim: int = 1024
    gamma: float = 12.0
    p: int = 1
    relation_dim: int | None = None
    dropout: float = 0.0
    input_dropout: float = 0.2
    feature_map_dropout: float = 0.2
    hidden_dropout: float = 0.3
    embedding_height: int = 10
    embedding_width: int = 20

    # -- Optimisation --
    lr: float = 1e-3
    batch_size: int = 4096
    epochs: int = 5
    weight_decay: float = 1e-6
    grad_clip: float = 2.0

    # -- Negative sampling --
    neg_ratio: int = 1
    adv_temp: float = 0.0  # 0 = uniform averaging, >0 = self-adversarial
    use_reciprocal: bool = False

    # -- Scheduler --
    scheduler: str = "cosine"  # "cosine" or "none"
    warmup_ratio: float = 0.1

    # -- Mixed precision / compile --
    amp: bool = False
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = True
    compile_warmup_steps: int = 0

    # -- Parallelism / device --
    cpu: bool = False
    multi_gpu: bool = False

    # -- Reproducibility --
    seed: int = 3

    # -- DataLoader --
    num_workers: int = 2

    # -- Early stopping --
    use_early_stopping: bool = False
    patience: int = 10

    # -- Evaluation --
    eval_chunk_size: int = 2048
    eval_limit: int = 0
    valid_eval_every: int = 0
    valid_eval_queries: int = 0
    report_train_mrr: bool = True
    eval_num_corruptions: int = 100
    corruption_scheme: str = "both"  # "head", "tail", or "both"


# Backward-compat alias so existing ``from kge_kernels.training import
# KGETrainConfig`` still works.
KGETrainConfig = TrainConfig


@dataclass
class TrainArtifacts:
    """Output of a completed ``train_model`` run."""

    entity2id: dict
    relation2id: dict
    config_path: str
    weights_path: str
    metrics: Optional[Dict[str, float]] = None


__all__ = ["KGETrainConfig", "TrainArtifacts", "TrainConfig"]
