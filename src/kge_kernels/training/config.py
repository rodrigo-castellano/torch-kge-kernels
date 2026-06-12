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

from kge_kernels.runs import LoggingConfig


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
    evaluation schedule. All fields have defaults so :func:`pipeline`
    can be called with just ``TrainConfig(dataset=..., epochs=N)``.
    """

    # -- Data paths --
    # Disk persistence is opt-in: by default ``pipeline`` writes nothing.
    # Real runs go through :func:`kge_kernels.runs.run_cli` (or its
    # per-run primitive :func:`kge_kernels.runs.run_one`), which
    # supplies the contract path (``output/runs/<experiment>/<run>/``) via
    # :class:`RunContext`. Setting ``save_dir`` directly is for ad-hoc CLI
    # use only and bypasses the shared logging contract — prefer
    # ``run_cli``. The legacy default ``"checkpoints"`` was removed
    # because it polluted the repo root on every standalone pipeline
    # call (notably from scripts/run_3way_sweep.py).
    save_dir: str | None = None
    run_signature: str | None = None
    train_path: str | None = None
    dataset: str | None = None
    data_root: str = field(default_factory=_default_data_root)
    train_split: str = "train.txt"
    valid_path: str | None = None
    test_path: str | None = None
    valid_split: str = "valid.txt"
    test_split: str = "test.txt"
    domain_file: str | None = None  # e.g. "domain2constants.txt" for typed corruption

    # -- Model --
    model: str = "RotatE"
    dim: int = 1024
    gamma: float = 12.0
    p: int = 2
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

    # -- Loss --
    loss: str = "nssa"  # "nssa" or "bce"

    # -- Negative sampling --
    neg_ratio: int = 1
    adv_temp: float = 0.0  # 0 = uniform averaging, >0 = self-adversarial
    use_reciprocal: bool = False
    precompute_negatives: bool = True  # pre-compute once (fast convergence, needs validation)

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

    # tkk run-bundle policy (output_root / model save mode / etc.).
    # Default-built; populated by :meth:`logging_config`.
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ─────────────────────────────────────────────────────────────────
    # tkk run-bundle metadata hooks (duck-typed by run_cli)
    # ─────────────────────────────────────────────────────────────────
    def family(self) -> str:
        """Run family — the parent directory under ``output/runs/``."""
        if self.logging.family:
            return self.logging.family
        return self.dataset or "kge"

    def signature(self) -> str:
        """Stable signature for this run (used as the run-dir name)."""
        if self.run_signature:
            return self.run_signature
        return f"{self.dataset or 'kge'}_{self.model}_seed{self.seed}"

    def logging_config(self) -> LoggingConfig:
        """Run-bundle policy: where to write + which model save mode."""
        return self.logging


# Backward-compat alias so existing ``from kge_kernels.training import
# KGETrainConfig`` still works.
KGETrainConfig = TrainConfig


@dataclass
class TrainArtifacts:
    """Output of a completed ``pipeline`` run.

    ``config_path`` and ``weights_path`` are only set when ``save_dir`` was
    provided; with the default ``save_dir=None`` they're empty strings and
    no disk writes happen.
    """

    entity2id: dict
    relation2id: dict
    config_path: str
    weights_path: str
    metrics: Optional[Dict[str, float]] = None


__all__ = ["KGETrainConfig", "TrainArtifacts", "TrainConfig"]
