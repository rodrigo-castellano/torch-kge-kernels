"""Generic KGE training configuration.

``KGETrainConfig`` holds the hyperparameters that matter for any KGE
training loop (optimizer, scheduler, compile, AMP, grad clip, batch /
neg sampling sizes). Dataset path resolution, checkpoint save dirs, and
early-stopping orchestration are intentionally NOT here — those are
consumer-specific and belong in each repo's own config wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KGETrainConfig:
    """Training hyperparameters for a pure KGE model.

    The defaults match the DpRL RotatE-on-FB15k-237 recipe and are
    reasonable for most translation / bilinear / complex KGE models.
    """

    # Optimization
    lr: float = 1e-3
    batch_size: int = 4096
    epochs: int = 5
    weight_decay: float = 1e-6
    grad_clip: float = 2.0

    # Negative sampling
    neg_ratio: int = 1
    adv_temp: float = 0.0  # 0 = uniform averaging, >0 = self-adversarial

    # Scheduler
    scheduler: str = "cosine"  # "cosine" or "none"
    warmup_ratio: float = 0.1

    # Mixed precision / compile
    amp: bool = False
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = True
    compile_warmup_steps: int = 0

    # Parallelism / device
    cpu: bool = False
    multi_gpu: bool = False

    # Reproducibility
    seed: int = 3

    # DataLoader
    num_workers: int = 2


__all__ = ["KGETrainConfig"]
