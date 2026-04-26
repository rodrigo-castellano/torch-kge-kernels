"""Shared type definitions for torch-kge-kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

LongTensor = torch.LongTensor


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for the vectorized corruption sampler."""

    num_entities: int
    num_relations: int
    device: torch.device
    default_mode: Literal["head", "tail", "both"] = "both"
    seed: int = 0
    order_negatives: bool = False
    min_entity_idx: int = 1


__all__ = [
    "LongTensor",
    "SamplerConfig",
    "Tensor",
]
