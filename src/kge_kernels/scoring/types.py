"""Shared type definitions for torch-kge-kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor

LongTensor = torch.LongTensor


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for the vectorized corruption sampler."""

    num_entities: int
    num_relations: int
    device: torch.device
    default_mode: Literal["head", "tail", "both", "bernoulli"] = "both"
    seed: int = 0
    order_negatives: bool = False
    min_entity_idx: int = 1


class SupportsCorruptWithMask(Protocol):
    """Protocol for samplers that expose padded corruptions plus validity masks."""

    def corrupt_with_mask(
        self,
        positives: LongTensor,
        *,
        num_negatives: int | None = None,
        mode: Literal["head", "tail", "both", "bernoulli"] | None = None,
        device: torch.device | None = None,
        filter: bool = True,
        unique: bool = True,
    ) -> tuple[LongTensor, torch.BoolTensor]:
        """Generate corruptions plus an explicit validity mask."""


@dataclass(frozen=True)
class CorruptionOutput:
    """Result returned by the public corruption entry point."""

    negatives: LongTensor
    valid_mask: torch.BoolTensor


__all__ = [
    "CorruptionOutput",
    "LongTensor",
    "SamplerConfig",
    "SupportsCorruptWithMask",
    "Tensor",
]
