"""Streaming MRR + Hits@k accumulator for training-time observability.

Per-batch labeled-prediction streams (``y_pred``, ``y_true``) → running
MRR / Hits@k without any GPU→CPU sync until ``streaming_compute`` is
called. Distinct from :class:`kge_kernels.eval.RankingEvaluator`, which
operates on a held-out triple set + corruption sampler.

Functional API: a small mutable :class:`StreamingRanking` state plus
three free functions (:func:`streaming_reset`, :func:`streaming_update`,
:func:`streaming_compute`). The legacy class name
:class:`StreamingRankingMetrics` is kept as a thin compat wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from ..eval.ranking import compute_ranks


@dataclass
class StreamingRanking:
    """Mutable state for streaming MRR + Hits@k.

    GPU accumulators are allocated lazily on the first
    :func:`streaming_update` call (so the state is device-agnostic at
    construction time and can be reused across train/val splits).

    Args:
        ks: Hits@k cutoffs.
        pad_value: ``y_true`` value marking ignored cells.
        positive_value: ``y_true`` value marking the positive cell.
        metric_key: Output key for MRR. ``"MRR"`` matches
            :func:`kge_kernels.eval.metrics_from_ranks`; ``"mrrmetric"``
            uses the lowercase ``"hits@k"`` convention some consumers expect.
    """
    ks: Tuple[int, ...] = (1, 3, 10)
    pad_value: int = -1
    positive_value: int = 1
    metric_key: str = "MRR"
    # GPU accumulators (None until first update)
    _mrr_sum: Optional[Tensor] = field(default=None, repr=False)
    _count: Optional[Tensor] = field(default=None, repr=False)
    _hits_sums: Dict[int, Tensor] = field(default_factory=dict, repr=False)


def streaming_reset(state: StreamingRanking) -> None:
    """Zero all accumulators in-place. Reuses allocated tensors if any."""
    if state._mrr_sum is not None and state._count is not None:
        state._mrr_sum.zero_()
        state._count.zero_()
        for k in state.ks:
            state._hits_sums[k].zero_()
    else:
        state._mrr_sum = None
        state._count = None
        state._hits_sums = {}


def streaming_update(state: StreamingRanking, y_pred: Tensor, y_true: Tensor) -> None:
    """Accumulate one batch without GPU→CPU sync.

    ``y_true`` is a per-cell label tensor: ``positive_value`` marks
    positives, ``pad_value`` marks ignored cells, anything else is a
    distractor. We pick the best-scored positive per row and rank it
    among valid cells.

    Float32 accumulators match torch-ns's pre-consolidation
    ``FusedRankingMetrics`` byte-for-byte (float64 is overkill for MRR
    summations and adds a measurable per-batch promotion cost).
    """
    if state._mrr_sum is None:
        device = y_pred.device
        state._mrr_sum = torch.zeros((), device=device, dtype=torch.float32)
        state._count = torch.zeros((), device=device, dtype=torch.long)
        for k in state.ks:
            state._hits_sums[k] = torch.zeros((), device=device, dtype=torch.long)

    valid_cells = y_true != state.pad_value
    is_positive = y_true == state.positive_value
    valid = valid_cells.any(dim=1) & is_positive.any(dim=1)
    best_pos_idx = y_pred.masked_fill(~is_positive, float("-inf")).argmax(dim=1)
    ranks = compute_ranks(y_pred, best_pos_idx, valid_mask=valid_cells)

    inv_ranks = (1.0 / ranks).masked_fill_(~valid, 0.0)
    state._mrr_sum.add_(inv_ranks.sum())
    state._count.add_(valid.sum())
    for k in state.ks:
        state._hits_sums[k].add_((ranks.le(k) & valid).sum())


def streaming_compute(state: StreamingRanking) -> Dict[str, float]:
    """Return ``{metric_key: mrr, "Hits@k": ...}`` as Python floats.

    Final division uses float32 (matching torch-ns legacy). The hits
    prefix follows ``state.metric_key``: ``"MRR"`` → ``"Hits@k"``,
    ``"mrrmetric"`` → ``"hits@k"``.
    """
    hits_prefix = "hits" if state.metric_key == "mrrmetric" else "Hits"
    if state._count is None or state._mrr_sum is None:
        out: Dict[str, float] = {state.metric_key: 0.0}
        for k in state.ks:
            out[f"{hits_prefix}@{k}"] = 0.0
        return out
    count = state._count.clamp(min=1).float()
    results: Dict[str, float] = {state.metric_key: (state._mrr_sum / count).item()}
    for k in state.ks:
        results[f"{hits_prefix}@{k}"] = (state._hits_sums[k].float() / count).item()
    return results


class StreamingRankingMetrics:
    """Compat shim around :class:`StreamingRanking`.

    Existing callers (``ns/experiment.py:617-623``, tests) use the
    class-style API. The class delegates to the functional API so all
    semantics are preserved bit-for-bit.
    """

    def __init__(
        self,
        ks: Tuple[int, ...] = (1, 3, 10),
        pad_value: int = -1,
        positive_value: int = 1,
        metric_key: str = "MRR",
    ) -> None:
        self._state = StreamingRanking(
            ks=tuple(ks), pad_value=pad_value,
            positive_value=positive_value, metric_key=metric_key,
        )

    @property
    def ks(self) -> Tuple[int, ...]:
        return self._state.ks

    @property
    def metric_key(self) -> str:
        return self._state.metric_key

    def reset(self) -> None:
        streaming_reset(self._state)

    def update(self, y_pred: Tensor, y_true: Tensor) -> None:
        streaming_update(self._state, y_pred, y_true)

    def compute(self) -> Dict[str, float]:
        return streaming_compute(self._state)


__all__ = [
    "StreamingRanking",
    "StreamingRankingMetrics",
    "streaming_compute",
    "streaming_reset",
    "streaming_update",
]
