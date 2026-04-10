"""TrajRepr implementations: per-state Repr → per-trajectory Repr.

Two interfaces, both implemented by every class:

  forward(state_repr, evidence) -> Repr
      Batch interface used by exhaustive methods (SBR/DCR/R2N).
      Reduces over the depth dimension when ``evidence.D > 0``.

  init(B, device) -> Repr
  step(accum, state_repr, info) -> Repr
      Incremental interface used by sequential methods (DPrL beam/greedy).
      The two interfaces MUST agree:
          init + step(.,d=0) + step(.,d=1) + ... + step(.,d=D-1)
          ≡ forward(state_repr_full, evidence)
      This parity is exercised by ``tests/framework/test_traj_repr.py``.

The element-wise implementations (sum/min/max/cumulative) are agnostic to
the shape of ``state_repr`` in the incremental interface — they work for
``[B]`` (greedy), ``[B*beam]`` (beam), or any other leading shape.
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .repr import Repr
from .types import ProofEvidence, SelectInfo


def _reduce_depth_scores(scores: Tensor, evidence: ProofEvidence, fn) -> Tensor:
    """Reduce a per-state, per-depth score tensor along the depth dim.

    Structured layout: scores ``[B, C, D]`` → reduce → ``[B, C]``
    Legacy layout:     scores ``[B, C]``    → identity (no D dim)

    Padded depths (where ``evidence.body_count[..., d] == 0``) are masked
    out so they don't affect the reduction. ``fn`` decides the reduction
    semantics by accepting ``(masked_scores, depth_valid)``.
    """
    if scores.dim() == 3:
        body_count = evidence.body_count                    # [B, C, D]
        if body_count.dim() != 3:
            raise ValueError(
                "TrajRepr expects body_count [B,C,D] when state_repr has depth dim"
            )
        depth_valid = body_count > 0                        # [B, C, D]
        return fn(scores, depth_valid)
    return scores


# ═══════════════════════════════════════════════════════════════════════
# Element-wise t-norm
# ═══════════════════════════════════════════════════════════════════════


class TNormTrajRepr(nn.Module):
    """T-norm reduction (min or product) over depth steps."""

    def __init__(self, tnorm: Literal["min", "product"] = "min") -> None:
        super().__init__()
        if tnorm not in ("min", "product"):
            raise ValueError(f"Unknown t-norm: {tnorm}")
        self.tnorm = tnorm

    def _identity(self, dtype: torch.dtype) -> float:
        return 1.0  # identity for both min (over [0,1] scores) and product

    def forward(self, state_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not state_repr.has_scores:
            raise ValueError("TNormTrajRepr requires state_repr.scores")

        def reduce(scores: Tensor, depth_valid: Tensor) -> Tensor:
            if self.tnorm == "min":
                big = torch.finfo(scores.dtype).max
                masked = torch.where(depth_valid, scores, torch.full_like(scores, big))
                reduced = masked.min(dim=-1).values
                any_valid = depth_valid.any(dim=-1)
                return torch.where(any_valid, reduced, torch.zeros_like(reduced))
            ones = torch.ones_like(scores)
            masked = torch.where(depth_valid, scores, ones)
            return masked.prod(dim=-1)

        return Repr(scores=_reduce_depth_scores(state_repr.scores, evidence, reduce))

    def init(self, B: int, device) -> Repr:
        return Repr(scores=torch.full((B,), self._identity(torch.float32), device=device))

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr:
        if self.tnorm == "min":
            return Repr(scores=torch.minimum(accum.scores, state_repr.scores))
        return Repr(scores=accum.scores * state_repr.scores)


# ═══════════════════════════════════════════════════════════════════════
# Cumulative log-score
# ═══════════════════════════════════════════════════════════════════════


class CumulativeLogTrajRepr(nn.Module):
    """Sum of ``log(score)`` over depth steps (DPrL ``cumulative_log_scores``)."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, state_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not state_repr.has_scores:
            raise ValueError("CumulativeLogTrajRepr requires state_repr.scores")
        log_scores = torch.log(state_repr.scores.clamp(min=self.eps))

        def reduce(scores: Tensor, depth_valid: Tensor) -> Tensor:
            zeros = torch.zeros_like(scores)
            masked = torch.where(depth_valid, scores, zeros)
            return masked.sum(dim=-1)

        return Repr(scores=_reduce_depth_scores(log_scores, evidence, reduce))

    def init(self, B: int, device) -> Repr:
        return Repr(scores=torch.zeros(B, device=device))

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr:
        log_step = torch.log(state_repr.scores.clamp(min=self.eps))
        return Repr(scores=accum.scores + log_step)


# ═══════════════════════════════════════════════════════════════════════
# Min log-score (worst single step)
# ═══════════════════════════════════════════════════════════════════════


class MinStepTrajRepr(nn.Module):
    """Worst (minimum) log-score across depth steps (DPrL ``min_step``)."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, state_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not state_repr.has_scores:
            raise ValueError("MinStepTrajRepr requires state_repr.scores")
        log_scores = torch.log(state_repr.scores.clamp(min=self.eps))

        def reduce(scores: Tensor, depth_valid: Tensor) -> Tensor:
            big = torch.finfo(scores.dtype).max
            masked = torch.where(depth_valid, scores, torch.full_like(scores, big))
            reduced = masked.min(dim=-1).values
            any_valid = depth_valid.any(dim=-1)
            return torch.where(any_valid, reduced, torch.zeros_like(reduced))

        return Repr(scores=_reduce_depth_scores(log_scores, evidence, reduce))

    def init(self, B: int, device) -> Repr:
        return Repr(scores=torch.full((B,), float("inf"), device=device))

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr:
        log_step = torch.log(state_repr.scores.clamp(min=self.eps))
        return Repr(scores=torch.minimum(accum.scores, log_step))


# ═══════════════════════════════════════════════════════════════════════
# Best cumulative (max of cumulative log-score over prefixes)
# ═══════════════════════════════════════════════════════════════════════


class BestCumulativeTrajRepr(nn.Module):
    """Best (maximum) cumulative log-score over any prefix.

    Replaces DPrL ``best_cumulative``. Has no closed-form ``forward``
    independent of order, so ``forward`` materializes the prefix sums.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, state_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not state_repr.has_scores:
            raise ValueError("BestCumulativeTrajRepr requires state_repr.scores")
        log_scores = torch.log(state_repr.scores.clamp(min=self.eps))
        if log_scores.dim() < 3:
            return Repr(scores=log_scores)
        body_count = evidence.body_count
        depth_valid = body_count > 0
        masked = torch.where(depth_valid, log_scores, torch.zeros_like(log_scores))
        cum = masked.cumsum(dim=-1)
        # Mask out prefixes that include padded depths so they cannot win.
        neg_inf = torch.finfo(cum.dtype).min
        masked_cum = torch.where(depth_valid, cum, torch.full_like(cum, neg_inf))
        best = masked_cum.max(dim=-1).values
        any_valid = depth_valid.any(dim=-1)
        best = torch.where(any_valid, best, torch.zeros_like(best))
        return Repr(scores=best)

    def init(self, B: int, device) -> Repr:
        return Repr(
            scores=torch.full((B,), float("-inf"), device=device),
            embeddings=torch.zeros(B, device=device),  # running cumulative
        )

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr:
        log_step = torch.log(state_repr.scores.clamp(min=self.eps))
        # accum.embeddings holds the running cumulative; accum.scores holds best so far.
        cum = accum.embeddings + log_step
        best = torch.maximum(accum.scores, cum)
        return Repr(scores=best, embeddings=cum)


# ═══════════════════════════════════════════════════════════════════════
# Policy log-prob product (for PPO scoring)
# ═══════════════════════════════════════════════════════════════════════


class PolicyProductTrajRepr(nn.Module):
    """Sum of policy log-probs along the chosen path.

    Reads ``info.log_probs`` rather than ``state_repr.scores``. Used by
    sequential samplers when scoring "the probability we took this path".
    """

    def forward(self, state_repr: Repr, evidence: ProofEvidence) -> Repr:
        # No batch closed form — sequential by construction.
        raise NotImplementedError(
            "PolicyProductTrajRepr is incremental-only; use init/step in a search loop"
        )

    def init(self, B: int, device) -> Repr:
        return Repr(scores=torch.zeros(B, device=device))

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr:
        if info is None or info.log_probs is None:
            raise ValueError("PolicyProductTrajRepr requires SelectInfo.log_probs at each step")
        return Repr(scores=accum.scores + info.log_probs)


# ═══════════════════════════════════════════════════════════════════════
# SBR body min (worst raw atom score across the proof path)
# ═══════════════════════════════════════════════════════════════════════


class SBRBodyMinTrajRepr(nn.Module):
    """Min raw atom score across the entire proof path.

    Replaces DPrL ``sbr_body_min``. Equivalent to TNormTrajRepr("min")
    when ``state_repr`` already encodes the per-step body min.
    """

    def forward(self, state_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not state_repr.has_scores:
            raise ValueError("SBRBodyMinTrajRepr requires state_repr.scores")

        def reduce(scores: Tensor, depth_valid: Tensor) -> Tensor:
            big = torch.finfo(scores.dtype).max
            masked = torch.where(depth_valid, scores, torch.full_like(scores, big))
            reduced = masked.min(dim=-1).values
            any_valid = depth_valid.any(dim=-1)
            return torch.where(any_valid, reduced, torch.ones_like(reduced))

        return Repr(scores=_reduce_depth_scores(state_repr.scores, evidence, reduce))

    def init(self, B: int, device) -> Repr:
        return Repr(scores=torch.ones(B, device=device))

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr:
        return Repr(scores=torch.minimum(accum.scores, state_repr.scores))


__all__ = [
    "BestCumulativeTrajRepr",
    "CumulativeLogTrajRepr",
    "MinStepTrajRepr",
    "PolicyProductTrajRepr",
    "SBRBodyMinTrajRepr",
    "TNormTrajRepr",
]
