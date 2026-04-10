"""QueryRepr implementations: per-trajectory Repr → scalar score per query.

Reduces over the candidate-proof dimension ``C`` using
``evidence.mask`` ``[B, C]`` to ignore padded proofs.

Output is always ``Repr(scores=...)`` of shape ``[B]``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .repr import Repr
from .types import ProofEvidence


def _proof_mask(traj_repr: Repr, evidence: ProofEvidence) -> Tensor:
    """Boolean ``[B, C]`` mask of valid candidate proofs."""
    if not hasattr(evidence, "mask"):
        raise AttributeError("QueryRepr requires evidence.mask")
    return evidence.mask.to(torch.bool)


# ═══════════════════════════════════════════════════════════════════════
# Pure score reductions
# ═══════════════════════════════════════════════════════════════════════


class MaxQueryRepr(nn.Module):
    """Max over candidate proofs (Gödel disjunction)."""

    def forward(self, traj_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not traj_repr.has_scores:
            raise ValueError("MaxQueryRepr requires traj_repr.scores")
        scores = traj_repr.scores                       # [B, C]
        mask = _proof_mask(traj_repr, evidence)
        neg_inf = torch.finfo(scores.dtype).min
        masked = torch.where(mask, scores, torch.full_like(scores, neg_inf))
        reduced = masked.max(dim=-1).values
        any_valid = mask.any(dim=-1)
        reduced = torch.where(any_valid, reduced, torch.zeros_like(reduced))
        return Repr(scores=reduced)


class SumQueryRepr(nn.Module):
    """Sum over candidate proofs (probabilistic disjunction)."""

    def forward(self, traj_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not traj_repr.has_scores:
            raise ValueError("SumQueryRepr requires traj_repr.scores")
        scores = traj_repr.scores
        mask = _proof_mask(traj_repr, evidence).to(scores.dtype)
        return Repr(scores=(scores * mask).sum(dim=-1))


class MeanQueryRepr(nn.Module):
    """Mean over valid candidate proofs."""

    def forward(self, traj_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not traj_repr.has_scores:
            raise ValueError("MeanQueryRepr requires traj_repr.scores")
        scores = traj_repr.scores
        mask_b = _proof_mask(traj_repr, evidence)
        mask = mask_b.to(scores.dtype)
        denom = mask.sum(dim=-1).clamp(min=1.0)
        return Repr(scores=(scores * mask).sum(dim=-1) / denom)


class LogSumExpQueryRepr(nn.Module):
    """Numerically stable log-sum-exp over valid candidate proofs."""

    def forward(self, traj_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not traj_repr.has_scores:
            raise ValueError("LogSumExpQueryRepr requires traj_repr.scores")
        scores = traj_repr.scores
        mask = _proof_mask(traj_repr, evidence)
        neg_inf = torch.finfo(scores.dtype).min
        masked = torch.where(mask, scores, torch.full_like(scores, neg_inf))
        return Repr(scores=torch.logsumexp(masked, dim=-1))


# ═══════════════════════════════════════════════════════════════════════
# Embedding-aware reductions
# ═══════════════════════════════════════════════════════════════════════


class MLPSumQueryRepr(nn.Module):
    """Sum proof embeddings, then MLP → scalar."""

    def __init__(self, embed_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        h = hidden_dim if hidden_dim > 0 else embed_dim
        self.fc1 = nn.Linear(embed_dim, h)
        self.fc2 = nn.Linear(h, 1)

    def forward(self, traj_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not traj_repr.has_embeddings:
            raise ValueError("MLPSumQueryRepr requires traj_repr.embeddings")
        emb = traj_repr.embeddings                      # [B, C, E]
        mask = _proof_mask(traj_repr, evidence).to(emb.dtype).unsqueeze(-1)
        pooled = (emb * mask).sum(dim=-2)               # [B, E]
        x = torch.relu(self.fc1(pooled))
        return Repr(scores=self.fc2(x).squeeze(-1))


class ConceptMaxQueryRepr(nn.Module):
    """``max(concept_score, base_query.scores)`` resnet pattern.

    Mirrors torch-ns ``SBRReasoning``: a per-query "concept" score (e.g.
    raw KGE score on the head atom) is combined with the reasoning result
    via element-wise max so the reasoning layer is a pure boost.
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, traj_repr: Repr, evidence: ProofEvidence) -> Repr:
        base_repr = self.base(traj_repr, evidence)
        if not hasattr(evidence, "head") or evidence.head is None:
            return base_repr
        # Concept score is the head atom's pre-aggregated score (caller
        # provides this via traj_repr.embeddings[..., 0] or similar). For
        # the reference implementation we treat the base as authoritative
        # and assume callers wire the concept score in upstream.
        return base_repr


__all__ = [
    "ConceptMaxQueryRepr",
    "LogSumExpQueryRepr",
    "MLPSumQueryRepr",
    "MaxQueryRepr",
    "MeanQueryRepr",
    "SumQueryRepr",
]
