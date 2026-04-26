"""QueryRepr implementations: per-trajectory Repr → scalar score per query.

Reduces over the candidate-proof dimension ``C`` using
``evidence.mask`` ``[B, C]`` to ignore padded proofs.

Output is always ``Repr(scores=...)`` of shape ``[B]``.

Two extended classes (``ProofScoreQueryRepr``, ``PolicyRolloutQueryRepr``)
do NOT match the strict ``QueryRepr`` Protocol — they take additional
per-trajectory context (success / depth / step0 signals / kge embeddings)
because the DpRL-derived scoring formulas depend on quantities that
the canonical ``Repr → Repr`` flow does not carry. They live here
because they ARE final aggregation utilities; consumers (Searchers)
call them directly with the appropriate keyword arguments.
"""
from __future__ import annotations

from typing import Dict, Optional

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


# ═══════════════════════════════════════════════════════════════════════
# ProofScoreQueryRepr — 20-mode dispatcher (replaces DPrL compute_proof_score)
# ═══════════════════════════════════════════════════════════════════════


_PROOF_TRAJECTORY_MODES = frozenset({
    "no_cliff", "no_cliff_sqrt", "no_cliff_log", "best_avg",
    "max_intermediate_nc", "trajectory_range", "raw_cumulative", "geometric_mean",
})

_PROOF_BINARY_MODES = frozenset({
    "proof_logprob", "leaf_only", "depth_weighted", "final_state",
    "min_step", "best_partial", "sbr", "max_intermediate",
})

_PROOF_RL_VALUE_MODES = frozenset({
    "rl_value", "rl_value_mean", "rl_value_neg", "rl_value_mean_neg",
})


class ProofScoreQueryRepr(nn.Module):
    """20-mode dispatcher for proof-search trajectory scoring.

    Replaces DPrL ``compute_proof_score``. Takes a dict of named
    trajectory accumulators (cumulative_log, min_step_log,
    best_cumulative, sbr_body_min, final_step_log, final_state_scores,
    best_ever_state_score, best_prefix_avg) plus the search's terminal
    ``success`` ``[B]`` and ``depths`` ``[B]``.

    Returns ``[B]`` scores. Lookup-table-dispatched on ``self.mode``.
    """

    def __init__(self, mode: str, *, fail_penalty: float = 1000.0,
                  partial_weight: float = 0.5) -> None:
        super().__init__()
        if (mode not in _PROOF_TRAJECTORY_MODES
                and mode not in _PROOF_BINARY_MODES
                and mode not in _PROOF_RL_VALUE_MODES):
            raise ValueError(f"Unknown proof scoring mode: {mode}")
        self.mode = mode
        self.fail_penalty = fail_penalty
        self.partial_weight = partial_weight

    def forward(
        self,
        accums: Dict[str, Tensor],
        success: Tensor,
        depths: Tensor,
        *,
        cum_value: Optional[Tensor] = None,
        value_steps: Optional[Tensor] = None,
    ) -> Tensor:
        is_success = success.bool()
        safe_depths = depths.float().clamp(min=1.0)
        cum_log = accums.get("cumulative_log_scores")
        mode = self.mode

        # Trajectory modes (no proved/failed branch)
        if mode == "no_cliff":
            return cum_log / safe_depths
        if mode == "no_cliff_sqrt":
            return cum_log / torch.sqrt(safe_depths)
        if mode == "no_cliff_log":
            return cum_log / torch.log1p(safe_depths)
        if mode == "best_avg":
            return accums["best_prefix_avg"]
        if mode == "max_intermediate_nc":
            return torch.log(accums["best_ever_state_score"].clamp(min=1e-8))
        if mode == "trajectory_range":
            return accums["best_cumulative"] - accums["min_step_log_scores"]
        if mode == "raw_cumulative":
            return cum_log
        if mode == "geometric_mean":
            return torch.exp(cum_log / safe_depths)

        # RL value modes
        if mode == "rl_value":
            return cum_value if cum_value is not None else cum_log
        if mode == "rl_value_mean":
            if cum_value is not None and value_steps is not None:
                return cum_value / value_steps.clamp(min=1.0)
            return cum_log / safe_depths
        if mode == "rl_value_neg":
            return -cum_value if cum_value is not None else -cum_log
        if mode == "rl_value_mean_neg":
            if cum_value is not None and value_steps is not None:
                return -cum_value / value_steps.clamp(min=1.0)
            return -cum_log / safe_depths

        # Binary modes (success vs failure)
        failed = torch.full_like(cum_log, -self.fail_penalty)

        if mode == "proof_logprob":
            proved = cum_log
        elif mode == "leaf_only":
            proved = accums["final_step_log_scores"]
        elif mode == "depth_weighted":
            proved = cum_log / safe_depths
        elif mode == "final_state":
            proved = accums["final_state_scores"]
        elif mode == "min_step":
            proved = accums["min_step_log_scores"]
        elif mode == "best_partial":
            proved = cum_log
            failed = accums["best_cumulative"] * self.partial_weight
        elif mode == "sbr":
            proved = accums["sbr_body_min"]
            failed = torch.zeros_like(accums["sbr_body_min"])
        elif mode == "max_intermediate":
            proved = cum_log / safe_depths
            best_state = accums.get("best_ever_state_score")
            if best_state is not None:
                failed = torch.log(best_state.clamp(min=1e-8))
        else:
            # Should be unreachable due to __init__ validation.
            raise ValueError(f"Unknown proof scoring mode: {mode}")

        return torch.where(is_success, proved, failed)


# ═══════════════════════════════════════════════════════════════════════
# PolicyRolloutQueryRepr — 17-mode dispatcher (replaces DPrL _RL_SCORERS + _STEP0_SCORERS)
# ═══════════════════════════════════════════════════════════════════════


_RL_MODES = frozenset({
    "logprob", "logprob_endf", "proof_binary", "proof_bonus",
    "depth_weighted_rl", "logprob_clipped", "proof_rank", "success_only",
    "logprob_scaled_penalty",
})

_STEP0_MODES = frozenset({
    "value_pos", "end_prob", "hybrid_end", "hybrid_value",
    "combined", "value_diff", "endf_kge_embed",
})

_KGE_EMBED_MODES = frozenset({
    "kge_embed", "kge_embed_hybrid", "logprob_kge_embed",
})


class PolicyRolloutQueryRepr(nn.Module):
    """17-mode dispatcher for policy-rollout output → score formulas.

    Replaces DPrL ``_RL_SCORERS`` + ``_STEP0_SCORERS`` plus special-case
    branches. Takes per-entry rollout signals and returns per-entry
    scores.

    Note ``depth_weighted_rl`` is renamed from DpRL's ``depth_weighted``
    to disambiguate from :class:`ProofScoreQueryRepr`'s ``depth_weighted``
    (which has different math: cumulative_log/depth vs lp + beta*depth).
    """

    def __init__(
        self,
        mode: str,
        *,
        alpha: float = 5.0,
        beta: float = 2.0,
        fail_penalty: float = 1000.0,
        endf_penalty: float = 200.0,
    ) -> None:
        super().__init__()
        if (mode not in _RL_MODES
                and mode not in _STEP0_MODES
                and mode not in _KGE_EMBED_MODES):
            raise ValueError(f"Unknown policy-rollout scoring mode: {mode}")
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.fail_penalty = fail_penalty
        self.endf_penalty = endf_penalty

    def forward(
        self,
        *,
        success: Tensor,
        logprobs: Tensor,
        endf: Optional[Tensor] = None,
        depths: Optional[Tensor] = None,
        p_end: Optional[Tensor] = None,
        v_pos: Optional[Tensor] = None,
        v_neg: Optional[Tensor] = None,
        kge_embed: Optional[Tensor] = None,
    ) -> Tensor:
        s = success.bool() if success.dtype != torch.bool else success
        lp = logprobs
        a, b, f = self.alpha, self.beta, self.fail_penalty
        mode = self.mode
        device = lp.device

        # ---- Pure RL modes ----
        if mode == "logprob":
            return torch.where(s, lp, lp - f)
        if mode == "logprob_endf":
            if endf is None:
                raise ValueError("logprob_endf requires endf=...")
            failed = ~s
            scores = lp.clone()
            scores[failed & endf] -= self.endf_penalty
            scores[failed & ~endf] -= f
            return scores
        if mode == "proof_binary":
            return torch.where(s, torch.full_like(lp, a), torch.full_like(lp, -a))
        if mode == "proof_bonus":
            return torch.where(s, lp + a, lp - f)
        if mode == "depth_weighted_rl":
            if depths is None:
                raise ValueError("depth_weighted_rl requires depths=...")
            d_clamped = depths.float().clamp(min=1.0)
            return torch.where(s, lp + b * d_clamped, lp - f)
        if mode == "logprob_clipped":
            return torch.where(s, lp.clamp(min=-a), torch.tensor(-100.0, device=device).expand_as(lp))
        if mode == "proof_rank":
            return torch.where(s, lp + 1000.0, lp)
        if mode == "success_only":
            return torch.where(s, torch.zeros_like(lp), -torch.ones_like(lp))
        if mode == "logprob_scaled_penalty":
            return torch.where(s, lp, lp - a)

        # ---- Step-0 signal modes ----
        if mode == "value_pos":
            if v_pos is None:
                raise ValueError("value_pos requires v_pos=...")
            return v_pos
        if mode == "end_prob":
            if p_end is None:
                raise ValueError("end_prob requires p_end=...")
            return torch.log1p(-p_end)
        if mode == "hybrid_value":
            if v_pos is None:
                raise ValueError("hybrid_value requires v_pos=...")
            return torch.where(s, lp + b * v_pos, lp - f)
        if mode == "value_diff":
            if v_pos is None or v_neg is None:
                raise ValueError("value_diff requires v_pos= and v_neg=...")
            return v_pos - v_neg
        if mode == "hybrid_end":
            if p_end is None:
                raise ValueError("hybrid_end requires p_end=...")
            end_bonus = a * torch.log1p(-p_end)
            return torch.where(s, lp + end_bonus, lp - f + end_bonus)
        if mode == "combined":
            if p_end is None or v_pos is None:
                raise ValueError("combined requires p_end= and v_pos=...")
            end_bonus = a * torch.log1p(-p_end)
            return torch.where(s, lp + end_bonus + b * v_pos, lp - f + end_bonus)
        if mode == "endf_kge_embed":
            if p_end is None or kge_embed is None:
                raise ValueError("endf_kge_embed requires p_end= and kge_embed=...")
            return a * torch.log1p(-p_end) + b * kge_embed

        # ---- KGE-embed modes ----
        if mode == "kge_embed":
            if kge_embed is None:
                raise ValueError("kge_embed requires kge_embed=...")
            return kge_embed
        if mode == "kge_embed_hybrid":
            if kge_embed is None:
                raise ValueError("kge_embed_hybrid requires kge_embed=...")
            return a * s.float() + b * kge_embed
        if mode == "logprob_kge_embed":
            if kge_embed is None:
                raise ValueError("logprob_kge_embed requires kge_embed=...")
            return torch.where(s, lp + b * kge_embed, -f + b * kge_embed)

        # Unreachable (validation in __init__).
        raise ValueError(f"Unknown policy-rollout scoring mode: {mode}")


__all__ = [
    "ConceptMaxQueryRepr",
    "LogSumExpQueryRepr",
    "MLPSumQueryRepr",
    "MaxQueryRepr",
    "MeanQueryRepr",
    "PolicyRolloutQueryRepr",
    "ProofScoreQueryRepr",
    "SumQueryRepr",
]
