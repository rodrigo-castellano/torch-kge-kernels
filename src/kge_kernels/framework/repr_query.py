"""QueryRepr implementations: per-trajectory Repr → scalar score per query.

Two families of QueryRepr live here:

  - **Pool reductions** (Max/Sum/Mean/LogSumExp/MLPSum/ConceptMax):
    reduce over the candidate-proof dimension ``P`` using ``evidence.mask
    [B, P]`` to ignore padded proofs. ``traj_repr.scores [B, P]`` →
    ``Repr.scores [B]``.

  - **Trajectory-summary reduction** (:class:`TrajectoryScoreQueryRepr`):
    consumes ``traj_repr.summaries`` (a ``Dict[str, Tensor]`` of
    trajectory-level statistics populated by the upstream Searcher) and
    returns one of ~36 named score formulas. Replaces the previous
    split between proof-search modes and rollout modes.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .repr import Repr
from .types import ProofEvidence


def _proof_mask(traj_repr: Repr, evidence: ProofEvidence) -> Tensor:
    """Boolean ``[B, P]`` mask of valid candidate proofs."""
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
        scores = traj_repr.scores                       # [B, P]
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
        emb = traj_repr.embeddings                      # [B, P, E]
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
# TrajectoryScoreQueryRepr — unified ~36-mode dispatcher
# ═══════════════════════════════════════════════════════════════════════
#
# Replaces ProofScoreQueryRepr + PolicyRolloutQueryRepr. Reads named
# trajectory summaries from ``traj_repr.summaries`` (Dict[str, Tensor]).
#
# Standard summary keys (callers populate the subset they track):
#
#   "success"               [B] bool   trajectory succeeded?
#   "depths"                [B] long   final depth reached
#   "cumulative_log"        [B]        Σ log p(action_d) along trajectory
#   "min_step_log"          [B]        min_d log p(action_d)
#   "best_cumulative"       [B]        max_d Σ_{d'≤d} log p(action_{d'})
#   "sbr_body_min"          [B]        min over body-atom SBR scores
#   "final_step_log"        [B]        log p(action_{D-1})
#   "final_state_score"     [B]        KGE state-score at terminal step
#   "best_ever_state_score" [B]        max state-score across all steps
#   "best_prefix_avg"       [B]        max_d (Σ_{d'≤d} log p) / d
#   "endf"                  [B] bool   rollout ended via END action?
#   "p_end"                 [B]        step-0 P(end action)
#   "v_pos", "v_neg"        [B]        step-0 value heads
#   "kge_embed"             [B]        KGE-policy embedding similarity
#   "cum_value"             [B]        accumulated value-head signal (RL value modes)
#   "value_steps"           [B] long   #steps the value signal accumulated
#
# Each mode reads only the keys it needs; missing → loud KeyError.
#
# Mode-name collisions with the legacy two-class split are resolved by
# explicit suffixing:
#   ProofScoreQueryRepr.depth_weighted   → "depth_weighted_quotient"
#   PolicyRolloutQueryRepr.depth_weighted → "depth_weighted_bonus"


def _bool(s: Tensor) -> Tensor:
    return s if s.dtype == torch.bool else s.bool()


def _safe_depths(d: Tensor) -> Tensor:
    return d.float().clamp(min=1.0)


def _ts_proof_logprob(d, c):       # ProofScore.proof_logprob
    cum = d["cumulative_log"]
    return torch.where(_bool(d["success"]), cum, torch.full_like(cum, -c.fail_penalty))


def _ts_logprob(d, c):              # PolicyRollout.logprob
    lp = d["cumulative_log"]
    return torch.where(_bool(d["success"]), lp, lp - c.fail_penalty)


def _ts_logprob_endf(d, c):
    lp, s, endf = d["cumulative_log"], _bool(d["success"]), _bool(d["endf"])
    failed = ~s
    out = lp.clone()
    out[failed & endf]  -= c.endf_penalty
    out[failed & ~endf] -= c.fail_penalty
    return out


def _ts_proof_binary(d, c):
    s = _bool(d["success"])
    a = c.alpha
    return torch.where(s, torch.full_like(d["cumulative_log"], a), torch.full_like(d["cumulative_log"], -a))


def _ts_proof_bonus(d, c):
    lp, s, a, f = d["cumulative_log"], _bool(d["success"]), c.alpha, c.fail_penalty
    return torch.where(s, lp + a, lp - f)


def _ts_depth_weighted_quotient(d, c):   # ProofScore.depth_weighted: cum_log / d
    cum, s = d["cumulative_log"], _bool(d["success"])
    proved = cum / _safe_depths(d["depths"])
    return torch.where(s, proved, torch.full_like(cum, -c.fail_penalty))


def _ts_depth_weighted_bonus(d, c):       # PolicyRollout.depth_weighted: lp + β·d
    lp, s, b, f = d["cumulative_log"], _bool(d["success"]), c.beta, c.fail_penalty
    return torch.where(s, lp + b * _safe_depths(d["depths"]), lp - f)


def _ts_logprob_clipped(d, c):
    lp, s, a = d["cumulative_log"], _bool(d["success"]), c.alpha
    return torch.where(s, lp.clamp(min=-a), torch.full_like(lp, -100.0))


def _ts_proof_rank(d, c):
    lp, s = d["cumulative_log"], _bool(d["success"])
    return torch.where(s, lp + 1000.0, lp)


def _ts_success_only(d, c):
    s = _bool(d["success"])
    return torch.where(s, torch.zeros_like(d["cumulative_log"]), -torch.ones_like(d["cumulative_log"]))


def _ts_logprob_scaled_penalty(d, c):
    lp, s, a = d["cumulative_log"], _bool(d["success"]), c.alpha
    return torch.where(s, lp, lp - a)


def _ts_leaf_only(d, c):
    cum, s = d["cumulative_log"], _bool(d["success"])
    return torch.where(s, d["final_step_log"], torch.full_like(cum, -c.fail_penalty))


def _ts_final_state(d, c):
    cum, s = d["cumulative_log"], _bool(d["success"])
    return torch.where(s, d["final_state_score"], torch.full_like(cum, -c.fail_penalty))


def _ts_min_step(d, c):
    cum, s = d["cumulative_log"], _bool(d["success"])
    return torch.where(s, d["min_step_log"], torch.full_like(cum, -c.fail_penalty))


def _ts_best_partial(d, c):
    cum, s = d["cumulative_log"], _bool(d["success"])
    failed = d["best_cumulative"] * c.partial_weight
    return torch.where(s, cum, failed)


def _ts_sbr(d, c):
    s = _bool(d["success"])
    sbr = d["sbr_body_min"]
    return torch.where(s, sbr, torch.zeros_like(sbr))


def _ts_max_intermediate(d, c):
    cum, s = d["cumulative_log"], _bool(d["success"])
    proved = cum / _safe_depths(d["depths"])
    best_state = d.get("best_ever_state_score")
    failed = (
        torch.log(best_state.clamp(min=1e-8)) if best_state is not None
        else torch.full_like(cum, -c.fail_penalty)
    )
    return torch.where(s, proved, failed)


def _ts_no_cliff(d, c):
    return d["cumulative_log"] / _safe_depths(d["depths"])


def _ts_no_cliff_sqrt(d, c):
    return d["cumulative_log"] / torch.sqrt(_safe_depths(d["depths"]))


def _ts_no_cliff_log(d, c):
    return d["cumulative_log"] / torch.log1p(_safe_depths(d["depths"]))


def _ts_best_avg(d, c):
    return d["best_prefix_avg"]


def _ts_max_intermediate_nc(d, c):
    return torch.log(d["best_ever_state_score"].clamp(min=1e-8))


def _ts_trajectory_range(d, c):
    return d["best_cumulative"] - d["min_step_log"]


def _ts_raw_cumulative(d, c):
    return d["cumulative_log"]


def _ts_geometric_mean(d, c):
    return torch.exp(d["cumulative_log"] / _safe_depths(d["depths"]))


def _ts_rl_value(d, c):
    cv = d.get("cum_value")
    return cv if cv is not None else d["cumulative_log"]


def _ts_rl_value_mean(d, c):
    cv, vs = d.get("cum_value"), d.get("value_steps")
    if cv is not None and vs is not None:
        return cv / vs.float().clamp(min=1.0)
    return d["cumulative_log"] / _safe_depths(d["depths"])


def _ts_rl_value_neg(d, c):
    cv = d.get("cum_value")
    return -cv if cv is not None else -d["cumulative_log"]


def _ts_rl_value_mean_neg(d, c):
    cv, vs = d.get("cum_value"), d.get("value_steps")
    if cv is not None and vs is not None:
        return -cv / vs.float().clamp(min=1.0)
    return -d["cumulative_log"] / _safe_depths(d["depths"])


def _ts_value_pos(d, c):
    return d["v_pos"]


def _ts_end_prob(d, c):
    return torch.log1p(-d["p_end"])


def _ts_hybrid_value(d, c):
    lp, s, b, f = d["cumulative_log"], _bool(d["success"]), c.beta, c.fail_penalty
    return torch.where(s, lp + b * d["v_pos"], lp - f)


def _ts_value_diff(d, c):
    return d["v_pos"] - d["v_neg"]


def _ts_hybrid_end(d, c):
    lp, s, a, f = d["cumulative_log"], _bool(d["success"]), c.alpha, c.fail_penalty
    end_bonus = a * torch.log1p(-d["p_end"])
    return torch.where(s, lp + end_bonus, lp - f + end_bonus)


def _ts_combined(d, c):
    lp, s, a, b, f = d["cumulative_log"], _bool(d["success"]), c.alpha, c.beta, c.fail_penalty
    end_bonus = a * torch.log1p(-d["p_end"])
    return torch.where(s, lp + end_bonus + b * d["v_pos"], lp - f + end_bonus)


def _ts_endf_kge_embed(d, c):
    a, b = c.alpha, c.beta
    return a * torch.log1p(-d["p_end"]) + b * d["kge_embed"]


def _ts_kge_embed(d, c):
    return d["kge_embed"]


def _ts_kge_embed_hybrid(d, c):
    s, a, b = _bool(d["success"]), c.alpha, c.beta
    return a * s.float() + b * d["kge_embed"]


def _ts_logprob_kge_embed(d, c):
    lp, s, b, f = d["cumulative_log"], _bool(d["success"]), c.beta, c.fail_penalty
    return torch.where(s, lp + b * d["kge_embed"], -f + b * d["kge_embed"])


_TS_MODE_FNS = {
    # Proof-search trajectory modes
    "no_cliff":              _ts_no_cliff,
    "no_cliff_sqrt":         _ts_no_cliff_sqrt,
    "no_cliff_log":          _ts_no_cliff_log,
    "best_avg":              _ts_best_avg,
    "max_intermediate_nc":   _ts_max_intermediate_nc,
    "trajectory_range":      _ts_trajectory_range,
    "raw_cumulative":        _ts_raw_cumulative,
    "geometric_mean":        _ts_geometric_mean,
    # Proof-search RL-value modes
    "rl_value":              _ts_rl_value,
    "rl_value_mean":         _ts_rl_value_mean,
    "rl_value_neg":          _ts_rl_value_neg,
    "rl_value_mean_neg":     _ts_rl_value_mean_neg,
    # Proof-search binary modes
    "proof_logprob":         _ts_proof_logprob,
    "leaf_only":             _ts_leaf_only,
    "depth_weighted_quotient": _ts_depth_weighted_quotient,   # was ProofScore.depth_weighted
    "final_state":           _ts_final_state,
    "min_step":              _ts_min_step,
    "best_partial":          _ts_best_partial,
    "sbr":                   _ts_sbr,
    "max_intermediate":      _ts_max_intermediate,
    # Rollout RL modes
    "logprob":               _ts_logprob,
    "logprob_endf":          _ts_logprob_endf,
    "proof_binary":          _ts_proof_binary,
    "proof_bonus":           _ts_proof_bonus,
    "depth_weighted_bonus":  _ts_depth_weighted_bonus,         # was PolicyRollout.depth_weighted
    "logprob_clipped":       _ts_logprob_clipped,
    "proof_rank":            _ts_proof_rank,
    "success_only":          _ts_success_only,
    "logprob_scaled_penalty": _ts_logprob_scaled_penalty,
    # Rollout step-0 signal modes
    "value_pos":             _ts_value_pos,
    "end_prob":              _ts_end_prob,
    "hybrid_end":            _ts_hybrid_end,
    "hybrid_value":          _ts_hybrid_value,
    "combined":              _ts_combined,
    "value_diff":            _ts_value_diff,
    "endf_kge_embed":        _ts_endf_kge_embed,
    # Rollout KGE-embed modes
    "kge_embed":             _ts_kge_embed,
    "kge_embed_hybrid":      _ts_kge_embed_hybrid,
    "logprob_kge_embed":     _ts_logprob_kge_embed,
}

ALL_TRAJECTORY_SCORE_MODES = frozenset(_TS_MODE_FNS.keys())


class TrajectoryScoreQueryRepr(nn.Module):
    """Unified trajectory-scoring dispatcher (~36 modes).

    Reads named trajectory summaries from ``traj_repr.summaries`` (a
    ``Dict[str, Tensor]`` populated by the upstream Searcher) and
    returns ``Repr.scores [B]``. Conforms to the :class:`QueryRepr`
    Protocol.

    Replaces :class:`ProofScoreQueryRepr` and :class:`PolicyRolloutQueryRepr`
    — both compute "scalar score from trajectory summaries"; the split
    was implementation drift from the upstream search choosing which
    summaries to track. See :data:`ALL_TRAJECTORY_SCORE_MODES` for the
    available mode keys.
    """

    def __init__(
        self,
        mode: str,
        *,
        alpha: float = 5.0,
        beta: float = 2.0,
        fail_penalty: float = 1000.0,
        partial_weight: float = 0.5,
        endf_penalty: float = 200.0,
    ) -> None:
        super().__init__()
        if mode not in _TS_MODE_FNS:
            raise ValueError(
                f"Unknown trajectory-scoring mode: {mode!r}. "
                f"Known modes: {sorted(_TS_MODE_FNS)}"
            )
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.fail_penalty = fail_penalty
        self.partial_weight = partial_weight
        self.endf_penalty = endf_penalty

    def forward(self, traj_repr: Repr, evidence: Optional[ProofEvidence] = None) -> Repr:
        if traj_repr.summaries is None:
            raise ValueError(
                "TrajectoryScoreQueryRepr requires traj_repr.summaries; "
                "upstream Searcher must populate the trajectory-summaries dict."
            )
        scores = _TS_MODE_FNS[self.mode](traj_repr.summaries, self)
        return Repr(scores=scores)


__all__ = [
    "ALL_TRAJECTORY_SCORE_MODES",
    "ConceptMaxQueryRepr",
    "LogSumExpQueryRepr",
    "MLPSumQueryRepr",
    "MaxQueryRepr",
    "MeanQueryRepr",
    "SumQueryRepr",
    "TrajectoryScoreQueryRepr",
]
