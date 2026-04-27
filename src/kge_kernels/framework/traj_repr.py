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

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Sum of policy log-probs along the chosen path — incremental only.

    Reads ``info.log_probs`` from the ``SelectInfo`` returned by the
    ``Select`` primitive, not ``state_repr.scores``. The accumulated
    value at the end of a sequential search is the log-likelihood the
    policy assigns to the whole rollout — i.e. the quantity used for
    REINFORCE / PPO policy-gradient scoring.

    **Incremental-only.** This primitive does *not* implement the batch
    ``forward(state_repr, evidence)`` interface because the quantity
    depends on which action was chosen at each step, and the batched
    ``evidence`` does not carry per-step policy log-probs. Calling
    :meth:`forward` raises :class:`TypeError` with a pointer to the
    incremental API. Sequential search loops (including the reference
    :func:`kge_kernels.framework.search_and_score`) always go through
    :meth:`init` + :meth:`step`, so this does not limit the reference
    implementation.

    Contract::

        accum = traj_repr.init(B, device)
        for d in range(max_depth):
            state, info = select(evidence, s_repr)
            # info.log_probs has shape [B] and carries log π(a_t | s_t)
            accum = traj_repr.step(accum, s_repr, info)
    """

    def forward(self, state_repr: Repr, evidence: ProofEvidence) -> Repr:
        # Raise TypeError rather than NotImplementedError: this is an
        # interface contract violation, not a missing implementation.
        # The primitive is correctly incremental-only by design.
        raise TypeError(
            "PolicyProductTrajRepr has no batch forward(); it is incremental-only. "
            "Use traj_repr.init(B, device) and traj_repr.step(accum, s_repr, info) "
            "inside a sequential search loop."
        )

    def init(self, B: int, device) -> Repr:
        return Repr(scores=torch.zeros(B, device=device))

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr:
        if info is None or info.log_probs is None:
            raise ValueError(
                "PolicyProductTrajRepr requires SelectInfo.log_probs at each step"
            )
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


# ═══════════════════════════════════════════════════════════════════════
# Final-step log score (overwritten at each active step)
# ═══════════════════════════════════════════════════════════════════════


class FinalStepLogScoreTrajRepr(nn.Module):
    """Log-score at the last active step (DPrL ``final_step_log_scores``).

    Each ``step`` call overwrites the accumulator with the current
    log-score. After D steps, the value is whatever was last written —
    semantically the final step's contribution.

    The search loop is responsible for not stepping after a query is
    done (otherwise the value would be overwritten by the post-done
    score). This matches the existing tkk TrajRepr convention; the
    ``active`` masking logic lives in the search loop, not here.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, state_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not state_repr.has_scores:
            raise ValueError("FinalStepLogScoreTrajRepr requires state_repr.scores")
        log_scores = torch.log(state_repr.scores.clamp(min=self.eps))

        def reduce(scores: Tensor, depth_valid: Tensor) -> Tensor:
            # Pick the score at the last valid depth.
            last_valid_idx = depth_valid.long().sum(dim=-1) - 1   # [B, C]
            last_valid_idx = last_valid_idx.clamp(min=0)
            gathered = scores.gather(-1, last_valid_idx.unsqueeze(-1)).squeeze(-1)
            any_valid = depth_valid.any(dim=-1)
            return torch.where(any_valid, gathered, torch.zeros_like(gathered))

        return Repr(scores=_reduce_depth_scores(log_scores, evidence, reduce))

    def init(self, B: int, device) -> Repr:
        return Repr(scores=torch.zeros(B, device=device))

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr:
        log_step = torch.log(state_repr.scores.clamp(min=self.eps))
        return Repr(scores=log_step)


# ═══════════════════════════════════════════════════════════════════════
# Final state score (max raw state score at the last active step)
# ═══════════════════════════════════════════════════════════════════════


class FinalStateScoresTrajRepr(nn.Module):
    """Max raw state score at the last active step (DPrL ``final_state_scores``).

    Reads ``info.chosen_scores`` to get the per-step "best state" value
    (e.g., max over derived states); writes its max into the
    accumulator on every active step. The final value is from the
    last active step.

    Falls back to ``state_repr.scores`` if ``info`` is None or carries
    no chosen_scores — this lets the same primitive serve both the
    "max derived state at this step" reading (when select returned the
    full per-state vector) and the simpler reading (when state_repr
    already encodes the per-step max).
    """

    def forward(self, state_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not state_repr.has_scores:
            raise ValueError("FinalStateScoresTrajRepr requires state_repr.scores")

        def reduce(scores: Tensor, depth_valid: Tensor) -> Tensor:
            last_valid_idx = depth_valid.long().sum(dim=-1) - 1
            last_valid_idx = last_valid_idx.clamp(min=0)
            gathered = scores.gather(-1, last_valid_idx.unsqueeze(-1)).squeeze(-1)
            any_valid = depth_valid.any(dim=-1)
            return torch.where(any_valid, gathered, torch.zeros_like(gathered))

        return Repr(scores=_reduce_depth_scores(state_repr.scores, evidence, reduce))

    def init(self, B: int, device) -> Repr:
        return Repr(scores=torch.zeros(B, device=device))

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr:
        # Prefer info.chosen_scores when available (search-step max), else state_repr.scores.
        if info is not None and info.chosen_scores is not None:
            val = info.chosen_scores.amax(dim=-1) if info.chosen_scores.dim() > 1 else info.chosen_scores
        else:
            val = state_repr.scores
        return Repr(scores=val)


# ═══════════════════════════════════════════════════════════════════════
# Best ever state score (running max across all steps)
# ═══════════════════════════════════════════════════════════════════════


class BestEverStateScoreTrajRepr(nn.Module):
    """Running max of raw state scores across all steps (DPrL ``best_ever_state_score``).

    Same input semantics as :class:`FinalStateScoresTrajRepr` but
    accumulates a running max instead of overwriting.
    """

    def forward(self, state_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not state_repr.has_scores:
            raise ValueError("BestEverStateScoreTrajRepr requires state_repr.scores")

        def reduce(scores: Tensor, depth_valid: Tensor) -> Tensor:
            neg_inf = torch.finfo(scores.dtype).min
            masked = torch.where(depth_valid, scores, torch.full_like(scores, neg_inf))
            best = masked.max(dim=-1).values
            any_valid = depth_valid.any(dim=-1)
            return torch.where(any_valid, best, torch.zeros_like(best))

        return Repr(scores=_reduce_depth_scores(state_repr.scores, evidence, reduce))

    def init(self, B: int, device) -> Repr:
        return Repr(scores=torch.zeros(B, device=device))

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr:
        if info is not None and info.chosen_scores is not None:
            val = info.chosen_scores.amax(dim=-1) if info.chosen_scores.dim() > 1 else info.chosen_scores
        else:
            val = state_repr.scores
        return Repr(scores=torch.maximum(accum.scores, val))


# ═══════════════════════════════════════════════════════════════════════
# Best prefix average (max over prefixes of cumulative_log / step_count)
# ═══════════════════════════════════════════════════════════════════════


class BestPrefixAvgTrajRepr(nn.Module):
    """Best (max) of ``cumulative_log_score / step_count`` over any prefix.

    Replaces DPrL ``best_prefix_avg``. Like :class:`BestCumulativeTrajRepr`,
    it tracks both a running cumulative and the running best — encoded as
    ``embeddings`` and ``scores`` respectively.

    Step counter is implicit: each ``step`` call advances by one. Initial
    state has cumulative=0, best=-inf, and an embedded step counter
    starts at 0.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, state_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not state_repr.has_scores:
            raise ValueError("BestPrefixAvgTrajRepr requires state_repr.scores")
        log_scores = torch.log(state_repr.scores.clamp(min=self.eps))
        if log_scores.dim() < 3:
            return Repr(scores=log_scores)
        body_count = evidence.body_count
        depth_valid = body_count > 0
        masked = torch.where(depth_valid, log_scores, torch.zeros_like(log_scores))
        cum = masked.cumsum(dim=-1)
        # Step indices [1, 2, ..., D], broadcast against the cumulative.
        D = log_scores.shape[-1]
        step_idx = torch.arange(1, D + 1, device=log_scores.device, dtype=cum.dtype)
        prefix_avg = cum / step_idx                                # [..., D]
        neg_inf = torch.finfo(prefix_avg.dtype).min
        masked_avg = torch.where(depth_valid, prefix_avg, torch.full_like(prefix_avg, neg_inf))
        best = masked_avg.max(dim=-1).values
        any_valid = depth_valid.any(dim=-1)
        return Repr(scores=torch.where(any_valid, best, torch.zeros_like(best)))

    def init(self, B: int, device) -> Repr:
        # scores: best so far. embeddings: running cumulative + step counter packed as [B, 2].
        cum_and_step = torch.zeros(B, 2, device=device)
        return Repr(
            scores=torch.full((B,), float("-inf"), device=device),
            embeddings=cum_and_step,
        )

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr:
        log_step = torch.log(state_repr.scores.clamp(min=self.eps))
        cum = accum.embeddings[:, 0] + log_step
        step_count = accum.embeddings[:, 1] + 1.0
        prefix_avg = cum / step_count
        best = torch.maximum(accum.scores, prefix_avg)
        new_emb = torch.stack([cum, step_count], dim=-1)
        return Repr(scores=best, embeddings=new_emb)


# ═══════════════════════════════════════════════════════════════════════
# Multi-traj composite (parallel update of many TrajRepr instances)
# ═══════════════════════════════════════════════════════════════════════


class MultiTrajRepr(nn.Module):
    """Holds a dict of named TrajRepr instances and updates them in lockstep.

    Replaces DPrL's monolithic ``SearchAccumulators`` dataclass. Used by
    Searcher classes that need multiple statistics (cumulative log, min
    step, best cumulative, etc.) computed in one pass.

    The ``Repr`` carrier this primitive returns is unconventional:
    ``embeddings`` is None, ``scores`` is None, but the named per-stat
    Repr instances live in a public ``self._accums: Dict[str, Repr]``.
    Consumers that need the canonical ``Repr`` interface should pick
    one named accumulator via :meth:`get` rather than treat
    ``MultiTrajRepr``'s output as a Repr directly.

    Implementation note: this primitive is incremental-only by design.
    The batch ``forward`` interface is supported by delegating to each
    inner traj_repr's ``forward``, and returning a stacked Repr keyed
    by name in a side-channel dict.
    """

    def __init__(self, accums: dict) -> None:
        super().__init__()
        # accums: dict[str, TrajRepr-conforming object]
        self._accums = dict(accums)
        # Register children for nn.Module bookkeeping when applicable.
        for k, v in self._accums.items():
            if isinstance(v, nn.Module):
                self.add_module(k, v)

    def forward(self, state_repr: Repr, evidence: ProofEvidence) -> "MultiRepr":
        out = {k: a.forward(state_repr, evidence) for k, a in self._accums.items()}
        return MultiRepr(reprs=out)

    def init(self, B: int, device) -> "MultiRepr":
        return MultiRepr(reprs={k: a.init(B, device) for k, a in self._accums.items()})

    def step(self, accum: "MultiRepr", state_repr: Repr,
              info: Optional[SelectInfo]) -> "MultiRepr":
        return MultiRepr(reprs={
            k: a.step(accum.reprs[k], state_repr, info)
            for k, a in self._accums.items()
        })


@dataclass
class MultiRepr:
    """Sidecar carrier for :class:`MultiTrajRepr`. Holds a dict of named ``Repr``."""

    reprs: dict

    def get(self, name: str) -> Repr:
        return self.reprs[name]

    def __post_init__(self) -> None:
        if not self.reprs:
            raise ValueError("MultiRepr requires at least one named Repr")


# ═══════════════════════════════════════════════════════════════════════
# Rule-MLP per-depth aggregator (R2N)
# ═══════════════════════════════════════════════════════════════════════


class RuleMLPTrajRepr(nn.Module):
    """R2N's per-depth rule-specific MLP + min across depths (framework.pdf §6.3).

    For each (proof, depth) with rule index ``r_d = rule_idx[B, P, d]``::

        e_d = MLP_r(state_repr_embedding_at_depth_d)

    Then aggregate per-proof depth-wise via element-wise min::

        traj_emb = min_d(e_d)

    Requires ``state_repr.embeddings`` with shape ``[B, C, D, E_in]``
    (one embedding per (proof, depth)). Output shape ``[B, C, E_out]``.

    Per-rule MLPs are stored as ``[R, ...]`` parameter tensors and
    gathered by ``rule_idx``. Incremental ``init/step`` mode is not
    implemented because R2N is canonically exhaustive (one batch
    forward call); use :class:`CumulativeLogTrajRepr` or
    :class:`PolicyProductTrajRepr` for sequential search instead.
    """

    def __init__(
        self,
        num_rules: int,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_rules < 1:
            raise ValueError("num_rules must be >= 1")
        h = hidden_dim or out_dim
        self.num_rules = num_rules
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Per-rule two-layer MLP, stored as grouped parameters.
        self.l1 = nn.Parameter(torch.empty(num_rules, in_dim, h))
        self.b1 = nn.Parameter(torch.zeros(num_rules, h))
        self.l2 = nn.Parameter(torch.empty(num_rules, h, out_dim))
        self.b2 = nn.Parameter(torch.zeros(num_rules, out_dim))
        nn.init.kaiming_uniform_(self.l1, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.l2, a=5 ** 0.5)

    def forward(self, state_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not state_repr.has_embeddings:
            raise ValueError("RuleMLPTrajRepr requires state_repr.embeddings")
        emb = state_repr.embeddings           # [B, C, D, E_in]
        if emb.dim() != 4:
            raise ValueError(
                f"RuleMLPTrajRepr expected state_repr.embeddings shape [B, C, D, E_in]; "
                f"got {tuple(emb.shape)}. Pair with ConcatStateRepr or similar."
            )
        rule_idx = evidence.rule_idx          # [B, C, D]
        if rule_idx.dim() != 3:
            raise ValueError(
                f"RuleMLPTrajRepr expects evidence.rule_idx shape [B, C, D]; got {tuple(rule_idx.shape)}"
            )

        # Apply per-rule MLP via grouped parameter gather.
        l1_g = self.l1[rule_idx]              # [B, C, D, in_dim, h]
        b1_g = self.b1[rule_idx]              # [B, C, D, h]
        h1 = F.relu(torch.einsum("...e,...eh->...h", emb, l1_g) + b1_g)
        l2_g = self.l2[rule_idx]              # [B, C, D, h, out_dim]
        b2_g = self.b2[rule_idx]              # [B, C, D, out_dim]
        e_d = torch.einsum("...h,...hk->...k", h1, l2_g) + b2_g   # [B, C, D, out_dim]

        # Min over depth dim, masking padded depths with +inf.
        body_count = evidence.body_count                    # [B, C, D]
        depth_valid = body_count > 0
        big = torch.finfo(e_d.dtype).max
        masked = torch.where(
            depth_valid.unsqueeze(-1), e_d, torch.full_like(e_d, big),
        )
        reduced = masked.min(dim=-2).values                  # [B, C, out_dim]
        # If all depths invalid, collapse-from-+inf → 0.
        any_valid = depth_valid.any(dim=-1)
        reduced = torch.where(
            any_valid.unsqueeze(-1), reduced, torch.zeros_like(reduced),
        )
        return Repr(embeddings=reduced)

    def init(self, B: int, device) -> Repr:
        raise TypeError(
            "RuleMLPTrajRepr is batch-only (R2N is exhaustive). Call "
            "forward(state_repr, evidence) instead."
        )

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr:
        raise TypeError(
            "RuleMLPTrajRepr is batch-only (R2N is exhaustive). Use "
            "PolicyProductTrajRepr or CumulativeLogTrajRepr for sequential search."
        )


__all__ = [
    "BestCumulativeTrajRepr",
    "BestEverStateScoreTrajRepr",
    "BestPrefixAvgTrajRepr",
    "CumulativeLogTrajRepr",
    "FinalStateScoresTrajRepr",
    "FinalStepLogScoreTrajRepr",
    "MinStepTrajRepr",
    "MultiRepr",
    "MultiTrajRepr",
    "PolicyProductTrajRepr",
    "RuleMLPTrajRepr",
    "SBRBodyMinTrajRepr",
    "TNormTrajRepr",
]
