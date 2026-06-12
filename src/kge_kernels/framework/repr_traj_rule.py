"""K-iteration pool-iter loops for the rule-based reasoning path.

Each iteration:
  1. Gather body atom embeddings/scores from the pool by ``firings.body_pool_idx``
  2. Apply per-firing rule operator (``RuleStateRepr``) to get predictions
  3. Mask invalid scatter slots
  4. ``scatter_reduce(amax)`` predictions into a fresh base; merge with prior pool
  5. Track which slots were ever a target of a valid scatter (``ever_written``)

After ``K`` iterations the pool encodes the K-step closure of the rule
program. The downstream :class:`QueryRepr` gathers query slots from the
pool to produce per-query scores.

Three concrete loops:

* :class:`MinPoolLoop` — SBR scalar pool ``[N_pool]``. Identity = +1
  (T-norm-min), invalid body slots gather as +1 so they don't affect min.
* :class:`DCRPoolLoop` — DCR scalar pool ``[N_pool]``. Same scatter
  pattern as Min; the per-firing operator carries the filter+sign.
* :class:`RuleMLPPoolLoop` — R2N embedding pool ``[N_pool, E]``. Two
  prediction modes (``'head'`` / ``'full'``); see module docstring of
  :class:`RuleMLPPoolLoop`.

For SBR/DCR the per-firing operator is associative under ``max`` over
rules, so the pool-iter result is identical to the per-tree
forward-chaining "View A" of ``docs/reasoners_framework_vs_keras.md``.
For R2N the rule MLP is non-associative, so the pool-iter result is the
keras-faithful "View C" — different from any per-tree fold and the only
correct chaining of MLP_R through depth.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..models.base import det_gather_rows
from .protocols import RuleStateRepr
from .types import FiringsTensors


# ═══════════════════════════════════════════════════════════════════════
# Embedding-pool variant (R2N)
# ═══════════════════════════════════════════════════════════════════════


class RuleMLPPoolLoop(nn.Module):
    """K-iteration pool-iter loop with embedding pool ``[N_pool, E]`` (R2N).

    Two prediction modes:

    * ``'head'`` — rule MLP predicts only the head atom's embedding
      (``num_atoms_out=1``); scatter to ``firings.head_pool_idx`` only.
      Used when ``resnet=False`` (paper convention: no body update means
      no chained body re-evaluation).
    * ``'full'`` — rule MLP predicts embeddings for the head AND every
      body atom (``num_atoms_out=M+1``); scatter to all of body+head
      pool indices. Body atoms get re-written every iteration, so a
      body atom's value can propagate into the next rule firing's body
      that gathers it (cross-firing chaining at the body level).

    Statically known ``K`` keeps the loop compile-unroll-friendly.
    """

    def __init__(
        self,
        K: int,
        *,
        prediction_type: Literal["head", "full"] = "head",
        aggregation: Literal["max", "mean", "sum", "gated"] = "max",
    ) -> None:
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if prediction_type not in ("head", "full"):
            raise ValueError(
                f"prediction_type must be 'head' or 'full', got {prediction_type!r}"
            )
        if aggregation not in ("max", "mean", "sum", "gated"):
            raise ValueError(
                f"aggregation must be max|mean|sum|gated, got {aggregation!r}")
        self.K = K
        self.prediction_type = prediction_type
        # Cross-firing aggregation at the same atom. "max" (default) is the
        # keras-faithful per-dimension envelope (tensor_scatter_nd_max):
        # score is then MONOTONE in the firing set — any added firing can
        # only raise an atom's score, which collapses under low-precision
        # deep evidence (torch-ns docs/MONOTONICITY.md). "mean"/"sum" are
        # the other keras R2NReasoningLayer options. "gated" selects ONE
        # whole prediction vector per atom — the firing whose bodies score
        # best under the score-aligned pool (t-norm min of sigmoid(sum(dims)),
        # i.e. exactly the guided-beam body score): a weak/hypothesized-body
        # firing can no longer displace a factual-bodied one, and the score
        # is NOT monotone in firing count.
        self.aggregation = aggregation

    def forward(
        self,
        pool: Tensor,                # [N_pool, E]
        firings: FiringsTensors,
        state_repr_fn: RuleStateRepr,
    ) -> tuple[Tensor, Tensor]:
        N_f, M = firings.body_pool_idx.shape
        N_pool = pool.shape[0]
        E_in = pool.shape[-1]
        very_negative = -1e4
        ever_written = torch.zeros(N_pool, dtype=torch.bool, device=pool.device)

        if self.prediction_type == "head":
            scatter_target = firings.head_pool_idx.unsqueeze(-1)        # [N_f, 1]
            valid_for_scatter = firings.firing_valid.unsqueeze(-1)      # [N_f, 1]
        else:
            scatter_target = torch.cat(
                [firings.body_pool_idx, firings.head_pool_idx.unsqueeze(-1)],
                dim=-1,
            )                                                            # [N_f, M+1]
            valid_for_scatter = torch.cat(
                [firings.body_atom_valid, firings.firing_valid.unsqueeze(-1)],
                dim=-1,
            )                                                            # [N_f, M+1]

        N_scatter = scatter_target.shape[-1]

        for _ in range(self.K):
            # 1. Gather body atom embeddings; mask invalid body slots to 0.
            body_emb_raw = det_gather_rows(pool, firings.body_pool_idx)  # [N_f, M, E]
            body_emb = body_emb_raw * firings.body_atom_valid.unsqueeze(-1)
            body_flat = body_emb.reshape(N_f, M * E_in)
            # 2. Apply per-firing rule MLP → [N_f, K_out, E]
            preds = state_repr_fn(body_flat, firings.rule_idx)
            # 4. Flatten (firing × scatter_slot) for scatter_reduce.
            target_flat = scatter_target.reshape(N_f * N_scatter)
            valid_flat = valid_for_scatter.reshape(N_f * N_scatter)
            target_expanded = target_flat.unsqueeze(-1).expand(-1, E_in)

            if self.aggregation == "max":
                # 3. Mask invalid scatter slots (drive to -1e4 so
                #    scatter_reduce(amax) skips them in favor of any real
                #    prediction at the same atom).
                preds_m = torch.where(
                    valid_for_scatter.unsqueeze(-1),
                    preds,
                    torch.full_like(preds, very_negative),
                )
                preds_flat = preds_m.reshape(N_f * N_scatter, E_in)
                # 5. scatter_max into a fresh base. Written slots get the
                #    pure max over their MLP predictions (no KGE-init floor);
                #    unwritten slots keep their previous value (KGE on iter 1,
                #    prior iter's value on later iters).
                scattered = torch.full_like(pool, very_negative)
                scattered = scattered.scatter_reduce(
                    dim=0,
                    index=target_expanded,
                    src=preds_flat,
                    reduce="amax",
                    include_self=False,
                )
            elif self.aggregation in ("mean", "sum"):
                preds_z = preds * valid_for_scatter.unsqueeze(-1)   # invalid → 0
                preds_flat = preds_z.reshape(N_f * N_scatter, E_in)
                scattered = torch.zeros_like(pool).scatter_reduce(
                    dim=0, index=target_expanded, src=preds_flat,
                    reduce="sum", include_self=True,
                )
                if self.aggregation == "mean":
                    cnt = torch.zeros(
                        N_pool, dtype=pool.dtype, device=pool.device)
                    cnt.scatter_reduce_(
                        dim=0, index=target_flat,
                        src=valid_flat.to(pool.dtype),
                        reduce="sum", include_self=True)
                    scattered = scattered / cnt.clamp_min(1.0).unsqueeze(-1)
            else:  # "gated": whole-vector winner-take-all by body score
                # Per-firing gate = t-norm (min) over VALID bodies of the
                # score-aligned scalar sigmoid(sum(dims)); body-less firings
                # gate at 1.0 (fact-level certainty).
                body_scalar = torch.sigmoid(body_emb_raw.sum(-1))   # [N_f, M]
                body_scalar = body_scalar.masked_fill(
                    ~firings.body_atom_valid, 1.0)
                gate = body_scalar.amin(dim=-1)                      # [N_f]
                gate_slots = gate.unsqueeze(-1).expand(-1, N_scatter)
                gate_flat = torch.where(
                    valid_for_scatter, gate_slots,
                    torch.full_like(gate_slots, -1.0),
                ).reshape(N_f * N_scatter)
                # Pass 1: best gate per atom slot.
                best = torch.full(
                    (N_pool,), -1.0, dtype=pool.dtype, device=pool.device)
                best.scatter_reduce_(
                    dim=0, index=target_flat, src=gate_flat,
                    reduce="amax", include_self=True)
                # Pass 2: deterministic tie-break — lowest flat index among
                # the gate winners (scatter amin of index over winners).
                is_winner = valid_flat & (gate_flat >= best[target_flat])
                idx = torch.arange(
                    N_f * N_scatter, device=pool.device)
                idx_masked = torch.where(
                    is_winner, idx, torch.full_like(idx, N_f * N_scatter))
                win_idx = torch.full(
                    (N_pool,), N_f * N_scatter, dtype=idx.dtype,
                    device=pool.device)
                win_idx.scatter_reduce_(
                    dim=0, index=target_flat, src=idx_masked,
                    reduce="amin", include_self=True)
                final_win = is_winner & (idx == win_idx[target_flat])
                # Scatter ONLY the winning firing's full vector.
                preds_flat = preds.reshape(N_f * N_scatter, E_in)
                preds_w = torch.where(
                    final_win.unsqueeze(-1), preds_flat,
                    torch.full_like(preds_flat, very_negative))
                scattered = torch.full_like(pool, very_negative)
                scattered = scattered.scatter_reduce(
                    dim=0, index=target_expanded, src=preds_w,
                    reduce="amax", include_self=False,
                )
            iter_written_int = torch.zeros(N_pool, dtype=torch.long, device=pool.device)
            iter_written_int.scatter_reduce_(
                dim=0,
                index=target_flat,
                src=valid_flat.long(),
                reduce="amax",
                include_self=True,
            )
            iter_written = iter_written_int.bool()
            pool = torch.where(
                iter_written.unsqueeze(-1),
                scattered,
                pool,
            )
            # 6. Accumulate ever-written across K iterations.
            ever_written = ever_written | iter_written

        return pool, ever_written


# ═══════════════════════════════════════════════════════════════════════
# Scalar-pool variant (SBR / DCR)
# ═══════════════════════════════════════════════════════════════════════


class _ScalarPoolLoop(nn.Module):
    """Shared scalar-pool ``[N_pool]`` K-iteration loop (SBR / DCR).

    Pool stores one score per atom. Each firing reads body scores,
    applies the per-firing operator (which produces a scalar head
    prediction ``[N_f, 1]``), and scatters into the pool's head slot
    via ``scatter_reduce(amax)``. Scalar pool layout: ``[N_pool]`` —
    no embedding dim.

    Identity for the gather-mask is +1 (T-norm-min identity over [0, 1]
    body scores): invalid body slots gather as +1 so they don't affect
    the per-firing min.

    Subclasses choose the per-firing operator (``MinRuleState`` for SBR,
    ``FilterSignRuleState`` for DCR). The aggregation across firings
    targeting the same atom is always ``max`` (probabilistic disjunction).
    """

    def __init__(self, K: int) -> None:
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = K

    def forward(
        self,
        pool: Tensor,                # [N_pool]
        firings: FiringsTensors,
        state_repr_fn: RuleStateRepr,
    ) -> tuple[Tensor, Tensor]:
        if pool.dim() != 1:
            raise ValueError(
                f"_ScalarPoolLoop expects pool [N_pool]; got {tuple(pool.shape)}"
            )
        N_f, M = firings.body_pool_idx.shape
        N_pool = pool.shape[0]
        very_negative = -1e4
        ever_written = torch.zeros(N_pool, dtype=torch.bool, device=pool.device)

        scatter_target = firings.head_pool_idx                              # [N_f]
        valid_for_scatter = firings.firing_valid                            # [N_f]

        for _ in range(self.K):
            # 1. Gather body atom scores; mask invalid body slots to +1
            #    (T-norm-min identity).
            body_scores = det_gather_rows(pool, firings.body_pool_idx)      # [N_f, M]
            body_scores = torch.where(
                firings.body_atom_valid,
                body_scores,
                torch.ones_like(body_scores),
            )
            # 2. Apply per-firing operator → [N_f, 1] head score.
            preds = state_repr_fn(body_scores, firings.rule_idx)            # [N_f, 1]
            head_score = preds.squeeze(-1)                                  # [N_f]
            # 3. Mask invalid firings to -inf so they don't win the scatter.
            head_score = torch.where(
                valid_for_scatter,
                head_score,
                torch.full_like(head_score, very_negative),
            )
            # 4. scatter_max into a fresh base; unwritten slots keep prior pool.
            scattered = torch.full_like(pool, very_negative)
            scattered = scattered.scatter_reduce(
                dim=0,
                index=scatter_target,
                src=head_score,
                reduce="amax",
                include_self=False,
            )
            iter_written_int = torch.zeros(N_pool, dtype=torch.long, device=pool.device)
            iter_written_int.scatter_reduce_(
                dim=0,
                index=scatter_target,
                src=valid_for_scatter.long(),
                reduce="amax",
                include_self=True,
            )
            iter_written = iter_written_int.bool()
            pool = torch.where(iter_written, scattered, pool)
            ever_written = ever_written | iter_written

        return pool, ever_written


class MinPoolLoop(_ScalarPoolLoop):
    """SBR pool-iter loop. Pair with :class:`MinRuleState`."""


class DCRPoolLoop(_ScalarPoolLoop):
    """DCR pool-iter loop. Pair with :class:`FilterSignRuleState`."""


__all__ = ["DCRPoolLoop", "MinPoolLoop", "RuleMLPPoolLoop"]
