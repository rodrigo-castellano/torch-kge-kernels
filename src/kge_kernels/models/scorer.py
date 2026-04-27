"""Default scorer + memory heuristic for tkk-native KGE models.

The model layer owns these because both functions are about the model:

- :func:`kge_default_scorer` knows the ``model.score(h, r, t)`` API and
  adapts it to the evaluator's
  ``ScoreFn`` shape ``(q_buf [B, 3], pool_buf [B, C], mode) -> [B, C]``.
- :func:`recommended_eval_batch_size` reads ``model.dim`` /
  ``model.half_dim`` to pick a safe per-batch memory budget.

The evaluator (:class:`kge_kernels.eval.RankingEvaluator`) is the
consumer; it never has to know how a tkk model scores or how big a
batch it can handle — the model layer owns those facts.
"""
from __future__ import annotations

from typing import Literal

from torch import Tensor, nn


def kge_default_scorer(
    model: nn.Module,
    q_buf: Tensor,     # [B, 3]  int64, columns (r, h, t)
    pool_buf: Tensor,  # [B, C]  int64, entity indices to score
    mode: Literal["head", "tail"],
) -> Tensor:           # [B, C]  float
    """Default scorer for tkk-native KGE models with ``model.score(h, r, t)``.

    Three steps in one expression:

    1. **Reshuffle**: ``q_buf`` columns are ``(r, h, t)``; ``model.score``
       takes ``(h, r, t)``.
    2. **Mode dispatch**: pass ``None`` for the corrupted side, triggering
       the model's all-entities matmul fast path → ``[B, |E|]``.
    3. **Gather**: pull the ``C`` candidate columns from ``[B, |E|]`` via
       ``pool_buf``.

    Mode is a Python string fixed per compile — every call inside a
    given ``RankingEvaluator`` invocation has the same value, so
    ``torch.compile`` specializes on it without a tensor branch.
    """
    r = q_buf[:, 0]
    if mode == "tail":
        all_scores = model.score(q_buf[:, 1], r, None)
    else:
        all_scores = model.score(None, r, q_buf[:, 2])
    return all_scores.gather(1, pool_buf)


def recommended_eval_batch_size(
    model: nn.Module,
    num_entities: int,
    budget_gb: float = 2.0,
) -> int:
    """Largest safe compile-graph batch size for :func:`kge_default_scorer`.

    Default assumes matmul-style ``[B, |E|]`` output (ComplEx, DistMult,
    TransE, ModE, TuckER): peak intermediate is bounded by
    ``B × |E| × 4 bytes``, so ``B ≤ budget_gb × 2^30 / (4|E|)``.

    RotatE-family broadcasts to a ``[B, |E|, half_dim]`` intermediate —
    one dimension larger. Detect via ``getattr(model, 'half_dim', None)``
    and factor it into the budget.
    """
    half_dim = getattr(model, "half_dim", None)
    budget_b = budget_gb * (1 << 30)
    if half_dim is not None:
        bytes_per_B = 4 * int(num_entities) * int(half_dim)    # [B, |E|, H]
        return max(8, min(2048, int(budget_b / max(bytes_per_B, 1))))
    bytes_per_B = 4 * int(num_entities)                        # [B, |E|]
    return max(16, min(4096, int(budget_b / max(bytes_per_B, 1))))


__all__ = ["kge_default_scorer", "recommended_eval_batch_size"]
