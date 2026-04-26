"""Default scorer + memory heuristic for tkk-native KGE models.

Free functions, not methods. The :class:`~kge_kernels.eval.RankingEvaluator`
takes any callable matching :type:`~kge_kernels.eval.ScoreFn`; for a plain
KGE model with a ``score(h, r, t)`` API, wrap it in :func:`kge_default_scorer`.
For a model that has its own ``eval_scores`` method (ns ``ReasonerModel``,
DpRL ``PPOScorer``), pass the bound method directly.
"""
from __future__ import annotations

from torch import Tensor, nn

from .candidates import Mode


def kge_default_scorer(
    model: nn.Module,
    q_buf: Tensor,     # [B, 3]  int64, columns (r, h, t)
    pool_buf: Tensor,  # [B, P]  int64, entity indices to score
    mode: Mode,
) -> Tensor:           # [B, P]  float
    """Default scorer for tkk-native KGE models with ``model.score(h, r, t)``.

    Three steps in one expression:

    1. **Reshuffle**: ``q_buf`` columns are ``(r, h, t)``; ``model.score``
       takes ``(h, r, t)``.
    2. **Mode dispatch**: pass ``None`` for the corrupted side, triggering
       the model's all-entities matmul fast path → ``[B, |E|]``.
    3. **Gather**: pull the ``P`` candidate columns from ``[B, |E|]`` via
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
