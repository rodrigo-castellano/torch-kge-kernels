"""Eval-time scoring helpers.

Free functions over a tkk-native KGE model. Owned by the eval pipeline
rather than the model — the model exposes only ``score(h, r, t)`` (the
math), and these wrap it with the gather + memory-budget logic the eval
loop needs.

Subclasses with a different scoring path (e.g. ns's ``ReasonerModel``
candidate-pool replay) override by defining their own ``eval_scores``
method on the model; :func:`kge_kernels.eval.evaluate.evaluate` prefers
``model.eval_scores`` when present and falls back to this free function
otherwise.
"""
from __future__ import annotations

from typing import Literal

from torch import Tensor, nn


Mode = Literal["tail", "head"]


def eval_scores(
    model: nn.Module,
    q_buf: Tensor,     # [B, 3]  int64, columns (r, h, t)
    cand_buf: Tensor,  # [B, C]  int64, entity indices to score
    mode: Mode,
) -> Tensor:           # [B, C]  float
    """Score candidates against a query batch — the unified eval hook.

    The compile boundary for :func:`kge_kernels.eval.evaluate.evaluate`.
    Shapes are fixed across calls (``B``, ``C`` constant within a single
    :func:`evaluate` call) so this traces into one CUDA graph.

    Uses the matmul fast path inside ``model.score``: with ``t=None``
    (or ``h=None``) every model computes all-entity scores ``[B, |E|]``
    in one BLAS call, and we gather the ``C`` candidates here.

    ``mode`` is a Python string, not a tensor: ``torch.compile``
    specialises one graph per value, keeping the compiled region
    branch-free.
    """
    r = q_buf[:, 0]
    if mode == "tail":
        h = q_buf[:, 1]
        all_scores = model.score(h, r, None)                   # [B, |E|]
    else:
        t = q_buf[:, 2]
        all_scores = model.score(None, r, t)                   # [B, |E|]
    return all_scores.gather(1, cand_buf)                      # [B, C]


def recommended_eval_batch_size(
    model: nn.Module,
    num_entities: int,
    budget_gb: float = 2.0,
) -> int:
    """Largest safe compile-graph batch size for :func:`eval_scores`.

    Default assumes matmul-style ``[B, |E|]`` output (ComplEx,
    DistMult, TransE, ModE, TuckER): peak intermediate is bounded by
    ``B × |E| × 4 bytes``, so ``B ≤ budget_gb × 2^30 / (4|E|)``.

    RotatE-family broadcasts to a ``[B, |E|, half_dim]`` intermediate —
    one dimension larger. We detect them via ``getattr(model,
    'half_dim', None)`` and factor it into the budget.
    """
    half_dim = getattr(model, "half_dim", None)
    budget_b = budget_gb * (1 << 30)
    if half_dim is not None:
        bytes_per_B = 4 * int(num_entities) * int(half_dim)    # [B, |E|, H]
        return max(8, min(2048, int(budget_b / max(bytes_per_B, 1))))
    bytes_per_B = 4 * int(num_entities)                        # [B, |E|]
    return max(16, min(4096, int(budget_b / max(bytes_per_B, 1))))


__all__ = ["Mode", "eval_scores", "recommended_eval_batch_size"]
