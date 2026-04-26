"""Searcher protocol + ScoreFn type + reshape adapter.

The framework PDF §6 specifies the canonical scoring-loop output as
``Dict[str, Tensor]`` (per-query, per-mode score). tkk's
:class:`~kge_kernels.eval.RankingEvaluator` consumes a different
contract: ``ScoreFn(q_buf, pool_buf, mode) -> [B, P]`` (per-pool-entry
score). The two contracts are reconciled by one tiny adapter,
:func:`make_scorer_from_searcher`, which performs the K-major flat-pool
reshape outside any compile boundary.

Concrete Searcher classes live in their own modules in this package
(``greedy.py``, ``beam.py``, ``multi_restart.py``, ``multirollout.py``,
``exhaustive.py``, ``direct.py``).
"""
from __future__ import annotations

from typing import Callable, Dict, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

Mode = Literal["head", "tail"]
ScoreFn = Callable[[Tensor, Tensor, Mode], Tensor]


@runtime_checkable
class Searcher(Protocol):
    """Per-query search strategy returning ``{mode_name: [N]}``.

    Concrete classes compose framework primitives in a
    ``search_and_score``-equivalent loop. They may also use
    method-specific specializations (e.g., DpRL's compiled CUDA-graph
    rollout) — the Protocol only requires the callable signature.
    """

    def __call__(self, queries: Tensor) -> Dict[str, Tensor]: ...


def make_scorer_from_searcher(searcher: Searcher, mode_key: str) -> ScoreFn:
    """Reshape adapter: ``Searcher`` → tkk ``ScoreFn``.

    ``RankingEvaluator`` calls ``ScoreFn(q_buf, pool_buf, mode)`` per
    chunk × per mode. ``q_buf [B, 3]`` is a batch of (rel, head, tail)
    queries; ``pool_buf [B, P]`` is the candidate-entity pool per
    query. We construct the K-major flat triple pool and feed it to
    the searcher as if each pool slot were an independent query, then
    reshape the result back to ``[B, P]``.

    Pure tensor ops, eager. Not part of the compile boundary.
    """
    @torch.no_grad()
    def scorer(q_buf: Tensor, pool_buf: Tensor, mode: Mode) -> Tensor:
        B, P = pool_buf.shape
        if mode == "head":
            col = 1
        elif mode == "tail":
            col = 2
        else:
            raise ValueError(f"mode must be 'head' or 'tail', got {mode}")
        triples = q_buf.unsqueeze(0).expand(P, B, 3).clone()
        triples[:, :, col] = pool_buf.t()
        flat = triples.reshape(P * B, 3)              # K-major: index [k*B + b]
        result = searcher(flat)
        if mode_key not in result:
            raise KeyError(
                f"searcher did not return '{mode_key}' in its result dict; "
                f"keys: {sorted(result.keys())}"
            )
        return result[mode_key].view(P, B).t().contiguous()
    return scorer


__all__ = ["Mode", "ScoreFn", "Searcher", "make_scorer_from_searcher"]
