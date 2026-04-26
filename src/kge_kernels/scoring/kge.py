"""Single KGE scoring entry point.

Callers pass a tkk-native model (anything inheriting from
:class:`kge_kernels.models.base.KGEBase`) and bind whichever of ``h``,
``r``, ``t`` they have; the unbound side, if any, is scored
exhaustively. ``sigmoid`` controls output normalization (raw logit vs
probability) and ``d_chunk`` opts into the model's chunked-D variant
for memory-efficient exhaustive scoring.

Wrapping the model (DataParallel, ``torch.compile``) is the caller's
responsibility — unwrap before passing in if needed.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def kge_score(
    model: nn.Module,
    h: Optional[Tensor],
    r: Tensor,
    t: Optional[Tensor],
    *,
    sigmoid: bool = True,
    d_chunk: Optional[int] = None,
) -> Tensor:
    """Score (h, r, t) triples or rank against all heads/tails.

    Argument order matches tkk's ``model.score(h, r, t)`` so positional
    calls are unambiguous: head, relation, tail. Pass ``None`` for the
    side you want ranked exhaustively.

    Modes (decided by which of ``h``/``t`` are bound):

    - ``h, r, t`` all bound:    score given triples → ``[B]``
    - ``h, r`` bound, ``t=None``: rank all tails for ``(h, r)`` → ``[B, E]``
    - ``h=None``, ``r, t`` bound: rank all heads for ``(r, t)`` → ``[B, E]``

    Args:
        model: tkk-native KGE model exposing ``score`` /
            ``score_all_*_dchunked`` (see ``kge_kernels.models.base``).
        h, r, t: index tensors in (head, relation, tail) order. Exactly
            one of ``h`` / ``t`` may be ``None`` for exhaustive ranking;
            ``r`` is always required.
        sigmoid: apply ``torch.sigmoid`` to the model's raw output
            (default True — probabilities). Pass ``False`` for raw
            logits (e.g. inside BCE-with-logits training).
        d_chunk: when set and exhaustive mode, prefer the model's
            ``score_all_{tails,heads}_dchunked`` for memory-efficient
            chunked computation. Caller is responsible for ensuring the
            model implements it.
    """
    if d_chunk is not None and h is not None and t is None:
        raw = model.score_all_tails_dchunked(h, r, d_chunk=d_chunk)
    elif d_chunk is not None and h is None and t is not None:
        raw = model.score_all_heads_dchunked(r, t, d_chunk=d_chunk)
    else:
        raw = model.score(h, r, t)
    return torch.sigmoid(raw) if sigmoid else raw


__all__ = ["kge_score"]
