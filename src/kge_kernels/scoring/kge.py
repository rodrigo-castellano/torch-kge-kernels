"""Single KGE scoring entry point.

Thin wrapper over the model's own ``score(h, r, t, *, d_chunk=None)``
method that adds optional sigmoid normalisation. Wrapping the model
(DataParallel, ``torch.compile``) is the caller's responsibility —
unwrap before passing in if needed.
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

    Argument order matches tkk's ``model.score(h, r, t)``: head, relation,
    tail. Pass ``None`` for the side you want ranked exhaustively.

    Modes (decided by which of ``h``/``t`` are bound):

    - ``h, r, t`` all bound:    score given triples → ``[B]``
    - ``h, r`` bound, ``t=None``: rank all tails for ``(h, r)`` → ``[B, E]``
    - ``h=None``, ``r, t`` bound: rank all heads for ``(r, t)`` → ``[B, E]``

    Args:
        model: tkk-native KGE model (see ``kge_kernels.models.base``).
        h, r, t: index tensors in (head, relation, tail) order. Exactly
            one of ``h`` / ``t`` may be ``None`` for exhaustive ranking;
            ``r`` is always required.
        sigmoid: apply ``torch.sigmoid`` to the model's raw output
            (default True — probabilities). Pass ``False`` for raw
            logits (e.g. inside BCE-with-logits training).
        d_chunk: forwarded to ``model.score``. ``None`` = one pass over
            all entities; integer enables chunked-D scoring on models
            that support it (RotatE / RotatENS). See base.score docstring.
    """
    raw = model.score(h, r, t, d_chunk=d_chunk)
    return torch.sigmoid(raw) if sigmoid else raw


__all__ = ["kge_score"]
