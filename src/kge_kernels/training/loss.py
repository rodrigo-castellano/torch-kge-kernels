"""Unified training-step loss.

Free function over a tkk-native KGE model. Owned by the training
pipeline rather than each model — the model exposes only
``score(h, r, t)`` (the math), and this function wraps it with the
mask-aware BCE reduction the training loop needs.

Subclasses with a different atom layout (e.g. ns's ``ReasonerModel``)
override this by defining their own ``train_step`` method on the
model; :func:`kge_kernels.training.epoch.train_epoch` prefers
``model.train_step`` when present and falls back to this free function
otherwise.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def train_step(
    model: nn.Module,
    pos: Tensor,           # [B, 3]       (r, h, t) positive triples
    neg: Tensor,           # [B, K, 3]    candidate negatives
    mask: Tensor,          # [B, K]       which negatives are valid
    pos_valid: Tensor,     # [B]          which positives are real (not tail-pad)
    *,
    from_logits: bool | None = None,
) -> Tensor:               # scalar loss
    """Mask-aware BCE training loss — the unified compile boundary.

    Every shape is a compile-time constant: ``[B, 3]``, ``[B, K, 3]``,
    ``[B, K]``, ``[B]`` never vary. Mask-aware BCE reduction gates
    invalid negatives (``mask=False``) and padded positives
    (``pos_valid=False``, only non-trivial for the last partial batch
    of an epoch). This makes the whole function
    ``torch.compile(fullgraph=True, mode='reduce-overhead')`` compatible
    — one CUDA graph per ``(model, B, K)`` combo, reused across every
    training step.

    Args:
        model: tkk-native KGE model (must expose ``score(h, r, t)``).
        pos, neg, mask, pos_valid: see shape annotations above.
        from_logits: if ``None`` (default), read
            ``model._train_loss_is_from_logits`` (defaults to ``True``);
            ``True`` uses ``binary_cross_entropy_with_logits``, ``False``
            applies sigmoid then ``binary_cross_entropy``. RotatE-family
            sets ``False`` because raw scores ``γ - ||h rot r - t||`` are
            large enough that BCE-with-logits saturates the gradient on
            confident negatives.
    """
    if from_logits is None:
        from_logits = bool(getattr(model, "_train_loss_is_from_logits", True))

    B, K, _ = neg.shape
    # Flat, static-shape atom layout: [B + B*K, 3], cols (r, h, t).
    neg_flat = neg.reshape(B * K, 3)
    all_items = torch.cat([pos, neg_flat], dim=0)
    # score consumes (h, r, t); items are in (r, h, t).
    scores = model.score(all_items[:, 1], all_items[:, 0], all_items[:, 2])
    pos_scores = scores[:B]                          # [B]
    neg_scores = scores[B:]                          # [B*K]

    pos_w = pos_valid.to(dtype=pos_scores.dtype)     # [B]
    neg_w = mask.to(dtype=neg_scores.dtype).reshape(-1)   # [B*K]

    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    if from_logits:
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, reduction="none")
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, reduction="none")
    else:
        pos_loss = F.binary_cross_entropy(torch.sigmoid(pos_scores), pos_labels, reduction="none")
        neg_loss = F.binary_cross_entropy(torch.sigmoid(neg_scores), neg_labels, reduction="none")

    numer = (pos_loss * pos_w).sum() + (neg_loss * neg_w).sum()
    denom = (pos_w.sum() + neg_w.sum()).clamp_min(1.0)
    return numer / denom


__all__ = ["train_step"]
