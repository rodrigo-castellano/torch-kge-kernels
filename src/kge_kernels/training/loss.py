"""Unified training-step losses.

Free functions over a tkk-native KGE model, sharing the
``(model, pos, neg, mask, pos_valid)`` signature so they plug into
:func:`kge_kernels.training.epoch.train_epoch` interchangeably.

Two losses today:

- :func:`train_step` — mask-aware BCE (default, used when
  ``cfg.loss == "bce"``).
- :func:`nssa_train_step` — self-adversarial negative-sampling loss
  (Sun et al., 2019; used when ``cfg.loss == "nssa"``).

Both compute scores via ``model.score(h, r, t)`` and reduce them
with mask gating so the static-shape last-partial-batch handling in
``train_epoch`` works unchanged.

Subclasses with a different atom layout (e.g. ns's ``ReasonerModel``)
override loss selection by defining their own ``train_step`` method on
the model; ``train_epoch`` prefers ``model.train_step`` when present.
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


def nssa_train_step(
    model: nn.Module,
    pos: Tensor,           # [B, 3]       (r, h, t) positive triples
    neg: Tensor,           # [B, K, 3]    candidate negatives
    mask: Tensor,          # [B, K]       which negatives are valid
    pos_valid: Tensor,     # [B]          which positives are real
    *,
    adv_temp: float = 0.0,
) -> Tensor:               # scalar loss
    """Self-adversarial NS loss (Sun et al., 2019) as a train_epoch plug-in.

    Same shape contract as :func:`train_step` so the unified
    ``train_epoch`` outer loop drives both losses with a single static-
    buffer compiled step. ``adv_temp > 0`` upweights hard negatives via
    a stop-gradient softmax over ``adv_temp * neg_scores``; ``adv_temp
    == 0`` recovers plain uniform negative averaging. Mask gating
    handles invalid negatives and the padded tail of the last partial
    batch.
    """
    B, K, _ = neg.shape
    neg_flat = neg.reshape(B * K, 3)
    all_items = torch.cat([pos, neg_flat], dim=0)
    scores = model.score(all_items[:, 1], all_items[:, 0], all_items[:, 2])
    pos_scores = scores[:B]                            # [B]
    neg_scores = scores[B:].view(B, K)                 # [B, K]

    pos_w = pos_valid.to(dtype=pos_scores.dtype)       # [B]
    neg_w = mask.to(dtype=neg_scores.dtype)            # [B, K]

    pos_loss_per_row = -F.logsigmoid(pos_scores) * pos_w           # [B]
    neg_log = -F.logsigmoid(-neg_scores)                            # [B, K]

    if adv_temp > 0:
        # Numerically-stable masked softmax: subtract per-row max,
        # exp, multiply by mask, normalise. Detached so weights are
        # stop-gradient (matching the reference NSSALoss).
        logits = adv_temp * neg_scores
        logits = logits - logits.max(dim=1, keepdim=True).values
        e = torch.exp(logits) * neg_w                              # [B, K]
        weights = (e / e.sum(dim=1, keepdim=True).clamp_min(1e-12)).detach()
        neg_loss_per_row = (weights * neg_log).sum(dim=1)          # [B]
    else:
        neg_loss_per_row = (
            (neg_log * neg_w).sum(dim=1)
            / neg_w.sum(dim=1).clamp_min(1.0)
        )

    total = pos_loss_per_row + neg_loss_per_row * pos_w            # [B]
    return total.sum() / pos_w.sum().clamp_min(1.0)


__all__ = ["nssa_train_step", "train_step"]
