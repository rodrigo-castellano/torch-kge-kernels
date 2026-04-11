"""Loss functions for KGE and neural-symbolic training.

All losses are pure ``nn.Module`` classes with no dependencies outside
``torch``. Shape conventions:

  ``y_pred``: ``[B, N]`` scores or logits
  ``y_true``: ``[B, N]`` labels, where 1 = positive, 0 = negative, -1 = padding
              (unless otherwise noted)

Factory:
  ``build_loss(name, **kwargs)`` returns an instance given a config name.
  Names: ``bce_masked``, ``bce_weighted``, ``bce_ragged``, ``bce_balanced``,
  ``pairwise``, ``pairwise_balanced``, ``categorical``, ``hinge``, ``l2``.
"""
from __future__ import annotations

from typing import Any

from torch import nn

from .classification import (
    BinaryCrossEntropyRagged,
    BinaryCrossEntropyWithMask,
    CategoricalCrossEntropyRagged,
    WeightedBinaryCrossEntropy,
)
from .ranking_losses import (
    HingeLossRagged,
    L2LossRagged,
    NSSALoss,
    PairwiseCrossEntropyRagged,
)


def build_loss(name: str, **kwargs: Any) -> nn.Module:
    """Factory that maps a name to a loss instance.

    Recognized names::

        bce_masked              -> BinaryCrossEntropyWithMask
        bce_weighted            -> WeightedBinaryCrossEntropy
        bce_ragged              -> BinaryCrossEntropyRagged
        bce_balanced            -> BinaryCrossEntropyRagged(balance_negatives=True)
        pairwise                -> PairwiseCrossEntropyRagged
        pairwise_balanced       -> PairwiseCrossEntropyRagged(balance_negatives=True)
        categorical             -> CategoricalCrossEntropyRagged
        hinge                   -> HingeLossRagged
        l2                      -> L2LossRagged

    Legacy aliases matching ``torch-ns.ns_lib.nn.losses.KgeLossFactory`` are
    also accepted: ``binary_crossentropy``, ``balanced_binary_crossentropy``,
    ``balanced_pairwise_crossentropy``, ``categorical_crossentropy``.
    """
    key = name.lower().replace("-", "_")
    aliases = {
        "binary_crossentropy": "bce_ragged",
        "balanced_binary_crossentropy": "bce_balanced",
        "balanced_pairwise_crossentropy": "pairwise_balanced",
        "categorical_crossentropy": "categorical",
    }
    key = aliases.get(key, key)

    if key == "bce_masked":
        return BinaryCrossEntropyWithMask(**kwargs)
    if key == "bce_weighted":
        return WeightedBinaryCrossEntropy(**kwargs)
    if key == "bce_ragged":
        return BinaryCrossEntropyRagged(**kwargs)
    if key == "bce_balanced":
        kwargs.setdefault("balance_negatives", True)
        return BinaryCrossEntropyRagged(**kwargs)
    if key == "pairwise":
        return PairwiseCrossEntropyRagged(**kwargs)
    if key == "pairwise_balanced":
        kwargs.setdefault("balance_negatives", True)
        return PairwiseCrossEntropyRagged(**kwargs)
    if key == "categorical":
        return CategoricalCrossEntropyRagged(**kwargs)
    if key == "hinge":
        return HingeLossRagged(**kwargs)
    if key == "l2":
        return L2LossRagged()
    if key == "nssa":
        return NSSALoss(**kwargs)
    raise ValueError(f"Unknown loss: {name!r}")


__all__ = [
    "BinaryCrossEntropyRagged",
    "BinaryCrossEntropyWithMask",
    "CategoricalCrossEntropyRagged",
    "HingeLossRagged",
    "L2LossRagged",
    "NSSALoss",
    "PairwiseCrossEntropyRagged",
    "WeightedBinaryCrossEntropy",
    "build_loss",
]
