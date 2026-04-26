"""Atom-shape classification for proof-state scoring.

Generic atom-type masks: padding, true terminal, false terminal, ground
(both args constants), partial (one variable / one constant). Used by
KGE-state-scoring composites where ground atoms get a KGE score and
partial atoms get a precomputed table lookup score.

The convention (constant ids in ``[1..constant_no]``, variables ids
``> constant_no``) is the one used by both DPrL and torch-ns; this
module documents it as the canonical convention.
"""
from __future__ import annotations

from typing import Tuple

from torch import Tensor


def classify_atoms(
    preds: Tensor,
    args1: Tensor,
    args2: Tensor,
    constant_no: int,
    padding_idx: int,
    true_pred_idx: int,
    false_pred_idx: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute atom-type masks. Pure tensor ops, branchless.

    Atoms are classified by their predicate id (padding sentinel /
    True / False / regular) and the bind-status of their arguments
    (constant id in ``[1..constant_no]`` versus runtime-variable id
    ``> constant_no``).

    Args:
        preds: ``[N]`` predicate indices.
        args1, args2: ``[N]`` argument indices.
        constant_no: largest constant id; ids greater than this are variables.
        padding_idx: predicate id reserved for padding rows.
        true_pred_idx: predicate id for the ⊤ terminal.
        false_pred_idx: predicate id for the ⊥ terminal.

    Returns:
        ``(is_padding, is_true, is_false, is_ground, is_partial)`` —
        broadcast-compatible boolean masks. ``is_ground`` covers atoms
        with both arguments bound to constants. ``is_partial`` covers
        atoms with exactly one argument bound (one variable, one
        constant).
    """
    is_padding = preds == padding_idx
    is_true = preds == true_pred_idx
    is_false = preds == false_pred_idx
    is_terminal = is_true | is_false
    a1_const = (args1 > 0) & (args1 <= constant_no)
    a2_const = (args2 > 0) & (args2 <= constant_no)
    is_ground = ~is_padding & ~is_terminal & a1_const & a2_const
    is_partial = ~is_padding & ~is_terminal & ~is_ground & (a1_const | a2_const)
    return is_padding, is_true, is_false, is_ground, is_partial


__all__ = ["classify_atoms"]
