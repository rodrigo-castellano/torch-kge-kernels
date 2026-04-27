"""Generic representation container shared by framework primitives.

A ``Repr`` carries any combination of three payload kinds:

  - ``embeddings`` — vector per element, shape ``[*leading, E]``
  - ``scores``     — scalar per element, shape ``[*leading]``
  - ``summaries``  — named per-trajectory tensors, ``Dict[str, Tensor]``
                     where each value has shape ``[B]`` (or ``[B, *]``)

The first two flow through atom_repr → state_repr → traj_repr; the third
is populated by Searchers that track trajectory-level statistics
(cumulative_log, success, depths, p_end, v_pos, v_neg, kge_embed, …) and
consumed by trajectory-scoring QueryReprs.

At least one of the three must be set. Embeddings and scores must share
leading shape when both are present; summaries values are not shape-checked
against them (they live at the trajectory level, not the atom/state level).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from torch import Tensor


@dataclass
class Repr:
    """Container for embeddings, scores, and/or trajectory summaries.

    At least one of ``embeddings`` / ``scores`` / ``summaries`` must be set.
    When ``embeddings`` and ``scores`` are both present their leading
    dimensions must be identical (``embeddings`` has one extra trailing
    embedding dim ``E``). ``summaries`` values are independent.
    """

    embeddings: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    summaries: Optional[Dict[str, Tensor]] = None

    def __post_init__(self) -> None:
        if (self.embeddings is None and self.scores is None
                and self.summaries is None):
            raise ValueError(
                "Repr requires at least one of embeddings / scores / summaries"
            )
        if self.embeddings is not None and self.scores is not None:
            emb_lead = tuple(self.embeddings.shape[:-1])
            sc_lead = tuple(self.scores.shape)
            if emb_lead != sc_lead:
                raise ValueError(
                    f"Repr leading shape mismatch: embeddings {emb_lead} vs scores {sc_lead}"
                )

    @property
    def has_embeddings(self) -> bool:
        return self.embeddings is not None

    @property
    def has_scores(self) -> bool:
        return self.scores is not None

    @property
    def has_summaries(self) -> bool:
        return self.summaries is not None

    @property
    def leading_shape(self) -> tuple:
        """Shape excluding the embedding dim. Falls back to scores shape."""
        if self.embeddings is not None:
            return tuple(self.embeddings.shape[:-1])
        if self.scores is not None:
            return tuple(self.scores.shape)
        raise ValueError("Repr has neither embeddings nor scores; leading_shape undefined")


__all__ = ["Repr"]
