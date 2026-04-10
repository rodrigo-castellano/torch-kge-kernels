"""Generic representation container shared by framework primitives.

A ``Repr`` carries either a vector embedding, a scalar score, or both for
each element of an arbitrary leading shape. The leading shape is whatever
the consumer wants — atom-level, state-level, trajectory-level. Primitives
agree on the convention that:

  - ``embeddings`` has shape ``[*leading, E]``
  - ``scores``     has shape ``[*leading]``

so the same ``Repr`` can flow through ``atom_repr → state_repr → traj_repr
→ query_repr`` without each layer needing a bespoke type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class Repr:
    """Container for embeddings and/or scores with matching leading shape.

    At least one of ``embeddings`` / ``scores`` must be set. When both are
    present their leading dimensions must be identical (``embeddings`` has
    one extra trailing embedding dim ``E``).
    """

    embeddings: Optional[Tensor] = None
    scores: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.embeddings is None and self.scores is None:
            raise ValueError("Repr requires at least one of embeddings or scores")
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
    def leading_shape(self) -> tuple:
        """Shape excluding the embedding dim. Falls back to scores shape."""
        if self.embeddings is not None:
            return tuple(self.embeddings.shape[:-1])
        assert self.scores is not None
        return tuple(self.scores.shape)


__all__ = ["Repr"]
