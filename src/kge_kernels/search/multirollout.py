"""MultiRolloutSearcher: K rollouts of any base Searcher with Gumbel noise.

Higher-order Searcher: takes any base that supports
``set_gumbel_scale(float)`` and runs it K times, each with a
different scale. Per-query (and per-mode), keeps the element-wise max
across rollouts.

This is the post-eval-refactor home of DpRL's old
``eval_ppo_multirollout.py`` algorithm. It composes with any base
Searcher (DpRL's compiled fast-path Searcher in ``ppo/``, or any tkk
reference Searcher).
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .searcher import Searcher


class MultiRolloutSearcher(nn.Module):
    """Run a base Searcher K times with different Gumbel scales; per-query max."""

    def __init__(self, base: Searcher, *, scales: Sequence[float]) -> None:
        super().__init__()
        if not hasattr(base, "set_gumbel_scale"):
            raise TypeError(
                f"{type(base).__name__} doesn't support set_gumbel_scale; "
                f"MultiRolloutSearcher requires it."
            )
        if len(scales) < 1:
            raise ValueError("scales must have at least one entry")
        self.base = base
        self.scales = tuple(scales)

    @torch.no_grad()
    def __call__(self, queries: Tensor) -> Dict[str, Tensor]:
        best = None
        for scale in self.scales:
            self.base.set_gumbel_scale(scale)
            scores = self.base(queries)
            if best is None:
                best = {k: v.clone() for k, v in scores.items()}
            else:
                for k in best:
                    best[k] = torch.maximum(best[k], scores[k])
        assert best is not None
        return best

    def set_gumbel_scale(self, scale: float) -> None:
        """Forward to base — supports nested MultiRolloutSearcher composition."""
        self.base.set_gumbel_scale(scale)


__all__ = ["MultiRolloutSearcher"]
