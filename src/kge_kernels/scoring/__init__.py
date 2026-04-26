"""Scoring pipeline: types, sampler, partial scoring.

Three modules:

- :mod:`.types` — shared dataclasses + protocol (``CorruptionOutput``,
  ``SamplerConfig``, ``SupportsCorruptWithMask``).
- :mod:`.sampler` — vectorized corruption (filter, domain pools,
  Bernoulli) plus on-the-fly Python-list helpers.
- :mod:`.partial` — partial-atom scoring (precomputed tables + lazy
  per-batch caching) on top of ``model.score``.

KGE scoring itself is the model's own ``score(h, r, t, *, d_chunk=None)``
method (see :mod:`kge_kernels.models.base`). Callers that want sigmoid
normalisation just wrap it: ``torch.sigmoid(model.score(h, r, t))``.

RL+KGE fusion bridges and proof-state-shape atom classification live in
the consumer repos (DpRL ``kge_module/bridges`` and
``kge_module/scoring`` respectively) — they depend on caller-side
conventions, not tkk's KGE math.
"""

from .partial import (
    LazyPartialScorer,
    precompute_partial_scores,
    score_partial_atoms,
)
from .sampler import (
    Sampler,
    compute_bernoulli_probs,
    corrupt,
    corrupt_to_lists,
    corrupt_with_topup,
)
from .types import (
    CorruptionOutput,
    LongTensor,
    SamplerConfig,
    SupportsCorruptWithMask,
)

__all__ = [
    # Types
    "CorruptionOutput",
    "LongTensor",
    "SamplerConfig",
    "SupportsCorruptWithMask",
    # Sampler
    "Sampler",
    "compute_bernoulli_probs",
    "corrupt",
    "corrupt_to_lists",
    "corrupt_with_topup",
    # Partial scoring
    "LazyPartialScorer",
    "precompute_partial_scores",
    "score_partial_atoms",
]
