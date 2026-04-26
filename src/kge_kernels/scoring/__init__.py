"""Scoring pipeline: types, KGE entry point, sampler, partial scoring.

Four modules:

- :mod:`.types` — shared dataclasses + protocol (``CorruptionOutput``,
  ``SamplerConfig``, ``SupportsCorruptWithMask``).
- :mod:`.sampler` — vectorized corruption (filter, domain pools,
  Bernoulli) plus on-the-fly Python-list helpers.
- :mod:`.kge` — single ``kge_score`` entry point: triple scoring,
  exhaustive head/tail ranking, optional sigmoid normalization, optional
  chunked-D for memory-efficient ranking.
- :mod:`.partial` — partial-atom scoring (precomputed tables + lazy
  per-batch caching) on top of :func:`kge_score`.

RL+KGE fusion bridges and proof-state-shape atom classification live in
the consumer repos (DpRL ``kge_module/bridges`` and
``kge_module/scoring`` respectively) — they depend on caller-side
conventions, not tkk's KGE math.
"""

from .kge import kge_score
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
    # KGE entry point
    "kge_score",
    # Partial scoring
    "LazyPartialScorer",
    "precompute_partial_scores",
    "score_partial_atoms",
]
