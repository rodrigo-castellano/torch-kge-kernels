"""Scoring pipeline: types, KGE entry points, sampler, partial scoring, bridges.

Five modules:

- :mod:`.types` — shared dataclasses + protocol (``KGEBackend``,
  ``CorruptionOutput``, ``ScoreOutput``, ``SamplerConfig``,
  ``SupportsCorruptWithMask``).
- :mod:`.sampler` — vectorized corruption (filter, domain pools,
  Bernoulli) plus on-the-fly Python-list helpers.
- :mod:`.kge` — KGE scoring entry points: model adapter, low-level
  triple / all-tail / all-head kernels, atom-type classification,
  remap-aware triple scoring, unified ``score()``.
- :mod:`.partial` — partial-atom scoring (precomputed tables + lazy
  per-batch caching).
- :mod:`.bridges` — learnable RL+KGE score fusion modules (Linear /
  Gated / PerPredicate / MLP) plus the trainer.
"""

from .bridges import (
    GatedBridge,
    LinearBridge,
    MLPBridge,
    NeuralBridgeTrainer,
    PerPredicateBridge,
)
from .kge import (
    build_backend,
    classify_atoms,
    kge_score_all_heads,
    kge_score_all_heads_dchunked,
    kge_score_all_tails,
    kge_score_all_tails_dchunked,
    kge_score_triples,
    kge_score_triples_remapped,
    precompute_partial_scores as precompute_partial_scores_from_model,
    score,
)
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
    KGEBackend,
    LongTensor,
    SamplerConfig,
    ScoreAllHeadsFn,
    ScoreAllTailsFn,
    ScoreOutput,
    ScoreTriplesFn,
    SupportsCorruptWithMask,
)

__all__ = [
    # Types
    "CorruptionOutput",
    "KGEBackend",
    "LongTensor",
    "SamplerConfig",
    "ScoreAllHeadsFn",
    "ScoreAllTailsFn",
    "ScoreOutput",
    "ScoreTriplesFn",
    "SupportsCorruptWithMask",
    # Sampler
    "Sampler",
    "compute_bernoulli_probs",
    "corrupt",
    "corrupt_to_lists",
    "corrupt_with_topup",
    # KGE entry points
    "build_backend",
    "classify_atoms",
    "kge_score_all_heads",
    "kge_score_all_heads_dchunked",
    "kge_score_all_tails",
    "kge_score_all_tails_dchunked",
    "kge_score_triples",
    "kge_score_triples_remapped",
    "precompute_partial_scores",
    "precompute_partial_scores_from_model",
    "score",
    # Partial scoring
    "LazyPartialScorer",
    "score_partial_atoms",
    # Bridges
    "GatedBridge",
    "LinearBridge",
    "MLPBridge",
    "NeuralBridgeTrainer",
    "PerPredicateBridge",
]
