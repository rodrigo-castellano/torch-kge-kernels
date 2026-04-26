"""Scoring pipeline: types, sampler, partial scoring.

Three modules:

- :mod:`.types` — shared dataclasses (``SamplerConfig``).
- :mod:`.sampler` — :class:`Sampler` (head/tail/both corruption with
  filter, domain pools, optional validity-mask return) and the
  :class:`BernoulliSampler` subclass for per-triple coin-flip
  corruption (Bordes et al. trick). Probabilities for the latter come
  from :meth:`BernoulliSampler.compute_probs`.
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
    PartialScorer,
)
from .sampler import (
    BernoulliSampler,
    Sampler,
)
from .types import (
    LongTensor,
    SamplerConfig,
)

__all__ = [
    # Types
    "LongTensor",
    "SamplerConfig",
    # Sampler
    "BernoulliSampler",
    "Sampler",
    # Partial scoring
    "LazyPartialScorer",
    "PartialScorer",
]
