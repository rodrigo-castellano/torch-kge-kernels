"""Shared PyTorch kernels for KGE sampling and scoring."""

from .sampler import Sampler, SamplerConfig
from .scoring import (
    kge_score_all_heads,
    kge_score_all_tails,
    kge_score_k_heads,
    kge_score_k_tails,
    kge_score_triples,
    precompute_partial_scores,
    score_partial_atoms,
)
from .training_sampling import (
    build_known_triple_hash_tensor,
    compute_bernoulli_probs,
    sample_batch_negatives,
    sample_random_negatives,
    set_seed,
    sorted_membership,
)

__all__ = [
    "Sampler",
    "SamplerConfig",
    "build_known_triple_hash_tensor",
    "compute_bernoulli_probs",
    "kge_score_all_heads",
    "kge_score_all_tails",
    "kge_score_k_heads",
    "kge_score_k_tails",
    "kge_score_triples",
    "precompute_partial_scores",
    "sample_batch_negatives",
    "sample_random_negatives",
    "score_partial_atoms",
    "set_seed",
    "sorted_membership",
]
