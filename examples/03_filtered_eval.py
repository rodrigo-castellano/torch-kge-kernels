"""Example 3 — Standalone filtered-ranking evaluation.

Shows how to evaluate an already-trained KGE model on a test split
using ``kge_kernels.eval.RankingEvaluator``. This is what every
published KGE paper computes for its MRR / Hits@1 / Hits@3 / Hits@10
table.

Both exhaustive and sampled evaluation modes are demonstrated:
  - Exhaustive (``k=None``): rank against all entities
  - Sampled (``k=K``): rank against K random candidates per query
"""
from __future__ import annotations

import torch

from kge_kernels.eval import RankingEvaluator, SamplerCandidates
from kge_kernels.models import kge_default_scorer
from kge_kernels.models import TransE
from kge_kernels.scoring import Sampler


def main() -> None:
    torch.manual_seed(0)

    num_entities = 15
    num_relations = 3

    model = TransE(
        num_entities=num_entities, num_relations=num_relations, dim=16
    )

    train = [
        (0, 0, 1), (0, 1, 2), (0, 2, 3), (0, 3, 4),
        (1, 0, 5), (1, 2, 6), (1, 4, 7),
        (2, 1, 8), (2, 3, 9),
    ]
    valid = [(0, 4, 5), (1, 1, 3)]
    test = [(0, 5, 6), (1, 6, 7), (2, 2, 4)]

    sampler = Sampler.from_data(
        all_known_triples_idx=torch.tensor(train + valid + test, dtype=torch.long),
        num_entities=num_entities, num_relations=num_relations,
        device=torch.device("cpu"), min_entity_idx=0,
    )
    test_t = torch.tensor(test, dtype=torch.long)
    scorer = lambda q, p, m: kge_default_scorer(model, q, p, m)

    # Exhaustive filtered ranking
    ev_exhaustive = RankingEvaluator(
        scorer=scorer,
        candidates=SamplerCandidates(sampler, k=None),
        batch_size=8, modes=("head", "tail"),
        device=torch.device("cpu"), compile=False,
    )
    exhaustive = ev_exhaustive.evaluate(test_t).metrics()
    print("Exhaustive filtered ranking:")
    for k, v in exhaustive.items():
        print(f"  {k:>8}: {v:.4f}")

    # Sampled filtered ranking (K=10)
    ev_sampled = RankingEvaluator(
        scorer=scorer,
        candidates=SamplerCandidates(sampler, k=10),
        batch_size=8, modes=("head", "tail"), seed=42,
        device=torch.device("cpu"), compile=False,
    )
    sampled = ev_sampled.evaluate(test_t).metrics()
    print("\nSampled filtered ranking (K=10):")
    for k, v in sampled.items():
        print(f"  {k:>8}: {v:.4f}")


if __name__ == "__main__":
    main()
