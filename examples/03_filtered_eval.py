"""Example 3 — Standalone filtered-ranking evaluation.

Shows how to evaluate an already-trained KGE model on a test split
using ``kge_kernels.eval.Evaluator``. This is what every published KGE
paper computes for its MRR / Hits@1 / Hits@3 / Hits@10 table.

Both exhaustive and sampled evaluation modes are demonstrated:
  - Exhaustive (``num_corruptions=0``): rank against all entities
  - Sampled (``num_corruptions=K``): rank against K random candidates
"""
from __future__ import annotations

import torch

from kge_kernels.data import build_filter_maps
from kge_kernels.eval import Evaluator
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

    head_filter, tail_filter = build_filter_maps(train, valid, test)

    # Exhaustive filtered ranking (num_corruptions=0)
    evaluator = Evaluator(
        model, num_entities,
        head_filter=head_filter, tail_filter=tail_filter,
        device=torch.device("cpu"),
    )
    test_t = torch.tensor(test, dtype=torch.long)
    exhaustive = evaluator.evaluate(test_t)
    print("Exhaustive filtered ranking:")
    for k, v in exhaustive.items():
        print(f"  {k:>8}: {v:.4f}")

    # Sampled filtered ranking (num_corruptions=10)
    sampler = Sampler.from_data(
        all_known_triples_idx=torch.tensor(train + valid + test, dtype=torch.long),
        num_entities=num_entities, num_relations=num_relations,
        device=torch.device("cpu"),
    )
    sampled_evaluator = Evaluator(
        model, num_entities,
        num_corruptions=10,
        sampler=sampler,
        head_filter=head_filter, tail_filter=tail_filter,
        seed=42,
        device=torch.device("cpu"),
    )
    sampled = sampled_evaluator.evaluate(test_t)
    print("\nSampled filtered ranking (K=10):")
    for k, v in sampled.items():
        print(f"  {k:>8}: {v:.4f}")


if __name__ == "__main__":
    main()
