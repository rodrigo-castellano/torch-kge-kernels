"""Example 3 — Standalone filtered-ranking evaluation.

Shows how to evaluate an already-trained KGE model on a test split
using ``kge_kernels.eval.evaluate_filtered_ranking``. This is what
every published KGE paper computes for its MRR / Hits@1 / Hits@3 /
Hits@10 table.

Use this pattern in a ``python examples/03_filtered_eval.py`` style
script when you want to eval a checkpoint end-to-end without setting
up a training loop.
"""
from __future__ import annotations

import torch

from kge_kernels.data import build_filter_maps
from kge_kernels.eval import evaluate_filtered_ranking
from kge_kernels.models import TransE


def main() -> None:
    torch.manual_seed(0)

    num_entities = 15
    num_relations = 3

    # Random TransE — in real usage you would load a trained checkpoint
    # via ``kge_kernels.checkpoints.load_checkpoint`` and then
    # ``model.load_state_dict(state_dict)``.
    model = TransE(
        num_entities=num_entities, num_relations=num_relations, dim=16
    )

    # Toy train / valid / test triples. Real pipelines load these via
    # ``kge_kernels.data.load_triples_with_mappings``.
    train = [
        (0, 0, 1), (0, 1, 2), (0, 2, 3), (0, 3, 4),
        (1, 0, 5), (1, 2, 6), (1, 4, 7),
        (2, 1, 8), (2, 3, 9),
    ]
    valid = [(0, 4, 5), (1, 1, 3)]
    test = [(0, 5, 6), (1, 6, 7), (2, 2, 4)]

    # Build filter maps over ALL known positives so that filtered ranking
    # ignores entities that are provably correct for other (h, r) or
    # (r, t) pairs in the KG.
    head_filter, tail_filter = build_filter_maps(train, valid, test)

    # Exhaustive filtered ranking: ranks the true entity against every
    # other entity in the vocabulary, filtering out known positives.
    exhaustive = evaluate_filtered_ranking(
        model,
        triples=test,
        num_entities=num_entities,
        head_filter=head_filter,
        tail_filter=tail_filter,
        device=torch.device("cpu"),
    )
    print("Exhaustive filtered ranking:")
    for k, v in exhaustive.items():
        print(f"  {k:>8}: {v:.4f}")

    # Sampled filtered ranking: ranks the true entity against K random
    # (filtered) candidates. Faster, lower variance per query.
    sampled = evaluate_filtered_ranking(
        model,
        triples=test,
        num_entities=num_entities,
        head_filter=head_filter,
        tail_filter=tail_filter,
        device=torch.device("cpu"),
        eval_num_corruptions=10,
        seed=42,
    )
    print("\nSampled filtered ranking (K=10):")
    for k, v in sampled.items():
        print(f"  {k:>8}: {v:.4f}")


if __name__ == "__main__":
    main()
