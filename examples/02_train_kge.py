"""Example 2 — Standalone KGE training using ``kge_kernels.training``.

Trains a tiny TransE model on a synthesized toy KG using the lean
``train_kge`` loop, NSSA loss, cosine warmup scheduler, and filtered
ranking evaluation. No dataset files required — everything is built
in-memory so this example runs anywhere.

The training pipeline below is what you would typically write in any
consumer of ``tkk`` that needs a standalone KGE baseline: point the
dataset loader at real triple files, swap the in-memory sampler for
``kge_kernels.Sampler.from_data``, and the rest of the machinery is
reusable.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from kge_kernels.data import build_filter_maps
from kge_kernels.eval import Evaluator
from kge_kernels.losses import NSSALoss
from kge_kernels.models import TransE
from kge_kernels.training import (
    KGETrainConfig,
    TripleDataset,
    make_cosine_warmup_scheduler,
    set_seed,
    train_kge,
)


def _synth_kg(
    num_entities: int = 20,
    num_relations: int = 3,
    num_train: int = 200,
    num_test: int = 20,
    seed: int = 0,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """Synthesize a toy KG with disjoint train and test triples."""
    rng = random.Random(seed)
    all_triples = set()
    while len(all_triples) < num_train + num_test:
        r = rng.randrange(num_relations)
        h = rng.randrange(num_entities)
        t = rng.randrange(num_entities)
        if h != t:
            all_triples.add((r, h, t))
    listed = sorted(all_triples)
    rng.shuffle(listed)
    return listed[:num_train], listed[num_train : num_train + num_test]


def main() -> None:
    set_seed(0)

    num_entities, num_relations = 20, 3
    train_triples, test_triples = _synth_kg(
        num_entities=num_entities,
        num_relations=num_relations,
        num_train=200,
        num_test=20,
    )

    # Config: small dims, few epochs — this is a smoke example.
    cfg = KGETrainConfig(
        lr=5e-2,
        batch_size=32,
        epochs=3,
        neg_ratio=8,
        adv_temp=1.0,
        grad_clip=1.0,
        warmup_ratio=0.1,
        scheduler="cosine",
    )

    model = TransE(
        num_entities=num_entities, num_relations=num_relations, dim=16
    )
    dataloader = DataLoader(TripleDataset(train_triples), batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    total_steps = cfg.epochs * len(dataloader)
    scheduler = make_cosine_warmup_scheduler(optimizer, total_steps, cfg.warmup_ratio)
    loss_fn = NSSALoss(adv_temp=cfg.adv_temp, neg_ratio=cfg.neg_ratio)

    # Dumb random negative sampler for the example. Real pipelines use
    # ``kge_kernels.Sampler.from_data`` with filtered corruption.
    def sample_negatives(batch: torch.Tensor) -> torch.Tensor:
        bs = batch.shape[0]
        k = cfg.neg_ratio
        neg = batch.unsqueeze(1).repeat(1, k, 1).reshape(-1, 3)
        corrupt_head = torch.rand(bs * k) < 0.5
        rnd_entities = torch.randint(num_entities, (bs * k,))
        neg[corrupt_head, 1] = rnd_entities[corrupt_head]
        neg[~corrupt_head, 2] = rnd_entities[~corrupt_head]
        return neg

    def on_epoch_end(epoch: int, avg_loss: float, _model: torch.nn.Module) -> bool:
        print(f"  epoch {epoch}: loss={avg_loss:.4f}")
        return False  # never stop early

    print("Training...")
    losses = train_kge(
        cfg,
        model,
        dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        sample_negatives=sample_negatives,
        scheduler=scheduler,
        device=torch.device("cpu"),
        on_epoch_end=on_epoch_end,
    )
    print("Per-epoch losses: " + ", ".join(f"{loss:.4f}" for loss in losses))

    # Evaluate using filtered ranking on the test split.
    head_filter, tail_filter = build_filter_maps(train_triples, test_triples)
    evaluator = Evaluator(
        model, num_entities,
        head_filter=head_filter, tail_filter=tail_filter,
        device=torch.device("cpu"),
    )
    metrics = evaluator.evaluate(torch.tensor(test_triples, dtype=torch.long))
    print("Test metrics (exhaustive filtered ranking):")
    for k, v in metrics.items():
        print(f"  {k:>8}: {v:.4f}")


if __name__ == "__main__":
    main()
