"""Tests for kge_kernels.training: config, dataset, set_seed, scheduler,
model wrapping, and the train_kge loop."""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from kge_kernels.losses import NSSALoss
from kge_kernels.models import TransE
from kge_kernels.training import (
    KGETrainConfig,
    TripleDataset,
    make_cosine_warmup_scheduler,
    set_seed,
    train_kge,
    wrap_model_for_training,
)


def _toy_triples(n: int = 32, num_entities: int = 7, num_relations: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    triples = []
    for _ in range(n):
        r = int(torch.randint(num_relations, (1,), generator=g))
        h = int(torch.randint(num_entities, (1,), generator=g))
        t = int(torch.randint(num_entities, (1,), generator=g))
        triples.append((r, h, t))
    return triples


# ═══════════════════════════════════════════════════════════════════════
# KGETrainConfig defaults
# ═══════════════════════════════════════════════════════════════════════


def test_kge_train_config_defaults():
    cfg = KGETrainConfig()
    assert cfg.lr == 1e-3
    assert cfg.batch_size == 4096
    assert cfg.epochs == 5
    assert cfg.scheduler == "cosine"
    assert cfg.compile is False
    assert cfg.cpu is False


# ═══════════════════════════════════════════════════════════════════════
# TripleDataset
# ═══════════════════════════════════════════════════════════════════════


def test_triple_dataset_stores_and_indexes():
    triples = [(0, 1, 2), (1, 3, 4), (2, 5, 6)]
    ds = TripleDataset(triples)
    assert len(ds) == 3
    assert torch.equal(ds[0], torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(ds[2], torch.tensor([2, 5, 6], dtype=torch.long))


def test_triple_dataset_dataloader_batching():
    triples = _toy_triples(n=16)
    ds = TripleDataset(triples)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    for batch in loader:
        assert batch.shape == (4, 3)
        break


# ═══════════════════════════════════════════════════════════════════════
# set_seed
# ═══════════════════════════════════════════════════════════════════════


def test_set_seed_is_reproducible():
    set_seed(42)
    a = torch.randn(4)
    set_seed(42)
    b = torch.randn(4)
    assert torch.equal(a, b)


# ═══════════════════════════════════════════════════════════════════════
# make_cosine_warmup_scheduler
# ═══════════════════════════════════════════════════════════════════════


def test_cosine_warmup_schedule_shape():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sched = make_cosine_warmup_scheduler(opt, total_steps=10, warmup_ratio=0.2)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    # Warmup: steps 0..1 should have lr < 1.0 (fraction-of-warmup based)
    assert lrs[0] < 1.0
    assert lrs[1] >= lrs[0]
    # Peak near end of warmup (step 2 = first decay step ≈ 1.0)
    assert max(lrs) == pytest.approx(1.0, abs=1e-3)
    # Cosine decay: final lr should be near 0
    assert lrs[-1] < 0.1


def test_cosine_warmup_zero_warmup():
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sched = make_cosine_warmup_scheduler(opt, total_steps=4, warmup_ratio=0.0)
    # At step 0 with zero warmup, lr should already be at cosine(0)*1.0 = 1.0
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=1e-3)
    for _ in range(4):
        opt.step()
        sched.step()
    # After full schedule, lr should be ~0
    assert opt.param_groups[0]["lr"] < 0.05


# ═══════════════════════════════════════════════════════════════════════
# wrap_model_for_training
# ═══════════════════════════════════════════════════════════════════════


def test_wrap_model_no_compile_no_multigpu_is_identity_modulo_device():
    cfg = KGETrainConfig(compile=False, multi_gpu=False)
    model = TransE(num_entities=5, num_relations=3, dim=4)
    wrapped = wrap_model_for_training(model, torch.device("cpu"), cfg)
    assert wrapped is model  # no wrapping
    # and on CPU
    assert next(wrapped.parameters()).device.type == "cpu"


# ═══════════════════════════════════════════════════════════════════════
# train_kge
# ═══════════════════════════════════════════════════════════════════════


def test_train_kge_runs_and_reduces_loss():
    """Sanity check: train_kge runs a couple of epochs on a toy dataset
    and the loss should not increase.
    """
    set_seed(0)
    num_entities, num_relations = 7, 3
    triples = _toy_triples(n=64, num_entities=num_entities, num_relations=num_relations)
    ds = TripleDataset(triples)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = TransE(num_entities=num_entities, num_relations=num_relations, dim=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss = NSSALoss(adv_temp=0.0, neg_ratio=4)

    # Simple "negative sampler": random entity corruption
    def sample_negatives(batch):
        bs = batch.shape[0]
        k = 4
        neg = batch.unsqueeze(1).repeat(1, k, 1).reshape(-1, 3)
        # Corrupt the head with a random entity
        neg[:, 1] = torch.randint(num_entities, (bs * k,))
        return neg

    cfg = KGETrainConfig(epochs=3, batch_size=16, neg_ratio=4, grad_clip=0.0)
    losses = train_kge(
        cfg,
        model,
        loader,
        optimizer=opt,
        loss_fn=loss,
        sample_negatives=sample_negatives,
    )
    assert len(losses) == 3
    # Not strict monotonic, but the last epoch should be no worse than the first by much
    assert losses[-1] <= losses[0] * 1.5


def test_train_kge_early_stop_via_callback():
    set_seed(0)
    num_entities, num_relations = 5, 2
    triples = _toy_triples(n=32, num_entities=num_entities, num_relations=num_relations)
    ds = TripleDataset(triples)
    loader = DataLoader(ds, batch_size=8)

    model = TransE(num_entities=num_entities, num_relations=num_relations, dim=4)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss = NSSALoss(adv_temp=0.0, neg_ratio=1)

    def sample_negatives(batch):
        bs = batch.shape[0]
        neg = batch.clone()
        neg[:, 1] = torch.randint(num_entities, (bs,))
        return neg

    epochs_seen = []

    def on_epoch_end(epoch, avg_loss, _model):
        epochs_seen.append(epoch)
        return epoch == 2  # stop after epoch 2

    cfg = KGETrainConfig(epochs=10, batch_size=8, neg_ratio=1)
    losses = train_kge(
        cfg,
        model,
        loader,
        optimizer=opt,
        loss_fn=loss,
        sample_negatives=sample_negatives,
        on_epoch_end=on_epoch_end,
    )
    assert epochs_seen == [1, 2]
    assert len(losses) == 2
