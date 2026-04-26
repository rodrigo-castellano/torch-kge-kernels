"""Tests for kge_kernels.eval subpackage."""

import torch
from torch import nn

from kge_kernels.eval import (
    EvalResults,
    RankingEvaluator,
    SamplerCandidates,
    rrf,
    zscore_fusion,
)
from kge_kernels.models import kge_default_scorer
from kge_kernels.scoring import Sampler


class _FakeModel(nn.Module):
    """Fake model for testing: score = sum of (h, r, t) indices."""

    def __init__(self, num_entities: int = 50):
        super().__init__()
        self.num_entities = num_entities
        self.dim = 4
        self._param = nn.Parameter(torch.zeros(1))  # need at least one param

    def score(self, h, r, t, *, d_chunk=None):
        if h is not None and t is not None:
            return h.float() + r.float() * 2.0 + t.float()
        r_f = r.float()
        if r_f.dim() == 0:
            r_f = r_f.unsqueeze(0)
        if t is None:
            all_t = torch.arange(self.num_entities, device=h.device).float()
            return h.unsqueeze(1).float() + r_f.unsqueeze(1) * 2 + all_t.unsqueeze(0)
        all_h = torch.arange(self.num_entities, device=t.device).float()
        return all_h.unsqueeze(0) + r_f.unsqueeze(1) * 2 + t.unsqueeze(1).float()


def _make_sampler(num_entities: int = 50, num_relations: int = 5) -> Sampler:
    """Create a minimal sampler for testing."""
    device = torch.device("cpu")
    triples = torch.tensor([
        [0, 1, 2], [1, 3, 4], [2, 5, 6],
    ], dtype=torch.long)
    return Sampler.from_data(
        all_known_triples_idx=triples,
        num_entities=num_entities,
        num_relations=num_relations,
        device=device,
        min_entity_idx=1,
    )


def test_rrf_seeded_reproducible():
    device = torch.device("cpu")
    K, CQ = 20, 5
    scores = {
        "mode_a": torch.ones(CQ * K),
        "mode_b": torch.ones(CQ * K) * 2,
        "mode_c": torch.ones(CQ * K) * 3,
    }
    r1 = rrf(scores, pool_k=K, n_queries=CQ, device=device, seed=42)
    r2 = rrf(scores, pool_k=K, n_queries=CQ, device=device, seed=42)
    assert torch.equal(r1["rrf"], r2["rrf"])

    r3 = rrf(scores, pool_k=K, n_queries=CQ, device=device, seed=99)
    assert not torch.equal(r1["rrf"], r3["rrf"])


def test_rrf_fair_tiebreaking():
    device = torch.device("cpu")
    K = 10
    CQ = 100
    flat = torch.ones(CQ * K)
    scores = {"mode_a": flat}
    result = rrf(scores, pool_k=K, n_queries=CQ, device=device, seed=42)
    rrf_2d = result["rrf"].view(K, CQ).t()
    pos0_best = (rrf_2d[:, 0] == rrf_2d.max(dim=1).values).float().mean()
    assert pos0_best.item() < 0.5, f"Position 0 wins too often: {pos0_best.item():.2f}"


def test_zscore_fusion():
    device = torch.device("cpu")
    K = 5
    CQ = 4
    scores = {
        "mode_a": torch.randn(CQ * K),
        "mode_b": torch.randn(CQ * K),
    }
    result = zscore_fusion(scores, pool_k=K, n_queries=CQ, device=device)
    assert "zscore" in result
    assert result["zscore"].shape == (CQ * K,)
    fused_2d = result["zscore"].view(K, CQ).t()
    per_q_mean = fused_2d.mean(dim=1)
    assert per_q_mean.abs().max().item() < 0.5


def test_evaluate_sampled_mode():
    """RankingEvaluator with sampled SamplerCandidates returns valid metrics."""
    model = _FakeModel(num_entities=20)
    sampler = _make_sampler(num_entities=20)
    queries = torch.tensor([[0, 10, 15], [1, 12, 18]], dtype=torch.long)
    candidates = SamplerCandidates(sampler, k=10)
    ev = RankingEvaluator(
        scorer=lambda q, p, m: kge_default_scorer(model, q, p, m),
        candidates=candidates,
        batch_size=2, modes=("tail",), seed=42,
        device=torch.device("cpu"), compile=False,
    )
    metrics = ev.evaluate(queries).metrics()
    assert "MRR" in metrics
    assert metrics["MRR"] > 0


def test_evaluate_exhaustive_mode():
    """RankingEvaluator with exhaustive SamplerCandidates returns valid metrics."""
    model = _FakeModel(num_entities=10)
    sampler = _make_sampler(num_entities=10)
    queries = torch.tensor([[0, 3, 7], [1, 4, 8]], dtype=torch.long)
    candidates = SamplerCandidates(sampler, k=None)
    ev = RankingEvaluator(
        scorer=lambda q, p, m: kge_default_scorer(model, q, p, m),
        candidates=candidates,
        batch_size=2, device=torch.device("cpu"), compile=False,
    )
    metrics = ev.evaluate(queries).metrics()
    assert "MRR" in metrics
    assert 0.0 <= metrics["MRR"] <= 1.0


def test_eval_results_to_dict():
    """EvalResults.to_dict returns serializable dict."""
    r = EvalResults(
        metrics={"test": {"MRR": 0.5, "Hits@1": 0.3}},
        stats={"proved": 0.8},
        config={"seed": 42},
    )
    d = r.to_dict()
    assert d["metrics"]["test"]["MRR"] == 0.5
    assert d["stats"]["proved"] == 0.8
    assert d["config"]["seed"] == 42
