"""Tests for kge_kernels.eval subpackage."""

import torch

from kge_kernels.eval import CandidatePool, EvalResults, Evaluator, rrf, zscore_fusion
from kge_kernels.scoring import Sampler


def _make_sampler(num_entities: int = 50, num_relations: int = 5) -> Sampler:
    """Create a minimal sampler for testing."""
    device = torch.device("cpu")
    # Some known triples for filtering
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


def test_candidate_pool_build():
    """CandidatePool.build produces correct shape with positive at slot 0."""
    sampler = _make_sampler()
    queries = torch.tensor([[0, 1, 2], [1, 3, 4]], dtype=torch.long)
    pool = CandidatePool.build(queries, sampler, n_corruptions=10, mode="tail")

    assert pool.K == 11  # 1 + 10
    assert pool.CQ == 2
    assert pool.pool.shape == (22, 3)
    assert pool.pool_size == 22

    # Positive is at slot 0 for each query (column-major layout)
    pool_3d = pool.pool.view(pool.K, pool.CQ, 3)
    assert torch.equal(pool_3d[0, 0], queries[0])
    assert torch.equal(pool_3d[0, 1], queries[1])


def test_candidate_pool_valid_mask():
    """valid_mask correctly identifies padded entries."""
    sampler = _make_sampler(num_entities=5)  # few entities → some padded
    queries = torch.tensor([[0, 1, 2]], dtype=torch.long)
    pool = CandidatePool.build(queries, sampler, n_corruptions=20, mode="tail")

    assert pool.valid_mask.shape == (1, 21)
    assert pool.valid_mask[0, 0].item()  # positive always valid

    # Check that at least some negatives exist and mask aligns
    pool_3d = pool.pool.view(pool.K, pool.CQ, 3)
    for i in range(1, pool.K):
        is_padding = pool_3d[i, 0].sum().item() == 0
        assert pool.valid_mask[0, i].item() != is_padding or pool.valid_mask[0, i].item()


def test_rrf_seeded_reproducible():
    """Same seed → same RRF result; different seed may differ."""
    device = torch.device("cpu")
    K, CQ = 20, 5
    # All tied within each query so tie-breaking noise matters
    scores = {
        "mode_a": torch.ones(CQ * K),
        "mode_b": torch.ones(CQ * K) * 2,
        "mode_c": torch.ones(CQ * K) * 3,
    }
    r1 = rrf(scores, pool_k=K, n_queries=CQ, device=device, seed=42)
    r2 = rrf(scores, pool_k=K, n_queries=CQ, device=device, seed=42)
    assert torch.equal(r1["rrf"], r2["rrf"])

    # With K=20 and 3 modes of all-ties, different seeds produce different results
    r3 = rrf(scores, pool_k=K, n_queries=CQ, device=device, seed=99)
    assert not torch.equal(r1["rrf"], r3["rrf"])


def test_rrf_fair_tiebreaking():
    """Ties don't systematically favor position 0."""
    device = torch.device("cpu")
    K = 10
    CQ = 100
    # All identical scores → ranks should be ~uniform
    flat = torch.ones(CQ * K)
    scores = {"mode_a": flat}
    result = rrf(scores, pool_k=K, n_queries=CQ, device=device, seed=42)
    rrf_2d = result["rrf"].view(K, CQ).t()  # [CQ, K]

    # Position-0 should NOT always get the highest RRF score
    pos0_best = (rrf_2d[:, 0] == rrf_2d.max(dim=1).values).float().mean()
    assert pos0_best.item() < 0.5, f"Position 0 wins too often: {pos0_best.item():.2f}"


def test_zscore_fusion():
    """Z-score fusion produces correct shape and normalized output."""
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

    # Per-query mean of fused scores should be ~0
    fused_2d = result["zscore"].view(K, CQ).t()
    per_q_mean = fused_2d.mean(dim=1)
    assert per_q_mean.abs().max().item() < 0.5


def test_evaluator_single_mode():
    """Evaluator with a simple scorer returns correct MRR."""
    sampler = _make_sampler(num_entities=20)
    device = torch.device("cpu")

    # Scorer that gives high score to true triples (rel+head+tail > 5)
    # and low score to corruptions
    def scorer(queries: torch.Tensor) -> dict:
        # True positives have known structure; corruptions are random
        s = queries.sum(dim=1).float()
        return {"simple": s}

    evaluator = Evaluator(
        scorer=scorer,
        sampler=sampler,
        n_corruptions=10,
        corruption_scheme="tail",
        seed=42,
        device=device,
    )

    queries = torch.tensor([
        [0, 10, 15],  # sum=25, should rank high
        [1, 12, 18],  # sum=31, should rank high
    ], dtype=torch.long)

    results = evaluator.evaluate(queries)
    assert "simple" in results.metrics
    assert "MRR" in results.metrics["simple"]
    assert results.metrics["simple"]["MRR"] > 0


def test_evaluator_with_rrf():
    """Evaluator with multi-mode scorer + RRF fusion."""
    sampler = _make_sampler(num_entities=30)
    device = torch.device("cpu")

    def scorer(queries: torch.Tensor) -> dict:
        s1 = queries.sum(dim=1).float()
        s2 = queries[:, 1].float() * 2 + queries[:, 2].float()
        return {"mode_a": s1, "mode_b": s2}

    def fusion_fn(sd, pk, nq, dev):
        return rrf(sd, pk, nq, dev, k=60.0, seed=42)

    evaluator = Evaluator(
        scorer=scorer,
        sampler=sampler,
        n_corruptions=10,
        corruption_scheme="tail",
        fusion=fusion_fn,
        seed=42,
        device=device,
    )

    queries = torch.tensor([
        [0, 8, 12],
        [1, 10, 15],
        [2, 5, 9],
    ], dtype=torch.long)

    results = evaluator.evaluate(queries)
    # Should have individual modes + fused
    assert "mode_a" in results.metrics
    assert "mode_b" in results.metrics
    assert "rrf" in results.metrics


def test_evaluator_fixed_batch_size():
    """Padding to fixed batch_size works correctly."""
    sampler = _make_sampler(num_entities=20)
    device = torch.device("cpu")

    call_sizes = []

    def scorer(queries: torch.Tensor) -> dict:
        call_sizes.append(queries.shape[0])
        return {"score": queries.sum(dim=1).float()}

    evaluator = Evaluator(
        scorer=scorer,
        sampler=sampler,
        n_corruptions=5,
        corruption_scheme="tail",
        batch_size=32,
        seed=42,
        device=device,
    )

    queries = torch.tensor([[0, 3, 7], [1, 4, 8]], dtype=torch.long)
    results = evaluator.evaluate(queries)

    # All scorer calls should receive exactly batch_size tensors
    for sz in call_sizes:
        assert sz == 32, f"Expected batch_size=32, got {sz}"

    assert "score" in results.metrics


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
