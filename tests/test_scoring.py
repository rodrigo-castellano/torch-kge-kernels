import torch

from kge_kernels.sampler import Sampler
from kge_kernels.scoring import score
from kge_kernels.types import KGEBackend


def _make_backend() -> KGEBackend:
    def score_triples(h, r, t):
        return h.float() + 2.0 * r.float() - t.float()

    def score_all_tails(h, r):
        all_tails = torch.arange(4, device=h.device).unsqueeze(0).expand(h.shape[0], -1)
        return score_triples(h.unsqueeze(1).expand_as(all_tails), r.unsqueeze(1).expand_as(all_tails), all_tails)

    def score_all_heads(r, t):
        all_heads = torch.arange(4, device=t.device).unsqueeze(0).expand(t.shape[0], -1)
        return score_triples(all_heads, r.unsqueeze(1).expand_as(all_heads), t.unsqueeze(1).expand_as(all_heads))

    return KGEBackend(
        score_triples=score_triples,
        score_all_tails=score_all_tails,
        score_all_heads=score_all_heads,
    )


def test_score_triples_calls_explicit_backend():
    backend = _make_backend()
    out = score(
        backend,
        torch.tensor([[2, 1, 3]]),
        mode="triples",
    ).scores
    assert torch.equal(out, torch.tensor([2.0]))


def test_score_all_tails_calls_batched_backend():
    backend = _make_backend()
    out = score(
        backend,
        torch.tensor([[2, 1, 0]]),
        mode="tail",
    ).scores
    assert torch.equal(out, torch.tensor([[5.0, 4.0, 3.0, 2.0]]))


def test_score_all_heads_calls_batched_backend():
    backend = _make_backend()
    out = score(
        backend,
        torch.tensor([[2, 0, 3]]),
        mode="head",
    ).scores
    assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))


def test_score_sampled_tails_returns_masked_batch():
    backend = _make_backend()
    sampler = Sampler.from_data(
        all_known_triples_idx=torch.tensor([[2, 1, 3]], dtype=torch.long),
        num_entities=4,
        num_relations=3,
        device=torch.device("cpu"),
        min_entity_idx=0,
    )

    torch.manual_seed(0)
    out = score(
        backend,
        torch.tensor([[2, 1, 3]], dtype=torch.long),
        mode="tail",
        num_corruptions=2,
        sampler=sampler,
    )

    assert out.scores.shape == (1, 3)
    assert out.valid_mask is not None
    assert out.valid_mask.shape == (1, 2)
