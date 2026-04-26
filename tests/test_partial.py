import torch
import torch.nn as nn

from kge_kernels.scoring import precompute_partial_scores, score_partial_atoms


class _StubModel(nn.Module):
    """Minimal model exposing the tkk score(h, r, t) dispatch contract."""

    def score_triples(self, h, r, t):
        return h.float() + 2.0 * r.float() - t.float()

    def score_all_tails(self, h, r):
        all_tails = torch.arange(4, device=h.device).unsqueeze(0).expand(h.shape[0], -1)
        return self.score_triples(
            h.unsqueeze(1).expand_as(all_tails),
            r.unsqueeze(1).expand_as(all_tails),
            all_tails,
        )

    def score_all_heads(self, r, t):
        all_heads = torch.arange(4, device=t.device).unsqueeze(0).expand(t.shape[0], -1)
        return self.score_triples(
            all_heads,
            r.unsqueeze(1).expand_as(all_heads),
            t.unsqueeze(1).expand_as(all_heads),
        )

    def score(self, h, r, t):
        if h is not None and t is not None:
            return self.score_triples(h, r, t)
        if t is None:
            return self.score_all_tails(h, r)
        return self.score_all_heads(r, t)


def test_precompute_partial_scores_matches_model_maxima():
    model = _StubModel()
    pred_remap = torch.tensor([0, 1])
    const_remap = torch.tensor([0, 1, 2])

    max_tail_score, max_head_score = precompute_partial_scores(
        model, pred_remap, const_remap, batch_chunk=2, sigmoid=False,
    )

    expected_tail = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [2.0, 3.0, 4.0],
        ]
    )
    expected_head = torch.tensor(
        [
            [3.0, 2.0, 1.0],
            [5.0, 4.0, 3.0],
        ]
    )

    assert torch.equal(max_tail_score, expected_tail)
    assert torch.equal(max_head_score, expected_head)


def test_score_partial_atoms_uses_precomputed_tables():
    max_tail_score = torch.tensor([[0.0, 1.0, 2.0]])
    max_head_score = torch.tensor([[3.0, 2.0, 1.0]])
    out = score_partial_atoms(
        preds=torch.tensor([0, 0, 0]),
        args1=torch.tensor([1, 4, 4]),
        args2=torch.tensor([4, 2, 5]),
        constant_no=3,
        max_tail_score=max_tail_score,
        max_head_score=max_head_score,
    )
    assert torch.equal(out, torch.tensor([1.0, 1.0, 0.0]))
