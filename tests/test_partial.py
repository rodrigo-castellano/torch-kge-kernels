import torch
import torch.nn as nn

from kge_kernels.scoring import LazyPartialScorer, PartialScorer


class _StubModel(nn.Module):
    """Minimal model exposing the tkk score(h, r, t) contract."""

    NUM_E = 4

    def score(self, h, r, t, *, d_chunk=None):
        if h is not None and t is not None:
            return h.float() + 2.0 * r.float() - t.float()
        if t is None:
            all_t = torch.arange(self.NUM_E, device=h.device).unsqueeze(0).expand(h.shape[0], -1)
            return self.score(
                h.unsqueeze(1).expand_as(all_t),
                r.unsqueeze(1).expand_as(all_t),
                all_t,
            )
        all_h = torch.arange(self.NUM_E, device=t.device).unsqueeze(0).expand(t.shape[0], -1)
        return self.score(
            all_h,
            r.unsqueeze(1).expand_as(all_h),
            t.unsqueeze(1).expand_as(all_h),
        )


def test_partial_scorer_compute_all_matches_model_maxima():
    model = _StubModel()
    pred_remap = torch.tensor([0, 1])
    const_remap = torch.tensor([0, 1, 2])

    scorer = PartialScorer(
        model, pred_remap, const_remap, constant_no=3, sigmoid=False,
    ).compute_all(batch_chunk=2)

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

    assert torch.equal(scorer.max_tail_score, expected_tail)
    assert torch.equal(scorer.max_head_score, expected_head)


def test_score_atoms_uses_precomputed_tables():
    """``PartialScorer.score_atoms`` reads the lookup tables directly."""
    scorer = PartialScorer.__new__(PartialScorer)
    scorer.max_tail_score = torch.tensor([[0.0, 1.0, 2.0]])
    scorer.max_head_score = torch.tensor([[3.0, 2.0, 1.0]])
    scorer.constant_no = 3
    out = scorer.score_atoms(
        preds=torch.tensor([0, 0, 0]),
        args1=torch.tensor([1, 4, 4]),
        args2=torch.tensor([4, 2, 5]),
    )
    assert torch.equal(out, torch.tensor([1.0, 1.0, 0.0]))


def test_lazy_scorer_inherits_score_atoms():
    """LazyPartialScorer is-a PartialScorer (shares score_atoms)."""
    model = _StubModel()
    pred_remap = torch.tensor([0, 1])
    const_remap = torch.tensor([0, 1, 2])
    lazy = LazyPartialScorer(
        model, pred_remap, const_remap,
        constant_no=3, padding_idx=0, true_pred_idx=-1, false_pred_idx=-1,
        sigmoid=False,
    )
    assert isinstance(lazy, PartialScorer)
    # Tables start zero; score_atoms reads zeros (no fill yet).
    out = lazy.score_atoms(
        preds=torch.tensor([0]),
        args1=torch.tensor([1]),  # bound (≤ constant_no)
        args2=torch.tensor([5]),  # unbound (> constant_no)
    )
    assert out.item() == 0.0
