import torch

from kge_kernels.scoring import kge_score_triples


class RawModel:
    kge_output_mode = "raw"

    def score_triples(self, h, r, t):
        return h.float() + r.float() - t.float()


class SigmoidModel:
    kge_output_mode = "sigmoid"

    def score_atoms(self, preds, subjs, objs):
        del preds, objs
        return subjs.float()


def test_kge_score_triples_normalizes_raw_models():
    model = RawModel()
    out = kge_score_triples(
        model,
        torch.tensor([1]),
        torch.tensor([2]),
        torch.tensor([3]),
    )
    assert torch.allclose(out, torch.sigmoid(torch.tensor([0.0])))


def test_kge_score_triples_respects_sigmoid_models():
    model = SigmoidModel()
    out = kge_score_triples(
        model,
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([2]),
    )
    assert torch.equal(out, torch.tensor([1.0]))
