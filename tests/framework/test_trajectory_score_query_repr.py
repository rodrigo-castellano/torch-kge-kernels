"""A/B parity: TrajectoryScoreQueryRepr matches the two legacy dispatchers.

For every mode in ``ALL_TRAJECTORY_SCORE_MODES``, run the unified
class and the corresponding legacy class on the same inputs; assert
exact-equal outputs.

Mode-name collision resolution:
  - ProofScoreQueryRepr.depth_weighted   ↔ unified "depth_weighted_quotient"
  - PolicyRolloutQueryRepr.depth_weighted ↔ unified "depth_weighted_bonus"
"""
from __future__ import annotations

import pytest
import torch

from kge_kernels.framework import (
    ALL_TRAJECTORY_SCORE_MODES,
    PolicyRolloutQueryRepr,
    ProofScoreQueryRepr,
    Repr,
    TrajectoryScoreQueryRepr,
)


# Legacy proof-score modes (from ProofScoreQueryRepr's three frozensets).
_PROOF_TRAJ = {
    "no_cliff", "no_cliff_sqrt", "no_cliff_log", "best_avg",
    "max_intermediate_nc", "trajectory_range", "raw_cumulative",
    "geometric_mean",
}
_PROOF_BINARY = {
    "proof_logprob", "leaf_only", "depth_weighted", "final_state",
    "min_step", "best_partial", "sbr", "max_intermediate",
}
_PROOF_RL_VALUE = {
    "rl_value", "rl_value_mean", "rl_value_neg", "rl_value_mean_neg",
}
_LEGACY_PROOF_MODES = _PROOF_TRAJ | _PROOF_BINARY | _PROOF_RL_VALUE


# Legacy policy-rollout modes.
_LEGACY_ROLLOUT_MODES = {
    # _RL_MODES
    "logprob", "logprob_endf", "proof_binary", "proof_bonus",
    "depth_weighted", "logprob_clipped", "proof_rank", "success_only",
    "logprob_scaled_penalty",
    # _STEP0_MODES
    "value_pos", "end_prob", "hybrid_end", "hybrid_value",
    "combined", "value_diff", "endf_kge_embed",
    # _KGE_EMBED_MODES
    "kge_embed", "kge_embed_hybrid", "logprob_kge_embed",
}


# Map unified mode name → (legacy class, legacy mode name).
_RENAMED = {
    "depth_weighted_quotient": ("proof", "depth_weighted"),
    "depth_weighted_bonus":    ("rollout", "depth_weighted"),
}


def _legacy_for(mode: str):
    """Resolve unified mode → ("proof"|"rollout", legacy_mode_name)."""
    if mode in _RENAMED:
        return _RENAMED[mode]
    if mode in _LEGACY_ROLLOUT_MODES and mode in _LEGACY_PROOF_MODES:
        # No collisions remain after the rename above.
        raise AssertionError(f"unexpected collision for {mode}")
    if mode in _LEGACY_PROOF_MODES:
        return ("proof", mode)
    if mode in _LEGACY_ROLLOUT_MODES:
        return ("rollout", mode)
    raise AssertionError(f"mode {mode!r} not found in either legacy registry")


def _build_summaries(seed: int = 0):
    """Synthetic trajectory summaries with all standard keys populated."""
    g = torch.Generator().manual_seed(seed)
    B = 8
    success = torch.tensor([True, False, True, True, False, True, False, True])
    depths = torch.tensor([3, 5, 1, 4, 2, 7, 6, 1], dtype=torch.long)
    # log-prob-style scalars in roughly [-10, 0]
    cumulative_log = torch.empty(B).uniform_(-10.0, 0.0, generator=g)
    min_step_log   = torch.empty(B).uniform_(-3.0,  -0.1, generator=g)
    final_step_log = torch.empty(B).uniform_(-3.0,  -0.1, generator=g)
    best_cumulative = cumulative_log + torch.empty(B).uniform_(0.0, 1.0, generator=g)
    best_prefix_avg = cumulative_log / depths.float().clamp(min=1.0)
    sbr_body_min   = torch.empty(B).uniform_(0.0, 1.0, generator=g)
    final_state_score     = torch.empty(B).uniform_(0.0, 1.0, generator=g)
    best_ever_state_score = torch.empty(B).uniform_(0.0, 1.0, generator=g)
    # rollout-specific
    endf  = torch.tensor([True, True, False, True, False, False, True, False])
    p_end = torch.empty(B).uniform_(0.0, 0.99, generator=g)
    v_pos = torch.empty(B).uniform_(-1.0, 1.0, generator=g)
    v_neg = torch.empty(B).uniform_(-1.0, 1.0, generator=g)
    kge_embed = torch.empty(B).uniform_(-2.0, 2.0, generator=g)
    # RL-value (proof) signals
    cum_value = torch.empty(B).uniform_(-5.0, 5.0, generator=g)
    value_steps = torch.tensor([2, 3, 1, 2, 1, 4, 3, 1], dtype=torch.long)
    return {
        "success": success, "depths": depths,
        "cumulative_log": cumulative_log,
        "min_step_log": min_step_log,
        "best_cumulative": best_cumulative,
        "sbr_body_min": sbr_body_min,
        "final_step_log": final_step_log,
        "final_state_score": final_state_score,
        "best_ever_state_score": best_ever_state_score,
        "best_prefix_avg": best_prefix_avg,
        "endf": endf, "p_end": p_end, "v_pos": v_pos, "v_neg": v_neg,
        "kge_embed": kge_embed,
        "cum_value": cum_value, "value_steps": value_steps,
    }


def _legacy_proof_accums(d):
    """Map standard summary keys → ProofScoreQueryRepr's accums dict."""
    return {
        "cumulative_log_scores": d["cumulative_log"],
        "min_step_log_scores":   d["min_step_log"],
        "best_cumulative":       d["best_cumulative"],
        "sbr_body_min":          d["sbr_body_min"],
        "final_step_log_scores": d["final_step_log"],
        "final_state_scores":    d["final_state_score"],
        "best_ever_state_score": d["best_ever_state_score"],
        "best_prefix_avg":       d["best_prefix_avg"],
    }


@pytest.mark.parametrize("mode", sorted(ALL_TRAJECTORY_SCORE_MODES))
def test_unified_matches_legacy(mode):
    d = _build_summaries(seed=42)
    cfg_kwargs = dict(alpha=5.0, beta=2.0, fail_penalty=1000.0,
                       partial_weight=0.5, endf_penalty=200.0)

    new_qr = TrajectoryScoreQueryRepr(mode, **cfg_kwargs)
    new_out = new_qr(Repr(summaries=d), evidence=None).scores

    family, legacy_mode = _legacy_for(mode)
    if family == "proof":
        old_qr = ProofScoreQueryRepr(
            legacy_mode, fail_penalty=cfg_kwargs["fail_penalty"],
            partial_weight=cfg_kwargs["partial_weight"],
        )
        kwargs = {}
        if legacy_mode in _PROOF_RL_VALUE:
            kwargs = dict(cum_value=d["cum_value"], value_steps=d["value_steps"])
        old_out = old_qr(_legacy_proof_accums(d), d["success"], d["depths"],
                          **kwargs)
    else:  # rollout
        rollout_kwargs = {k: v for k, v in cfg_kwargs.items() if k != "partial_weight"}
        old_qr = PolicyRolloutQueryRepr(legacy_mode, **rollout_kwargs)
        old_out = old_qr(
            success=d["success"], logprobs=d["cumulative_log"],
            endf=d["endf"], depths=d["depths"],
            p_end=d["p_end"], v_pos=d["v_pos"], v_neg=d["v_neg"],
            kge_embed=d["kge_embed"],
        )

    assert torch.allclose(new_out, old_out, atol=0.0), (
        f"mode={mode}\n  new: {new_out.tolist()}\n  old: {old_out.tolist()}"
    )


def test_repr_summaries_alone_is_valid():
    """A Repr with only summaries (no embeddings/scores) must construct."""
    r = Repr(summaries={"cumulative_log": torch.zeros(4),
                         "success": torch.ones(4, dtype=torch.bool),
                         "depths": torch.ones(4, dtype=torch.long)})
    assert r.has_summaries
    assert not r.has_embeddings
    assert not r.has_scores


def test_unified_missing_required_key_raises():
    """Modes that read a key not in summaries must raise loudly."""
    summaries = {"success": torch.tensor([True, False])}
    qr = TrajectoryScoreQueryRepr("logprob")
    with pytest.raises(KeyError):
        qr(Repr(summaries=summaries), evidence=None)


def test_unified_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown trajectory-scoring mode"):
        TrajectoryScoreQueryRepr("does_not_exist")
