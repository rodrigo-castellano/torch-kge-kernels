"""Pre-embedded KGE scoring primitives.

These functions operate on already-looked-up embedding tensors rather
than on entity/relation indices. They exist so two different interfaces
can share the arithmetic:

* tkk's index-based models (``ComplEx``, ``RotatE``, ``RotatENS``,
  ``TransE``, ``DistMult``) do the ``nn.Embedding`` lookup and then
  call these ops.
* DpRL's pre-embedded atom embedders (``kge_experiments/nn/atom_embedders.py``)
  receive embeddings directly from the policy's embedder and call the
  same ops.

Keeping the math in one place means a fix in e.g. the RotatE L2
formula lands in every caller at once, instead of silently drifting
between the two repos.

Embedding layout convention (matches torch-ns / torch-kge-kernels):
    Complex-valued entity/relation vectors are stored as a single
    ``[..., 2H]`` tensor with the real half in ``[:H]`` and the
    imaginary half in ``[H:]``. Use :func:`complex_split` to separate.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

# ═══════════════════════════════════════════════════════════════════════
# Complex tensor helpers
# ═══════════════════════════════════════════════════════════════════════


def complex_split(z: Tensor) -> Tuple[Tensor, Tensor]:
    """Split ``[..., 2H]`` into real and imaginary halves ``(z_re, z_im)``.

    Both returned tensors are ``[..., H]`` and share storage with the
    input (views, no copy).
    """
    half = z.shape[-1] // 2
    return z[..., :half], z[..., half:]


def complex_dist(
    diff_re: Tensor, diff_im: Tensor, *, p: int = 2, eps: float = 1e-9,
) -> Tensor:
    """Lp distance between two complex vectors, summed over the last dim.

    For ``p=1``: ``Σ (|diff_re| + |diff_im|)`` — additive L1.
    For ``p=2``: ``sqrt(clamp_min(Σ (diff_re² + diff_im²), eps))`` —
    Euclidean, matching torch-ns's ns-aligned RotatE convention.

    Returns a scalar-per-sample tensor of shape ``[...]``.
    """
    if p == 1:
        return (diff_re.abs() + diff_im.abs()).sum(dim=-1)
    return torch.sqrt(
        torch.clamp((diff_re * diff_re + diff_im * diff_im).sum(dim=-1), min=eps)
    )


def complex_modulus_per_dim(
    diff_re: Tensor, diff_im: Tensor, *, eps: float = 1e-9,
) -> Tensor:
    """Per-dim complex modulus ``sqrt(re² + im² + eps)``.

    Unlike :func:`complex_dist` this does *not* sum over the last axis —
    it returns a ``[..., H]`` tensor, one modulus per complex component.
    Used by DpRL's atom embedders which keep the vector as an atom
    feature for downstream state aggregation.

    ``eps`` is added inside the sqrt (not via ``clamp``) so backward
    graphs stay simple — clamp-style ``ge+where`` trips up some fused
    Triton kernels.
    """
    return torch.sqrt(diff_re * diff_re + diff_im * diff_im + eps)


# ═══════════════════════════════════════════════════════════════════════
# RotatE primitives
# ═══════════════════════════════════════════════════════════════════════


def rotate_apply(
    h_emb: Tensor, r_phase: Tensor, *, norm_factor: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Rotate ``h`` by the phase ``r`` in complex space.

    Args:
        h_emb: ``[..., 2H]`` head embedding, ``[re | im]`` interleaved.
        r_phase: ``[..., H]`` raw relation phase values.
        norm_factor: multiplier applied to ``r_phase`` before
            ``cos/sin``. Use ``π / embedding_range`` for the ns-aligned
            RotatE (``RotatENS``) variant, ``1.0`` if ``r_phase`` is
            already in radians.

    Returns:
        ``(hr_re, hr_im)`` with shape ``[..., H]`` each.
    """
    h_re, h_im = complex_split(h_emb)
    phase = r_phase * norm_factor
    cos_p = torch.cos(phase)
    sin_p = torch.sin(phase)
    return h_re * cos_p - h_im * sin_p, h_re * sin_p + h_im * cos_p


def rotate_conj_apply_tail(
    r_phase: Tensor, t_emb: Tensor, *, norm_factor: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Compute ``conj(r) * t`` — the "rotate tail by inverse phase" form.

    ``score = |r * h - t|`` (tkk's canonical form) is numerically
    equivalent to ``|h - conj(r) * t|`` (DpRL's atom-embedder form),
    since relations live on the unit circle. Callers working in the
    second form — typically pre-embedded atom embedders that keep a
    per-dim residual as an atom feature — use this function.

    Args:
        r_phase: ``[..., H]`` raw phase values.
        t_emb: ``[..., 2H]`` tail embedding.
        norm_factor: see :func:`rotate_apply`.

    Returns:
        ``(v_re, v_im)`` each of shape ``[..., H]``, the ``conj(r) * t``
        components.
    """
    t_re, t_im = complex_split(t_emb)
    phase = r_phase * norm_factor
    cos_p = torch.cos(phase)
    sin_p = torch.sin(phase)
    # conj(r) * t = (cos - i*sin) * (t_re + i*t_im)
    #             = (cos*t_re + sin*t_im) + i*(cos*t_im - sin*t_re)
    return cos_p * t_re + sin_p * t_im, cos_p * t_im - sin_p * t_re


# ═══════════════════════════════════════════════════════════════════════
# ComplEx primitives
# ═══════════════════════════════════════════════════════════════════════


def complex_hermitian_real_vec(
    h_emb: Tensor, r_emb: Tensor, t_emb: Tensor,
) -> Tensor:
    """Per-dim real part of ``h ⊙ r ⊙ conj(t)``.

    Returns a ``[..., H]`` tensor. Sum over the last axis to get the
    ComplEx score; keep the vector for DpRL's atom-feature usage.

    Derivation (with ``h = h_re + i h_im``, ``r = r_re + i r_im``,
    ``t = t_re + i t_im``, ``conj(t) = t_re - i t_im``):
        Re(h*r*conj(t)) = h_re*r_re*t_re + h_im*r_re*t_im
                        + h_re*r_im*t_im - h_im*r_im*t_re
    """
    h_re, h_im = complex_split(h_emb)
    r_re, r_im = complex_split(r_emb)
    t_re, t_im = complex_split(t_emb)
    return (
        h_re * r_re * t_re
        + h_im * r_re * t_im
        + h_re * r_im * t_im
        - h_im * r_im * t_re
    )


def complex_hermitian_imag_vec(
    h_emb: Tensor, r_emb: Tensor, t_emb: Tensor,
) -> Tensor:
    """Per-dim imaginary part of ``h ⊙ r ⊙ conj(t)``.

    Used by :meth:`ComplEx.compose` which concatenates real and
    imaginary halves into a single feature vector.

    Im(h*r*conj(t)) = h_re*r_re*t_im - h_im*r_re*t_re
                    + h_re*r_im*t_re + h_im*r_im*t_im
    """
    h_re, h_im = complex_split(h_emb)
    r_re, r_im = complex_split(r_emb)
    t_re, t_im = complex_split(t_emb)
    return (
        h_re * r_re * t_im
        - h_im * r_re * t_re
        + h_re * r_im * t_re
        + h_im * r_im * t_im
    )


__all__ = [
    "complex_dist",
    "complex_hermitian_imag_vec",
    "complex_hermitian_real_vec",
    "complex_modulus_per_dim",
    "complex_split",
    "rotate_apply",
    "rotate_conj_apply_tail",
]
