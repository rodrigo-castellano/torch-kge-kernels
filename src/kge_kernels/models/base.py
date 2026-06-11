"""Abstract base class for KGE models in tkk.

A ``KGEModel`` is a ``torch.nn.Module`` whose contract is just the
KGE math. Training and eval orchestration are free functions in
:mod:`kge_kernels.training` and :mod:`kge_kernels.eval` that operate
over a model.

Required attributes (set in ``__init__``):
  - ``num_entities: int``
  - ``num_relations: int``
  - ``dim: int``
  - ``entity_embeddings: nn.Embedding``  (the entity table)

Abstract — subclass MUST implement:
  - ``score(h, r, t, *, d_chunk=None) -> Tensor`` — single scoring entry
    point. ``h, r, t`` all bound: returns ``[B]`` triple scores.
    ``t=None``: returns ``[B, E]`` ranking against all tails. ``h=None``:
    returns ``[B, E]`` ranking against all heads. ``d_chunk=None``
    (default) means score every entity in one pass; an integer enables
    memory-chunked scoring on models that support it (RotatE / RotatENS).
  - ``compose(h, r, t) -> Tensor`` — fused per-atom embedding consumed
    by ``KGEEmbedAtom`` (the layer that turns atoms into MLP features).
  - ``reset_parameters() -> None`` — re-init weights (called from
    ``__init__`` and from external repro/seed setups).

Provided by base:
  - ``forward(h, r, t)`` — ``nn.Module`` convention; delegates to
    ``score``.

Optional class flag:
  - ``_train_loss_is_from_logits: bool = True`` — read by
    :func:`kge_kernels.training.train_step` to decide BCE-with-logits
    vs sigmoid+BCE. RotatE / RotatENS set this to ``False``.

External orchestration (free functions over a model):
  - :func:`kge_kernels.training.train_step` — mask-aware BCE loss
    (unified compile boundary).
  - :func:`kge_kernels.models.kge_default_scorer` — adapt ``score`` to
    the evaluator's ``(q_buf, pool_buf, mode) -> [B, C]`` ScoreFn shape.
  - :func:`kge_kernels.models.recommended_eval_batch_size` — memory-aware
    batch-size hint.

Subclasses with a different atom layout (e.g. ns's ``ReasonerModel``
building a ``[B*(1+K), 3]`` pool for the grounder) override the
training / eval hooks by **defining their own ``train_step`` /
``eval_scores`` methods on the model**: tkk's
:func:`kge_kernels.training.train_epoch` calls ``model.train_step`` when
present, and :class:`kge_kernels.eval.RankingEvaluator` accepts any
``ScoreFn`` (including ``model.eval_scores``) — fall back to the
free-function defaults otherwise.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn


class _DetEmbedding(torch.autograd.Function):
    """Embedding lookup with a DETERMINISTIC-and-fast backward.

    Under ``torch.use_deterministic_algorithms(True)`` PyTorch's stock
    embedding backward falls back to ``indexing_backward_kernel`` — a
    serialized per-row kernel that measured 72% of a torch-ns reasoner
    train step's CUDA time (21.2 of 29.4 ms; 2026-06-10). This backward is
    deterministic BY CONSTRUCTION instead: stable-sort the indices, sum
    each segment via a cumsum difference (fixed summation order), and
    ``index_copy_`` the per-row sums into the grad table (unique targets).
    Same asymptotics as the fast nondeterministic atomics path; gradient
    values differ from it only in FP summation order, and runs are
    bit-reproducible regardless of the global determinism mode.
    """

    @staticmethod
    def forward(ctx, weight: Tensor, idx: Tensor) -> Tensor:
        ctx.save_for_backward(idx)
        ctx.num_rows = weight.shape[0]
        return weight[idx]

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (idx,) = ctx.saved_tensors
        D = grad_out.shape[-1]
        E = ctx.num_rows
        flat_idx = idx.reshape(-1)
        T = flat_idx.shape[0]
        g = grad_out.reshape(-1, D)
        sidx, perm = torch.sort(flat_idx, stable=True)
        g = g[perm]
        # Exclusive-prefix cumsum: csz[k] = sum of g[:k] (fixed summation
        # order → deterministic values).
        csz = torch.cat([g.new_zeros(1, D), torch.cumsum(g, 0)], dim=0)
        # Per-row segment bounds in the sorted layout (segments are
        # contiguous): first/last position per present row via order-free
        # scatter_reduce (amin/amax) — fixed shapes, fullgraph-traceable.
        pos = torch.arange(T, device=g.device)
        first = torch.full((E,), T, dtype=torch.long, device=g.device)
        first.scatter_reduce_(0, sidx, pos, reduce="amin", include_self=True)
        last = torch.full((E,), -1, dtype=torch.long, device=g.device)
        last.scatter_reduce_(0, sidx, pos, reduce="amax", include_self=True)
        present = last >= 0
        # row_sum = csz[last+1] - csz[first]; absent rows index harmlessly
        # (clamped) and are zeroed by the mask.
        hi = (last + 1).clamp(min=0, max=T)
        lo = first.clamp(min=0, max=T)
        grad_w = (csz[hi] - csz[lo]) * present.unsqueeze(1).to(g.dtype)
        return grad_w, None


def _det_embedding_grad(idx: Tensor, grad_out: Tensor, num_rows: int) -> Tensor:
    """Deterministic embedding grad: sort + cumsum segment-sum (fixed shapes,
    fixed summation order; traceable plain-torch ops — see _DetEmbedding)."""
    D = grad_out.shape[-1]
    flat_idx = idx.reshape(-1)
    T = flat_idx.shape[0]
    g = grad_out.reshape(-1, D)
    sidx, perm = torch.sort(flat_idx, stable=True)
    g = g[perm]
    csz = torch.cat([g.new_zeros(1, D), torch.cumsum(g, 0)], dim=0)
    pos = torch.arange(T, device=g.device)
    first = torch.full((num_rows,), T, dtype=torch.long, device=g.device)
    first.scatter_reduce_(0, sidx, pos, reduce="amin", include_self=True)
    last = torch.full((num_rows,), -1, dtype=torch.long, device=g.device)
    last.scatter_reduce_(0, sidx, pos, reduce="amax", include_self=True)
    present = last >= 0
    hi = (last + 1).clamp(min=0, max=T)
    lo = first.clamp(min=0, max=T)
    return (csz[hi] - csz[lo]) * present.unsqueeze(1).to(g.dtype)


# Custom op so the deterministic backward formula ALSO applies inside
# torch.compile regions: AOTAutograd traces the registered autograd formula
# (plain torch ops) instead of falling back to ATen's serialized
# ``indexing_backward_kernel`` under torch.use_deterministic_algorithms.
@torch.library.custom_op("kge_kernels::det_gather", mutates_args=())
def _det_gather(weight: Tensor, idx: Tensor) -> Tensor:
    return weight[idx].clone()


@_det_gather.register_fake
def _(weight, idx):
    return weight.new_empty(*idx.shape, weight.shape[-1])


def _det_gather_setup(ctx, inputs, output):
    weight, idx = inputs
    ctx.save_for_backward(idx)
    ctx.num_rows = weight.shape[0]


# Opaque identity: a realization barrier for inductor. Without it the
# scheduler can fuse the upstream (masked-slice) grad pointwise INTO the
# backward's ``g[perm]`` indirect gather, which trips a codegen bug in
# dynamic-shape regions (torch 2.10: `TritonSymbols.get_block_shape` →
# tensor_dim=None). The formula itself stays traced so inductor still
# fuses the gather+cumsum segment-sum (the fast path); only grad_out is
# forced to a materialized buffer first.
@torch.library.custom_op("kge_kernels::det_gather_grad_in", mutates_args=())
def _det_gather_grad_in(grad_out: Tensor) -> Tensor:
    return grad_out.clone()


@_det_gather_grad_in.register_fake
def _(grad_out):
    return torch.empty_like(grad_out)


def _det_gather_backward(ctx, grad_out):
    (idx,) = ctx.saved_tensors
    grad_out = _det_gather_grad_in(grad_out)
    return _det_embedding_grad(idx, grad_out, ctx.num_rows), None


_det_gather.register_autograd(_det_gather_backward, setup_context=_det_gather_setup)


def _det_gather_dispatch(table: Tensor, idx: Tensor) -> Tensor:
    """Route a 2-D row gather ``table[idx]`` to the right det-backward path.

    Eager + grad: the custom Function (zero-copy forward). Compiled: the
    ``kge_kernels::det_gather`` custom op, whose registered autograd formula
    is the same fixed-shape segment-sum — inductor codegens it instead of
    hitting ATen's det-mode ``indexing_backward_kernel`` fallback (measured
    72% of a reasoner train step). No-grad: plain gather.
    """
    if torch.compiler.is_compiling():
        if torch.is_grad_enabled() and table.requires_grad:
            return _det_gather(table, idx)
        return table[idx]
    if table.requires_grad and torch.is_grad_enabled():
        return _DetEmbedding.apply(table, idx)
    return table[idx]


def det_embedding(emb: nn.Embedding, idx: Tensor) -> Tensor:
    """``emb(idx)`` with the deterministic-fast backward, eager AND compiled.

    See :func:`_det_gather_dispatch` for the routing.
    """
    return _det_gather_dispatch(emb.weight, idx)


def det_gather_rows(table: Tensor, idx: Tensor) -> Tensor:
    """``table[idx]`` with the deterministic-fast backward; table [N] or [N, D].

    Shape-agnostic activation-gather variant of :func:`det_embedding` for
    grad-bearing float tables (e.g. reasoner atom pools): 1-D tables are
    viewed as ``[N, 1]`` for the row dispatch and the trailing dim is
    squeezed back off the output, so out-shape and dtype match plain
    ``table[idx]`` exactly. Int-index / no-grad gathers should stay plain
    indexing — this helper only pays off when the TABLE side carries grad.
    """
    if table.dim() == 1:
        return _det_gather_dispatch(table.unsqueeze(-1), idx).squeeze(-1)
    if table.dim() == 2:
        return _det_gather_dispatch(table, idx)
    raise ValueError(
        f"det_gather_rows expects a [N] or [N, D] table; got {tuple(table.shape)}"
    )


class KGEModel(nn.Module):
    """Base class for KGE models. See module docstring for the contract."""

    num_entities: int
    num_relations: int
    dim: int

    # Read by kge_kernels.training.train_step. Override on subclasses
    # that need sigmoid+BCE instead of BCE-with-logits (RotatE family).
    _train_loss_is_from_logits: bool = True

    @abstractmethod
    def score(
        self,
        h: Optional[Tensor],
        r: Tensor,
        t: Optional[Tensor],
        *,
        d_chunk: Optional[int] = None,
    ) -> Tensor:
        """Score (h, r, t) triples or rank against all heads/tails.

        Modes (decided by which of ``h`` / ``t`` are bound):

        - ``h, r, t`` all bound: returns ``[B]`` per-triple scores.
        - ``h, r`` bound, ``t=None``: returns ``[B, E]`` ranking all tails.
        - ``h=None``, ``r, t`` bound: returns ``[B, E]`` ranking all heads.

        ``d_chunk``:
          - ``None`` (default): score all entities in **one pass** (full
            broadcast / matmul). Use this whenever memory is not the
            bottleneck.
          - ``int``: enables memory-chunked exhaustive scoring along the
            embedding dimension. Currently implemented by RotatE /
            RotatENS — other models ignore the value and still do
            one-pass.
        """

    @abstractmethod
    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused per-atom embedding ``[B, *]`` consumed by KGEEmbedAtom."""

    @abstractmethod
    def reset_parameters(self) -> None:
        """Re-initialise weights."""

    def forward(
        self,
        h: Optional[Tensor],
        r: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:  # noqa: D401
        """Default forward delegates to ``score`` for nn.Module compatibility."""
        return self.score(h, r, t)


__all__ = ["KGEModel", "det_embedding", "det_gather_rows"]
