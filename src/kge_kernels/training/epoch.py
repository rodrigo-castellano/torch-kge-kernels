"""Shared one-epoch trainer with static-buffer compiled step.

The compile-boundary-agnostic outer loop that both tkk's
``train_model`` and torch-ns's ``train_loop`` call. Each iteration:

1. Pull a ``(pos [B, 3], neg [B, K_total, 3], mask [B, K_total])`` tuple
   from the sampler via :func:`iterate_epoch_batches`.
2. Copy into preallocated static buffers, padding the tail of partial
   last batches with zeros and gating via ``pos_valid``.
3. Call the compiled ``model.train_step(pos, neg, mask, pos_valid)`` —
   the model's hook that builds its preferred atom layout (static-flat
   ``[B + B*K, 3]`` for KGE, static-pool ``[B*(1+K), 3]`` for reasoners)
   and returns the scalar loss.
4. Backward + optimizer step in eager — still CUDA-graph-safe because
   ``zero_grad(set_to_none=False)`` keeps ``.grad`` addresses stable
   and shapes never vary.

The model-side ``train_step`` is the compile boundary. This outer loop
allocates buffers and drives ``torch.compile(fullgraph=True,
mode='reduce-overhead')`` + ``cudagraph_mark_step_begin`` between
steps — the DpRL PPO rollout pattern applied to KGE training. Same
model exposes one compiled graph per ``(B, K)`` combo, shared across
every epoch.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from .batching import iterate_epoch_batches

__all__ = ["train_epoch", "clear_train_cache"]


# Per-model cache, same pattern as ``_tkk_eval_state_cache``. Key is
# ``(B, K, compile, device)``; value is a :class:`_TrainState` with the
# static buffers + compiled step closure. Lifetime follows the model
# (cache dies with the model; CPython cyclic GC handles the closure ↔
# model reference cycle).
_TRAIN_STATE_ATTR = "_tkk_train_state_cache"


class _TrainState:
    """Cached train buffers + compiled step for one (model, B, K) combo.

    Buffers:

    - ``pos_buf       [B, 3]``      long, positive triples (r, h, t).
    - ``neg_buf       [B, K, 3]``   long, candidate negatives.
    - ``mask_buf      [B, K]``      bool, per-slot validity.
    - ``pos_valid_buf [B]``         bool, per-row positive validity
      (False for padded rows in the last partial batch).

    All four are mark-static-address so CUDA graph replay finds them
    at captured pointers. On first call the compiled step traces +
    captures; every subsequent call replays.
    """

    __slots__ = (
        "B", "K", "device",
        "pos_buf", "neg_buf", "mask_buf", "pos_valid_buf",
        "compiled_step",
    )

    def __init__(
        self,
        model: nn.Module,
        B: int,
        K: int,
        device: torch.device,
        compile_enabled: bool,
    ) -> None:
        self.B = B
        self.K = K
        self.device = device

        self.pos_buf = torch.empty(B, 3, dtype=torch.long, device=device)
        self.neg_buf = torch.empty(B, K, 3, dtype=torch.long, device=device)
        self.mask_buf = torch.empty(B, K, dtype=torch.bool, device=device)
        self.pos_valid_buf = torch.empty(B, dtype=torch.bool, device=device)

        if hasattr(torch, "_dynamo"):
            for _b in (self.pos_buf, self.neg_buf, self.mask_buf, self.pos_valid_buf):
                torch._dynamo.mark_static_address(_b)

        def _step(pos: Tensor, neg: Tensor, mask: Tensor, pos_valid: Tensor) -> Tensor:
            return model.train_step(pos, neg, mask, pos_valid)

        if compile_enabled:
            # ``mode='reduce-overhead'`` matches DpRL's PPO compile mode:
            # fullgraph + CUDA-graph capture via cudagraph trees. The
            # known "accessing tensor output of CUDAGraphs" error on KGE
            # models with multi-use parameter paths (ComplEx's real/imag
            # split, RotatE's two entity_embeddings lookups) is avoided
            # by the ordering in ``train_epoch``: mark_step_begin →
            # forward → zero_grad → backward → step, matching DpRL.
            self.compiled_step = torch.compile(
                _step, fullgraph=True, mode="reduce-overhead",
            )
        else:
            self.compiled_step = _step


def _get_train_state(
    model: nn.Module,
    B: int,
    K: int,
    device: torch.device,
    compile_enabled: bool,
) -> _TrainState:
    """Fetch or allocate the cached :class:`_TrainState` on ``model``."""
    cache = getattr(model, _TRAIN_STATE_ATTR, None)
    if cache is None:
        cache = {}
        object.__setattr__(model, _TRAIN_STATE_ATTR, cache)
    key = (B, K, bool(compile_enabled), str(device))
    st = cache.get(key)
    if st is None:
        st = _TrainState(model, B, K, device, compile_enabled)
        cache[key] = st
    return st


def clear_train_cache(model: nn.Module) -> None:
    """Drop the cached train buffers + compiled step on ``model``.

    Not normally needed: the cache dies with the model. Call only to
    force a rebuild (e.g. the model's compile spec changed in-place)
    or to reclaim CUDA-graph pools before destroying the model.
    """
    if hasattr(model, _TRAIN_STATE_ATTR):
        try:
            object.__delattr__(model, _TRAIN_STATE_ATTR)
        except AttributeError:
            pass


def train_epoch(
    model: nn.Module,
    sampler,                                   # kge_kernels.scoring.Sampler
    optimizer: torch.optim.Optimizer,
    train_triples: Tensor,                     # [N, 3]  int64, (r, h, t)
    *,
    batch_size: int,
    num_negatives: int,
    corrupt_modes: List[str],
    grad_clip: float = 0.0,
    scaler: Optional[torch.amp.GradScaler] = None,
    filter_negatives: bool = True,
    unique_negatives: bool = False,
    compile: bool = True,
) -> Dict[str, float]:
    """Run a single training epoch using static-buffer compiled steps.

    Args:
        model: Module with a ``train_step(pos, neg, mask, pos_valid) ->
            Tensor`` hook. :class:`kge_kernels.models.KGEModel` provides
            a lean static-flat default; ns's ``ReasonerModel`` overrides
            with a padded ``[B*(1+K), 3]`` pool.
        sampler: tkk corruption sampler (``from_data``-constructed).
        optimizer: Torch optimizer already holding model parameters.
        train_triples: Positive training triples on the target device.
        batch_size: Queries per mini-batch (matches tkk's convention).
        num_negatives: Negatives sampled per corruption mode per query.
        corrupt_modes: e.g. ``["head", "tail"]`` or one of them.
        grad_clip: If ``>0``, clip grad norm to this value per step.
        scaler: Optional ``torch.amp.GradScaler`` for mixed precision.
        filter_negatives: Exclude known positives from drawn negatives.
        unique_negatives: Deduplicate drawn negatives within a row.
        compile: If True, wrap ``train_step`` with
            ``torch.compile(fullgraph=True, mode='reduce-overhead')``
            and mark buffers static so CUDA graphs replay across
            every step of every epoch.

    Returns:
        ``{"loss": avg_loss, "n_batches": int}`` — averaged over steps.
    """
    model.train()
    use_amp = scaler is not None
    device = train_triples.device
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    n_batches = 0

    B = batch_size
    K = len(corrupt_modes) * num_negatives

    # Static buffers for BOTH ``compile=True`` and ``compile=False``.
    # Why both: KGE benefits from compiling the outer step; reasoners
    # already compile their ``forward`` internally, and feeding that
    # compiled forward variable-shape inputs per batch triggers a
    # re-capture every batch (18× slowdown in the ns speed regression).
    # Fixed-B static buffers keep the inner graph stable whether or not
    # the outer step is also compiled.
    state = _get_train_state(model, B, K, device, compile)
    pos_buf = state.pos_buf
    neg_buf = state.neg_buf
    mask_buf = state.mask_buf
    pos_valid_buf = state.pos_valid_buf
    compiled_step = state.compiled_step

    for pos, neg, mask in iterate_epoch_batches(
        train_triples, sampler,
        batch_size=B, num_negatives=num_negatives,
        corrupt_modes=corrupt_modes,
        filter=filter_negatives, unique=unique_negatives,
    ):
        actual_B = pos.shape[0]

        # Copy the live portion into static buffers; zero the tail so
        # padded rows produce zero gradient contribution (via mask +
        # pos_valid). Only the last partial batch trips this.
        pos_buf[:actual_B].copy_(pos)
        neg_buf[:actual_B].copy_(neg)
        mask_buf[:actual_B].copy_(mask)
        pos_valid_buf[:actual_B] = True
        if actual_B < B:
            pos_buf[actual_B:].zero_()
            neg_buf[actual_B:].zero_()
            mask_buf[actual_B:] = False
            pos_valid_buf[actual_B:] = False

        # DpRL PPO ordering: mark_step_begin BEFORE forward, then zero
        # grads AFTER forward but BEFORE backward. With reduce-overhead
        # CUDA-graph trees, this sequence is what avoids the
        # "accessing tensor output of CUDAGraphs that has been
        # overwritten" error on KGE models with multi-use parameter
        # paths (ComplEx real/imag split, RotatE h+t entity lookups).
        if compile:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = compiled_step(pos_buf, neg_buf, mask_buf, pos_valid_buf)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.detach()
        n_batches += 1

    avg = (total_loss / max(n_batches, 1)).item()
    return {"loss": avg, "n_batches": n_batches}
