"""Filtered-ranking evaluation for KGE models.

Implements the standard "filtered" evaluation protocol used by KGE
papers since TransE: for each test triple ``(h, r, t)``, rank ``h``
against all entities that do not appear in ``head_filter[(r, t)]`` and
``t`` against all entities that do not appear in ``tail_filter[(h, r)]``.
Metrics are averaged over both sides of every triple.

Supports two modes:

  - **Exhaustive** (``eval_num_corruptions == 0``): rank against every
    entity (minus filtered known positives).
  - **Sampled** (``eval_num_corruptions > 0``): rank against a seeded
    random subset of filtered candidates. Faster and matches DpRL's
    default protocol.

Optional per-relation *domain* constraints (``head_domain`` /
``tail_domain``) restrict candidates to entities the relation has ever
been observed with, matching the Countries / Kinship convention.

The model only needs to expose a single ``model(h, r, t)`` method: the
evaluator handles ``DataParallel`` / ``torch.compile`` unwrapping and
dispatches through :func:`kge_kernels.adapter.kge_score_all_heads` and
:func:`kge_kernels.adapter.kge_score_all_tails` which apply sigmoid
normalization internally.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor, nn

from ..adapter import kge_score_all_heads, kge_score_all_tails
from ..ranking import ranking_metrics, ranks_from_scores_matrix


def _sampled_ranks(
    scores: Tensor,
    true_indices: Tensor,
    num_corruptions: int,
    generator: torch.Generator,
) -> Tensor:
    """Rank true entities among a random subset of filtered candidates.

    ``scores`` already has ``float('-inf')`` at positions that are
    filtered out; invalid entries are dropped from the random sample.
    Ties are resolved with average tie-handling.
    """
    batch_size, num_entities = scores.shape
    device = scores.device

    valid = torch.isfinite(scores)
    rand_keys = torch.rand(
        batch_size, num_entities, generator=generator, device=device
    )
    rand_keys[~valid] = -1.0
    # Anchor the true entity at key=2.0 so it is always selected.
    rand_keys[torch.arange(batch_size, device=device), true_indices] = 2.0

    topk = min(num_corruptions + 1, num_entities)
    _, selected_idx = rand_keys.topk(topk, dim=1)
    selected_scores = scores.gather(1, selected_idx)

    true_pos = (selected_idx == true_indices.unsqueeze(1)).nonzero(as_tuple=True)[1]
    target_scores = selected_scores[torch.arange(batch_size, device=device), true_pos]

    greater = (selected_scores > target_scores.unsqueeze(1)).sum(
        dim=1, dtype=torch.float32
    )
    equal = (selected_scores == target_scores.unsqueeze(1)).sum(
        dim=1, dtype=torch.float32
    )
    return greater + 1.0 + 0.5 * (equal - 1.0).clamp(min=0)


def _unwrap_model(model: nn.Module) -> nn.Module:
    inner: nn.Module = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod  # type: ignore[assignment]
    return inner


def evaluate_filtered_ranking(
    model: nn.Module,
    triples: Sequence[Tuple[int, int, int]],
    num_entities: int,
    head_filter: Dict[Tuple[int, int], Set[int]],
    tail_filter: Dict[Tuple[int, int], Set[int]],
    device: torch.device,
    *,
    chunk_size: int = 0,
    head_domain: Optional[Dict[int, Set[int]]] = None,
    tail_domain: Optional[Dict[int, Set[int]]] = None,
    eval_num_corruptions: int = 0,
    seed: int = 0,
    show_progress: bool = False,
    progress_label: str = "Evaluation",
) -> Dict[str, float]:
    """Run filtered ranking evaluation on a KGE model.

    Returns the dict produced by :func:`kge_kernels.ranking.ranking_metrics`
    (``MRR``, ``Hits@1``, ``Hits@3``, ``Hits@10``), averaged over
    head-side and tail-side queries of every triple in ``triples``.

    Args:
        model: A KGE model exposing the ``KGEBackend`` contract
            (either directly or via ``score(h, r, None)`` /
            ``score(None, r, t)`` dispatch). ``DataParallel`` and
            ``torch.compile`` wrappers are handled automatically.
        triples: Test triples in ``(r, h, t)`` order.
        num_entities: Size of the entity vocabulary.
        head_filter: ``{(r, t): {known_heads}}`` — candidates to mask
            when ranking heads (the true head is not masked, so the
            rank against the other known-true heads is "filtered").
        tail_filter: ``{(h, r): {known_tails}}`` — analogous for tails.
        device: Device on which to run the model forward pass.
        chunk_size: Unused placeholder kept for API parity with the
            DpRL version (kept for future use; the chunking heuristic
            is computed internally from available GPU memory).
        head_domain: Optional ``{r: {valid_head_entities}}`` limiting
            head candidates to entities ever seen in that position for
            that relation. Used by Countries / Kinship.
        tail_domain: Analogous for tails.
        eval_num_corruptions: ``0`` → exhaustive ranking against all
            entities. Positive → sampled ranking against that many
            random filtered candidates (faster, lower variance).
        seed: Seed for the random candidate sampler.
        show_progress: If ``True``, print rolling MRR every ~5% of
            triples (printed to stdout, not a progress bar).
        progress_label: Prefix for the rolling-progress lines.
    """
    if not triples:
        return {
            "MRR": float("nan"),
            "Hits@1": float("nan"),
            "Hits@3": float("nan"),
            "Hits@10": float("nan"),
        }
    del chunk_size  # reserved for API compatibility

    training_mode = model.training
    model.eval()

    actual_model = _unwrap_model(model)

    sampled = eval_num_corruptions > 0
    generator: Optional[torch.Generator] = None
    if sampled:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    # Chunk size heuristic: cap at 512 or whatever fits in ~4 GB assuming
    # float32 scores of shape [chunk, num_entities, dim]. This matches
    # the DpRL heuristic.
    dim = 512
    if hasattr(actual_model, "dim"):
        dim = int(actual_model.dim)
    elif hasattr(actual_model, "entity_dim"):
        dim = int(actual_model.entity_dim)
    max_chunk = max(1, min(512, (4 * 1024**3) // max(1, num_entities * dim * 4)))

    by_relation: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for r, h, t in triples:
        by_relation[r].append((r, h, t))

    all_ranks_list: List[Tensor] = []
    processed = 0
    last_report = 0
    report_every = max(1, len(triples) // 20)

    with torch.no_grad():
        for r, rel_triples in by_relation.items():
            all_heads = torch.tensor(
                [h for _, h, _ in rel_triples], dtype=torch.long, device=device
            )
            all_tails = torch.tensor(
                [t for _, _, t in rel_triples], dtype=torch.long, device=device
            )
            r_tensor = torch.tensor(r, dtype=torch.long, device=device)
            t_domain = tail_domain.get(r) if tail_domain else None
            h_domain = head_domain.get(r) if head_domain else None

            for chunk_start in range(0, len(rel_triples), max_chunk):
                chunk_end = min(chunk_start + max_chunk, len(rel_triples))
                heads = all_heads[chunk_start:chunk_end]
                tails = all_tails[chunk_start:chunk_end]

                tail_scores = kge_score_all_tails(
                    actual_model,
                    heads,
                    r_tensor,
                    filter_map=tail_filter,
                    true_tails=tails,
                    domain=t_domain,
                )
                head_scores = kge_score_all_heads(
                    actual_model,
                    r_tensor,
                    tails,
                    filter_map=head_filter,
                    true_heads=heads,
                    domain=h_domain,
                )

                if sampled:
                    assert generator is not None
                    tail_ranks = _sampled_ranks(
                        tail_scores, tails, eval_num_corruptions, generator
                    )
                    head_ranks = _sampled_ranks(
                        head_scores, heads, eval_num_corruptions, generator
                    )
                else:
                    tail_ranks = ranks_from_scores_matrix(
                        tail_scores, tails, tie_handling="average"
                    )
                    head_ranks = ranks_from_scores_matrix(
                        head_scores, heads, tie_handling="average"
                    )

                all_ranks_list.append(torch.cat([tail_ranks, head_ranks]))

            processed += len(rel_triples)
            if show_progress and (
                processed >= len(triples) or processed - last_report >= report_every
            ):
                running_ranks = torch.cat(all_ranks_list)
                running_mrr = (1.0 / running_ranks.double()).mean().item()
                print(
                    f"{progress_label} {processed}/{len(triples)} triples | "
                    f"rolling_mrr={running_mrr:.4f}"
                )
                last_report = processed

    if training_mode:
        model.train()

    if not all_ranks_list:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    return ranking_metrics(torch.cat(all_ranks_list))


__all__ = ["evaluate_filtered_ranking"]
