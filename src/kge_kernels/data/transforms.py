"""Triple-set transforms: reciprocal augmentation, filter maps, domains."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple


def add_reciprocal_triples(
    triples: List[Tuple[int, int, int]],
    relation2id: Dict[str, int],
    inv_suffix: str = "__inv",
) -> Tuple[List[Tuple[int, int, int]], Dict[str, int], int]:
    """Augment a triple set with reciprocal relations.

    For each relation ``r`` with name ``name``, adds a new relation named
    ``name + inv_suffix`` with id ``r + num_relations``. For each triple
    ``(r, h, t)``, adds ``(r + num_relations, t, h)``.

    Returns ``(augmented_triples, expanded_relation2id, new_num_relations)``.
    """
    num_relations = len(relation2id)
    id2rel = {idx: name for name, idx in relation2id.items()}
    expanded = dict(relation2id)
    for ridx in range(num_relations):
        expanded[f"{id2rel[ridx]}{inv_suffix}"] = ridx + num_relations
    inv = [(r + num_relations, t, h) for (r, h, t) in triples]
    return triples + inv, expanded, num_relations * 2


def build_filter_maps(
    *triple_collections: Sequence[Tuple[int, int, int]],
) -> Tuple[Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int], Set[int]]]:
    """Build filtered-ranking lookup tables from one or more triple sets.

    Used by filtered MRR / Hits@K: when ranking head candidates for
    ``(?, r, t)``, every entity in ``head_filter[(r, t)]`` is a known
    positive and should not be counted as a negative. Analogous for tail.

    Returns ``(head_filter, tail_filter)``.
    """
    head_filter: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    tail_filter: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    for triples in triple_collections:
        if not triples:
            continue
        for r, h, t in triples:
            head_filter[(r, t)].add(h)
            tail_filter[(h, r)].add(t)
    return dict(head_filter), dict(tail_filter)


def build_relation_domains(
    triples: Sequence[Tuple[int, int, int]],
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Build per-relation observed head/tail entity sets.

    Returns ``(head_domain, tail_domain)`` mapping relation id → entity ids
    that ever appeared in that position for that relation in the input.
    """
    head_domain: Dict[int, Set[int]] = defaultdict(set)
    tail_domain: Dict[int, Set[int]] = defaultdict(set)
    for r, h, t in triples:
        head_domain[r].add(h)
        tail_domain[r].add(t)
    return dict(head_domain), dict(tail_domain)


__all__ = [
    "add_reciprocal_triples",
    "build_filter_maps",
    "build_relation_domains",
]
