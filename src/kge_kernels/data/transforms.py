"""Pure-functional triple-set transforms.

Operate on plain ``[(r, h, t), ...]`` lists of integer-indexed triples;
no I/O, no class state. Used by :class:`KnowledgeBase` (filter maps,
domain restriction) and by tkk's training / eval pipelines for the
reciprocal-relation augmentation and observed-domain restriction
needed at training time.

File parsers (``load_domain_file``, ``load_depth_file``) live in
:mod:`kge_kernels.data.loaders`.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple


def filter_queries_by_predicates(
    query_tuples: Sequence[Tuple[str, ...]],
    allowed_predicates: Set[str],
) -> Tuple[List[Tuple[str, ...]], List[int]]:
    """Keep only queries whose predicate is in ``allowed_predicates``.

    Used by rule-based filtering: queries whose head predicate doesn't
    match any rule head are unsolvable for a proof-based reasoner, so
    they're dropped before training. Content-agnostic — callers decide
    what set of predicates to allow.

    Returns:
        ``(filtered_queries, kept_indices)`` where ``kept_indices`` lists
        the positions in the original sequence that survived filtering, in
        order. Callers can use it to re-align parallel arrays (depths,
        labels, etc.) without re-iterating.
    """
    filtered: List[Tuple[str, ...]] = []
    kept_indices: List[int] = []
    for i, q in enumerate(query_tuples):
        if not q:
            continue
        if q[0] in allowed_predicates:
            filtered.append(q)
            kept_indices.append(i)
    return filtered, kept_indices


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


def build_relation_domains_from_file(
    triples: Sequence[Tuple[int, int, int]],
    entity2domain: Dict[int, str],
    domain2idx: Dict[str, List[int]],
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Build per-relation domain sets using domain-file memberships.

    Unlike :func:`build_relation_domains` (observed-only), this function
    looks up the domain of each position's entities and returns **all**
    entities belonging to that domain. Matches ns-old's
    ``_IndexedCorruptionAdapter`` evaluation protocol, which draws
    exhaustive corruption candidates from the full domain file pool.

    Falls back to observed entities for relations whose entities have no
    domain mapping.
    """
    head_domain: Dict[int, Set[int]] = {}
    tail_domain: Dict[int, Set[int]] = {}
    for r, h, t in triples:
        if r not in head_domain:
            h_dom = entity2domain.get(h)
            if h_dom is not None and h_dom in domain2idx:
                head_domain[r] = set(domain2idx[h_dom])
            else:
                head_domain[r] = set()
            head_domain[r].add(h)
        if r not in tail_domain:
            t_dom = entity2domain.get(t)
            if t_dom is not None and t_dom in domain2idx:
                tail_domain[r] = set(domain2idx[t_dom])
            else:
                tail_domain[r] = set()
            tail_domain[r].add(t)
    return head_domain, tail_domain


__all__ = [
    "add_reciprocal_triples",
    "build_filter_maps",
    "build_relation_domains",
    "build_relation_domains_from_file",
    "filter_queries_by_predicates",
]
