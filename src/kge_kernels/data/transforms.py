"""Triple-set transforms: reciprocal augmentation, filter maps, domains."""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def load_domain_file(
    path: str,
    entity2id: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, List[Any]], Dict[Any, str]]:
    """Parse a ``domain_name entity1 entity2 ...`` file into dicts.

    Each line lists a domain name followed by entity names belonging to
    it. Lines whose first token contains ``:`` and no whitespace before
    the colon are treated as predicate-domain mappings (``pred:head:tail``)
    and silently skipped — that legacy format coexists with entity-domain
    lines in some DpRL datasets.

    Two modes (selected by ``entity2id``):

    - **Indexed mode** (``entity2id`` provided): returns
      ``(domain2idx: dict[str, list[int]], entity2domain: dict[int, str])``.
      Unknown entities (not in ``entity2id``) are silently skipped so
      callers can pass a vocabulary that's a subset of the file's
      entities. This is the path tkk's training pipeline uses.

    - **String mode** (``entity2id=None``): returns
      ``(domain2entities: dict[str, list[str]], entity2domain: dict[str, str])``
      with raw entity names. Useful for consumers that need to read
      domains *before* id-assignment (ns's ``KGCDataHandler``, DpRL's
      ``DataHandler._load_domain_mapping``).

    If ``path`` does not exist the function returns empty dicts in either
    mode. Callers that require the file to exist should check before
    calling.
    """
    domain2members: Dict[str, List[Any]] = {}
    entity2domain: Dict[Any, str] = {}
    if not os.path.isfile(path):
        return domain2members, entity2domain
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip predicate domain lines like ``pred:head_dom:tail_dom``
            # that some DpRL dataset variants emit alongside the
            # ``domain_name entity1 ...`` lines.
            first = stripped.split(None, 1)[0]
            if ":" in first:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            domain_name = parts[0]
            if entity2id is None:
                members: List[Any] = list(parts[1:])
            else:
                members = [entity2id[c] for c in parts[1:] if c in entity2id]
            if not members:
                continue
            domain2members[domain_name] = members
            for member in members:
                entity2domain.setdefault(member, domain_name)
    return domain2members, entity2domain


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
    """Build per-relation domain sets using the domain-file memberships.

    Unlike :func:`build_relation_domains`, which only includes entities
    *observed* in a given position, this function looks up the domain of
    each position's entities and returns **all** entities belonging to
    that domain.  This matches the evaluation protocol of ns-old's
    ``_IndexedCorruptionAdapter``, which draws exhaustive corruption
    candidates from the full domain file pool.

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
    "load_domain_file",
]
