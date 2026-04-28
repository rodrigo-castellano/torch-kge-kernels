"""Knowledge-graph dataset utilities.

Public API surface:

- :class:`KnowledgeBase` — base loader running the full canonical
  pipeline (vocabulary discovery, split encoding, filter maps, domain
  mappings, optional materialize). Subclass to add consumer-specific
  structures (DpRL's RL extras, ns's reasoner-side bookkeeping).
- :class:`MaterializedSplit` — generic ``(queries, labels, depths)``
  tensor bundle returned by ``materialize`` for one split.
- File parsers — see :mod:`kge_kernels.data.loaders`.
- Triple-set transforms — see :mod:`kge_kernels.data.transforms`.

Id assignment is **1-based**: id ``0`` is reserved as the padding
sentinel across tkk standalone training, ns reasoner training, and
DpRL RL.
"""
from __future__ import annotations

from .knowledge_base import KnowledgeBase, MaterializedSplit
from .loaders import (
    TripleExample,
    detect_triple_format,
    encode_split_triples,
    load_depth_file,
    load_domain_file,
    load_probabilistic_facts_file,
    load_rules_file,
    load_triples,
    load_triples_with_mappings,
    parse_atom_str,
    parse_prolog_rule,
    resolve_split_path,
    resolve_train_path,
)
from .transforms import (
    add_reciprocal_triples,
    build_filter_maps,
    build_relation_domains,
    build_relation_domains_typed,
    filter_queries_by_predicates,
)

__all__ = [
    "KnowledgeBase",
    "MaterializedSplit",
    "TripleExample",
    "add_reciprocal_triples",
    "build_filter_maps",
    "build_relation_domains",
    "build_relation_domains_typed",
    "detect_triple_format",
    "encode_split_triples",
    "filter_queries_by_predicates",
    "load_depth_file",
    "load_domain_file",
    "load_probabilistic_facts_file",
    "load_rules_file",
    "load_triples",
    "load_triples_with_mappings",
    "parse_atom_str",
    "parse_prolog_rule",
    "resolve_split_path",
    "resolve_train_path",
]
