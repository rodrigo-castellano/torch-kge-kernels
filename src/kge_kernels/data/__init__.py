"""Knowledge-graph dataset utilities.

Pure-Python helpers shared by DpRL, torch-ns, and any future KGE consumer.

The high-level entry point is :class:`KnowledgeBase` — an extensible
base class that runs the full canonical loading pipeline (vocabulary
discovery, split encoding, filter maps, domain mappings) so consumers
only need to subclass and add their domain-specific structures.

Lower-level building blocks remain available for callers that need more
control:

  ``TripleExample``                 — named triple (head, relation, tail)
  ``load_triples``                  — parse TSV/CSV/Prolog triple files
  ``load_triples_with_mappings``    — load + assign entity/relation ids
  ``encode_split_triples``          — encode a split using an existing vocab
  ``add_reciprocal_triples``        — double the triple set with inverse relations
  ``build_filter_maps``             — filtered-ranking head/tail sets
  ``build_relation_domains``        — per-relation observed head/tail domains
  ``load_dataset_split``            — resolve ``<root>/<dataset>/<split>``
  ``resolve_train_path``            — pick explicit path or resolve via convention
  ``resolve_split_path``            — resolve optional eval split
"""
from __future__ import annotations

from .knowledge_base import KnowledgeBase, MaterializedSplit
from .loaders import (
    TripleExample,
    detect_triple_format,
    encode_split_triples,
    load_probabilistic_facts,
    load_rules_file,
    load_triples,
    load_triples_with_mappings,
)
from .paths import (
    load_dataset_split,
    resolve_split_path,
    resolve_train_path,
)
from .transforms import (
    add_reciprocal_triples,
    build_filter_maps,
    build_relation_domains,
    build_relation_domains_from_file,
    filter_queries_by_predicates,
    iter_queries_with_depth,
    load_depth_file,
    load_domain_file,
)

__all__ = [
    "KnowledgeBase",
    "MaterializedSplit",
    "TripleExample",
    "add_reciprocal_triples",
    "build_filter_maps",
    "build_relation_domains",
    "build_relation_domains_from_file",
    "detect_triple_format",
    "encode_split_triples",
    "filter_queries_by_predicates",
    "iter_queries_with_depth",
    "load_dataset_split",
    "load_depth_file",
    "load_domain_file",
    "load_probabilistic_facts",
    "load_rules_file",
    "load_triples",
    "load_triples_with_mappings",
    "resolve_split_path",
    "resolve_train_path",
]
