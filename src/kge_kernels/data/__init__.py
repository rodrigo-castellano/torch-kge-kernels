"""Knowledge-graph dataset utilities.

Pure-Python helpers shared by DpRL, torch-ns, and any future KGE consumer:

  ``TripleExample``                 — named triple (head, relation, tail)
  ``load_triples``                  — parse TSV/CSV/Prolog triple files
  ``load_triples_with_mappings``    — load + assign entity/relation ids
  ``encode_split_triples``          — encode a split using an existing vocab
  ``add_reciprocal_triples``        — double the triple set with inverse relations
  ``build_filter_maps``             — filtered-ranking head/tail sets
  ``build_relation_domains``        — per-relation observed head/tail domains

Dataset path resolution (``load_dataset_split``, ``resolve_*_path``) is
intentionally left out: those depend on each consumer's directory layout.
"""
from __future__ import annotations

from .loaders import (
    TripleExample,
    detect_triple_format,
    load_triples,
    load_triples_with_mappings,
    encode_split_triples,
)
from .transforms import (
    add_reciprocal_triples,
    build_filter_maps,
    build_relation_domains,
)

__all__ = [
    "TripleExample",
    "add_reciprocal_triples",
    "build_filter_maps",
    "build_relation_domains",
    "detect_triple_format",
    "encode_split_triples",
    "load_triples",
    "load_triples_with_mappings",
]
