"""Knowledge-graph dataset utilities.

Pure-Python helpers shared by DpRL, torch-ns, and any future KGE consumer:

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

from .loaders import (
    TripleExample,
    detect_triple_format,
    encode_split_triples,
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
)

__all__ = [
    "TripleExample",
    "add_reciprocal_triples",
    "build_filter_maps",
    "build_relation_domains",
    "detect_triple_format",
    "encode_split_triples",
    "load_dataset_split",
    "load_triples",
    "load_triples_with_mappings",
    "resolve_split_path",
    "resolve_train_path",
]
