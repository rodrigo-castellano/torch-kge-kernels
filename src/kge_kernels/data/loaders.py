"""Triple file loaders: TSV / CSV / Prolog auto-detection."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class TripleExample:
    """A single (head, relation, tail) example parsed from a file."""

    head: str
    relation: str
    tail: str


def detect_triple_format(sample_line: str) -> str:
    """Infer triple file format from a sample line.

    Returns one of ``"prolog"``, ``"tsv"``, ``"csv"``, or ``"unknown"``.
    """
    sample = sample_line.strip()
    if not sample:
        return "unknown"
    if "(" in sample and ")" in sample:
        return "prolog"
    if sample.count("\t") >= 2:
        return "tsv"
    if sample.count(",") >= 2:
        return "csv"
    return "unknown"


def _normalize_token(token: str) -> str:
    return token.strip().strip("'\"").strip()


def _parse_prolog_fact(line: str) -> Optional[TripleExample]:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    if raw.endswith("."):
        raw = raw[:-1]
    if "(" not in raw or ")" not in raw:
        return None
    predicate, remainder = raw.split("(", 1)
    args = remainder.split(")", 1)[0]
    terms = [_normalize_token(a) for a in args.split(",") if a.strip()]
    if len(terms) != 2:
        raise ValueError(f"Expected binary predicate, got '{line.strip()}'")
    return TripleExample(
        head=terms[0], relation=_normalize_token(predicate), tail=terms[1]
    )


def _iter_triples_from_file(
    path: str, format_hint: str = "auto"
) -> Iterable[TripleExample]:
    if format_hint == "auto":
        format_hint = "unknown"
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    format_hint = detect_triple_format(line)
                    break

    if format_hint == "prolog":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                triple = _parse_prolog_fact(line)
                if triple is not None:
                    yield triple
        return

    delimiter = "\t" if format_hint == "tsv" else ","
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            cleaned = [_normalize_token(item) for item in row if item.strip()]
            if len(cleaned) < 3:
                continue
            yield TripleExample(
                head=cleaned[0], relation=cleaned[1], tail=cleaned[2]
            )


def load_triples(path: str, format_hint: str = "auto") -> List[TripleExample]:
    """Load all triples from a file."""
    return list(_iter_triples_from_file(path, format_hint=format_hint))


def load_triples_with_mappings(
    path: str,
    format_hint: str = "auto",
) -> Tuple[List[Tuple[int, int, int]], Dict[str, int], Dict[str, int]]:
    """Load triples and assign ``entity2id`` / ``relation2id`` on the fly.

    Returns ``(triples, entity2id, relation2id)`` where triples are
    ``(relation_id, head_id, tail_id)`` tuples.
    """
    triples = load_triples(path, format_hint)
    entity2id: Dict[str, int] = {}
    relation2id: Dict[str, int] = {}
    next_e = 0
    next_r = 0
    triple_ids: List[Tuple[int, int, int]] = []
    for triple in triples:
        if triple.head not in entity2id:
            entity2id[triple.head] = next_e
            next_e += 1
        if triple.tail not in entity2id:
            entity2id[triple.tail] = next_e
            next_e += 1
        if triple.relation not in relation2id:
            relation2id[triple.relation] = next_r
            next_r += 1
        triple_ids.append(
            (
                relation2id[triple.relation],
                entity2id[triple.head],
                entity2id[triple.tail],
            )
        )
    return triple_ids, entity2id, relation2id


def encode_split_triples(
    path: str,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    split_name: str = "split",
) -> List[Tuple[int, int, int]]:
    """Encode a split file using existing vocabularies.

    Unseen entities and relations are added to the vocabularies in-place
    with a warning print. Returns ``(r, h, t)`` tuples.
    """
    raw_triples = load_triples(path)
    encoded: List[Tuple[int, int, int]] = []
    missing_entities: Set[str] = set()
    missing_relations: Set[str] = set()

    for example in raw_triples:
        h = entity2id.get(example.head)
        r = relation2id.get(example.relation)
        t = entity2id.get(example.tail)
        if h is None:
            missing_entities.add(example.head)
        if t is None:
            missing_entities.add(example.tail)
        if r is None:
            missing_relations.add(example.relation)
        if h is None or r is None or t is None:
            continue
        encoded.append((r, h, t))

    if missing_entities or missing_relations:
        next_e = max(entity2id.values()) + 1 if entity2id else 0
        next_r = max(relation2id.values()) + 1 if relation2id else 0
        for ent in sorted(missing_entities):
            entity2id[ent] = next_e
            next_e += 1
        for rel in sorted(missing_relations):
            relation2id[rel] = next_r
            next_r += 1
        if missing_entities:
            print(
                f"  [{split_name}] Added {len(missing_entities)} unseen entities "
                f"(new total: {len(entity2id)})"
            )
        if missing_relations:
            print(
                f"  [{split_name}] Added {len(missing_relations)} unseen relations "
                f"(new total: {len(relation2id)})"
            )
        encoded = [
            (
                relation2id[example.relation],
                entity2id[example.head],
                entity2id[example.tail],
            )
            for example in raw_triples
        ]
    return encoded


__all__ = [
    "TripleExample",
    "detect_triple_format",
    "encode_split_triples",
    "load_triples",
    "load_triples_with_mappings",
]
