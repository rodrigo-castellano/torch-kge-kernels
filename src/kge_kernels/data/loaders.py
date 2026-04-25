"""Triple file loaders: TSV / CSV / Prolog auto-detection."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


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


def _parse_prolog_fact(line: str, *, permissive: bool = False) -> Optional[TripleExample]:
    raw = line.strip()
    # Empty / comment lines. ``%`` is the standard Prolog comment marker;
    # ``#`` is added so files written in either convention parse cleanly.
    if not raw or raw.startswith("#") or raw.startswith("%"):
        return None
    # Permissive mode: skip Prolog rule lines (``head :- body``) and
    # meta-predicates that DpRL Prolog datasets mix into fact files.
    if permissive and (":-" in raw or "findall" in raw):
        return None
    if raw.endswith("."):
        raw = raw[:-1]
    if "(" not in raw or ")" not in raw:
        return None
    predicate, remainder = raw.split("(", 1)
    args = remainder.split(")", 1)[0]
    terms = [_normalize_token(a) for a in args.split(",") if a.strip()]
    if len(terms) != 2:
        if permissive:
            return None
        raise ValueError(f"Expected binary predicate, got '{line.strip()}'")
    return TripleExample(
        head=terms[0], relation=_normalize_token(predicate), tail=terms[1]
    )


def _iter_triples_from_file(
    path: str, format_hint: str = "auto", *, permissive: bool = False,
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
                triple = _parse_prolog_fact(line, permissive=permissive)
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


def load_triples(
    path: str, format_hint: str = "auto", *, permissive: bool = False,
) -> List[TripleExample]:
    """Load all triples from a file.

    ``permissive=True`` (Prolog only): skip rule lines (``:-``),
    meta-predicates (``findall``), and non-binary atoms instead of raising.
    Use this for DpRL-style Prolog files where facts and rules cohabit.
    """
    return list(_iter_triples_from_file(path, format_hint=format_hint, permissive=permissive))


def load_triples_with_mappings(
    path: str,
    format_hint: str = "auto",
    *,
    extra_paths: Sequence[str] = (),
    permissive: bool = False,
) -> Tuple[List[Tuple[int, int, int]], Dict[str, int], Dict[str, int]]:
    """Load triples and assign ``entity2id`` / ``relation2id``.

    When *extra_paths* is provided, entities and relations from all files
    are collected first and sorted alphabetically before assigning IDs.
    This matches torch-ns's ``read_ontology`` ordering, ensuring the same
    entity gets the same embedding index for the same seed.

    Returns ``(triples, entity2id, relation2id)`` where triples are
    ``(relation_id, head_id, tail_id)`` tuples (only from *path*, not
    the extra files).

    ``permissive`` is forwarded to :func:`load_triples` and skips Prolog
    rule lines / non-binary atoms instead of raising.
    """
    triples = load_triples(path, format_hint, permissive=permissive)

    if extra_paths:
        # Collect ALL entities/relations from all files, sort alphabetically.
        all_triples = list(triples)
        for ep in extra_paths:
            if ep and os.path.isfile(ep):
                all_triples.extend(load_triples(ep, format_hint, permissive=permissive))
        all_entities = sorted({t.head for t in all_triples} | {t.tail for t in all_triples})
        all_relations = sorted({t.relation for t in all_triples})
        entity2id = {e: i for i, e in enumerate(all_entities)}
        relation2id = {r: i for i, r in enumerate(all_relations)}
    else:
        entity2id = {}
        relation2id = {}
        for triple in triples:
            for name in (triple.head, triple.tail):
                if name not in entity2id:
                    entity2id[name] = len(entity2id)
            if triple.relation not in relation2id:
                relation2id[triple.relation] = len(relation2id)

    triple_ids: List[Tuple[int, int, int]] = []
    for triple in triples:
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
