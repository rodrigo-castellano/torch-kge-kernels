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


def load_rules_file(
    path: str, *, uppercase_args: bool = False,
) -> List[Tuple[Tuple[str, ...], List[Tuple[str, ...]]]]:
    """Parse a Prolog ``rules.txt`` into ``[(head_tuple, [body_tuple, ...])]``.

    Each rule is ``head :- body1, body2, ...``. Atoms are written as
    ``predicate(arg1, arg2)`` and arguments are usually variables (uppercase
    by Prolog convention) referenced across head and body. Empty lines and
    ``%``-prefixed comments are skipped; lines that fail to parse are
    silently skipped (matches the legacy DpRL loader's permissive behavior).

    Args:
        path: Path to a Prolog rule file.
        uppercase_args: If True, uppercase every atom argument. Used to
            match SB3-parity behavior where rule variables were always
            uppercased even if the source file mixed cases.

    Returns:
        A list of rules; each rule is ``(head, body)`` where ``head`` is a
        ``(predicate, *args)`` tuple and ``body`` is a list of such tuples.
    """
    rules: List[Tuple[Tuple[str, ...], List[Tuple[str, ...]]]] = []
    if not os.path.isfile(path):
        return rules
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            head_body = _parse_prolog_rule(stripped, uppercase_args=uppercase_args)
            if head_body is not None:
                rules.append(head_body)
    return rules


def _parse_prolog_rule(
    line: str, *, uppercase_args: bool = False,
) -> Optional[Tuple[Tuple[str, ...], List[Tuple[str, ...]]]]:
    """Parse a single rule line into ``(head_tuple, body_list)``.

    Supports the three formats found across our datasets:

    - ``head :- body, body, ...`` (standard Prolog).
    - ``body, body, ... -> head`` (DPL / ProbLog "left-to-right").
    - ``<rule_id>:<weight>:body, body, ... -> head`` (probabilistic
      family/countries format; the prefix is stripped before parsing).

    Returns ``None`` for malformed lines so callers can skip them.
    """
    raw = line.strip()
    if raw.endswith("."):
        raw = raw[:-1]

    # Strip a probabilistic-rule prefix like ``r3:0.72:`` if present.
    # Heuristic: ``rN:NUM:`` at the start, where N is a digit run and NUM
    # is float-or-int. The first colon must be before any '(' (rule prefix
    # never appears inside an atom).
    if raw.startswith("r"):
        first_colon = raw.find(":")
        first_paren = raw.find("(")
        if first_colon != -1 and (first_paren == -1 or first_colon < first_paren):
            parts = raw.split(":", 2)
            if len(parts) >= 3 and parts[0][1:].isdigit():
                try:
                    float(parts[1])
                    raw = parts[2].strip()
                except ValueError:
                    pass

    # Format dispatch.
    if ":-" in raw:
        head_str, body_str = raw.split(":-", 1)
    elif "->" in raw:
        body_str, head_str = raw.split("->", 1)
    else:
        # Standalone fact: head only, empty body.
        head = _parse_atom_str(raw, uppercase_args=uppercase_args)
        return (head, []) if head is not None else None

    head = _parse_atom_str(head_str, uppercase_args=uppercase_args)
    if head is None:
        return None
    body: List[Tuple[str, ...]] = []
    for atom_str in _split_atoms(body_str):
        atom = _parse_atom_str(atom_str, uppercase_args=uppercase_args)
        if atom is None:
            return None
        body.append(atom)
    return head, body


def _split_atoms(body_str: str) -> List[str]:
    """Split a Prolog rule body on commas that are NOT inside parens."""
    atoms: List[str] = []
    depth = 0
    current = []
    for ch in body_str:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            atom = "".join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        atoms.append(tail)
    return atoms


def _parse_atom_str(
    atom_str: str, *, uppercase_args: bool = False,
) -> Optional[Tuple[str, ...]]:
    """Parse ``predicate(a, b)`` → ``(predicate, a, b)``. Returns ``None`` on failure.

    Robust to a trailing ``.`` (Prolog statement terminator), which appears
    on standalone fact / query lines like ``aunt(1, 2).``.
    """
    raw = atom_str.strip()
    if raw.endswith("."):
        raw = raw[:-1].rstrip()
    if "(" not in raw or not raw.endswith(")"):
        return None
    predicate, remainder = raw.split("(", 1)
    args_str = remainder[:-1]  # drop trailing ')'
    args = [_normalize_token(a) for a in args_str.split(",") if a.strip()]
    if uppercase_args:
        args = [a.upper() for a in args]
    return (_normalize_token(predicate), *args)


def load_probabilistic_facts(
    path: str,
    *,
    topk_limit: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> List[Tuple[str, ...]]:
    """Parse a ``<fact_repr> <score> [<rank>]`` file into atom tuples.

    Each non-comment line is ``fact score [rank]`` with whitespace
    separators. ``fact_repr`` is parsed via the same per-atom format as
    rule heads (``predicate(arg1, arg2)``). Empty / ``#``-comment lines
    are skipped. Lines that fail to parse — missing score, malformed
    fact, etc. — are silently skipped.

    Args:
        path: Path to the probabilistic-facts file.
        topk_limit: Drop facts whose ``rank`` exceeds this. Negative or
            ``None`` means no rank cap.
        score_threshold: Drop facts with ``score < threshold``. ``None``
            means no threshold.

    Returns:
        Deduplicated list of ``(predicate, *args)`` tuples in input order.
    """
    out: List[Tuple[str, ...]] = []
    if not os.path.isfile(path):
        return out
    seen: Set[Tuple[str, ...]] = set()
    with open(path, "r", encoding="ascii") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                score = float(parts[1])
            except ValueError:
                continue
            rank = None
            if len(parts) >= 3:
                try:
                    rank = int(parts[2])
                except ValueError:
                    rank = None
            if (
                topk_limit is not None
                and topk_limit >= 0
                and rank is not None
                and rank > topk_limit
            ):
                continue
            if score_threshold is not None and score < score_threshold:
                continue
            atom = _parse_atom_str(parts[0])
            if atom is None or atom in seen:
                continue
            seen.add(atom)
            out.append(atom)
    return out


__all__ = [
    "TripleExample",
    "detect_triple_format",
    "encode_split_triples",
    "load_probabilistic_facts",
    "load_rules_file",
    "load_triples",
    "load_triples_with_mappings",
]
