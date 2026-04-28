"""File-format parsers and dataset path resolution.

Two responsibilities:

- **Parsers** for every on-disk file the KGE pipeline knows about: triples
  (TSV / CSV / Prolog), Prolog rules, atom strings, probabilistic-fact
  scores, domain assignments, and per-query depth annotations.
- **Path resolution** for the standard ``<root>/<dataset>/<split>``
  layout used by tkk standalone training, ns reasoner training, and
  DpRL RL training.

The main public-API surface is re-exported from ``kge_kernels.data``;
internal helpers (prefixed with ``_``) are kept here so the
:class:`KnowledgeBase` orchestrator and the few tkk submodules that
need them can compose them without going through the package facade.
"""
from __future__ import annotations

import csv
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple


# ============================================================================
# Triple example + format detection
# ============================================================================


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
    padding_idx: Optional[int] = 0,
) -> Tuple[List[Tuple[int, int, int]], Dict[str, int], Dict[str, int]]:
    """Load triples and assign ``entity2id`` / ``relation2id``.

    When *extra_paths* is provided, entities and relations from all files
    are collected first and sorted alphabetically before assigning IDs.
    This matches torch-ns's ``read_ontology`` ordering, ensuring the same
    entity gets the same embedding index for the same seed.

    ``padding_idx`` controls how ids are laid out:

    - ``0`` (default) — id ``0`` is reserved as the padding sentinel and
      real entities/relations get 1-based ids ``[1..N]``. Matches the
      DpRL / SB3 / ns convention so the same string resolves to the
      same integer id across all consumers.
    - ``None`` — no padding sentinel; ids are dense 0-based ``[0..N-1]``.
      Use for tkk standalone training where the model embeddings have no
      reserved row.
    - any other int — that slot is reserved (currently only ``0`` and
      ``None`` are exercised; out-of-range sentinels are an explicit TODO).

    Returns ``(triples, entity2id, relation2id)`` where triples are
    ``(relation_id, head_id, tail_id)`` tuples (only from *path*, not
    the extra files).

    ``permissive`` is forwarded to :func:`load_triples` and skips Prolog
    rule lines / non-binary atoms instead of raising.
    """
    triples = load_triples(path, format_hint, permissive=permissive)
    # Single shift derived from padding_idx: +1 when slot 0 is reserved,
    # 0 otherwise. ``padding_idx > 0`` is not yet supported by callers and
    # is treated like ``None`` (dense 0-based).
    offset = 1 if padding_idx == 0 else 0

    if extra_paths:
        # Collect ALL entities/relations from all files, sort alphabetically.
        all_triples = list(triples)
        for ep in extra_paths:
            if ep and os.path.isfile(ep):
                all_triples.extend(load_triples(ep, format_hint, permissive=permissive))
        all_entities = sorted({t.head for t in all_triples} | {t.tail for t in all_triples})
        all_relations = sorted({t.relation for t in all_triples})
        entity2id = {e: i + offset for i, e in enumerate(all_entities)}
        relation2id = {r: i + offset for i, r in enumerate(all_relations)}
    else:
        entity2id = {}
        relation2id = {}
        for triple in triples:
            for name in (triple.head, triple.tail):
                if name not in entity2id:
                    entity2id[name] = len(entity2id) + offset
            if triple.relation not in relation2id:
                relation2id[triple.relation] = len(relation2id) + offset

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
        # When the existing map is empty there's no convention to follow
        # — start from 0. With non-empty maps we extend past the current
        # max, which preserves whatever id-offset the caller chose (1-
        # based with reserved 0, dense 0-based, or other).
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


# ============================================================================
# Atom + rule parsing
# ============================================================================


class RuleSpec(NamedTuple):
    """Parsed rule with optional metadata.

    - ``head`` / ``body`` are atom tuples ``(predicate, *args)``.
    - ``name`` is the rule id from a ``rN:`` prefix when present (e.g. "r3"),
      else ``None``.
    - ``weight`` is the float from a ``rN:weight:`` prefix when present,
      else ``1.0`` (hard rule).
    """

    head: Tuple[str, ...]
    body: List[Tuple[str, ...]]
    name: Optional[str] = None
    weight: float = 1.0


def parse_atom_str(
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


def parse_prolog_rule(
    line: str, *, uppercase_args: bool = False,
) -> Optional[RuleSpec]:
    """Parse a single rule line into a :class:`RuleSpec`.

    Returns ``None`` for malformed lines so callers can skip them.
    """
    raw = line.strip()
    if raw.endswith("."):
        raw = raw[:-1]

    # Strip a probabilistic-rule prefix like ``r3:0.72:`` if present and
    # capture the rule id + weight. Heuristic: ``rN:NUM:`` at the start,
    # N digit run, NUM float-or-int. The first colon must be before any
    # '(' (rule prefix never appears inside an atom).
    name: Optional[str] = None
    weight: float = 1.0
    if raw.startswith("r"):
        first_colon = raw.find(":")
        first_paren = raw.find("(")
        if first_colon != -1 and (first_paren == -1 or first_colon < first_paren):
            parts = raw.split(":", 2)
            if len(parts) >= 3 and parts[0][1:].isdigit():
                try:
                    weight = float(parts[1])
                    name = parts[0]
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
        head = parse_atom_str(raw, uppercase_args=uppercase_args)
        return RuleSpec(head=head, body=[], name=name, weight=weight) if head is not None else None

    head = parse_atom_str(head_str, uppercase_args=uppercase_args)
    if head is None:
        return None
    body: List[Tuple[str, ...]] = []
    for atom_str in _split_atoms(body_str):
        atom = parse_atom_str(atom_str, uppercase_args=uppercase_args)
        if atom is None:
            return None
        body.append(atom)
    return RuleSpec(head=head, body=body, name=name, weight=weight)


def load_rules_file(
    path: str, *, uppercase_args: bool = False,
) -> Tuple[List[RuleSpec], "OrderedDict[str, str]"]:
    """Parse a Prolog ``rules.txt``; return rule specs + optional var2domain.

    Each rule is one of:

    - ``head :- body, body, ...`` (standard Prolog).
    - ``body, body, ... -> head`` (DPL / ProbLog left-to-right).
    - ``<rule_id>:<weight>:body, body, ... -> head`` (probabilistic
      family/countries format; the prefix populates ``RuleSpec.name`` /
      ``RuleSpec.weight``).

    A leading line of the form ``var2domain X dom1 Y dom2 ...`` (preamble)
    is parsed into an ordered ``var → domain`` mapping that callers can
    use to type rule variables. The preamble must precede every rule;
    if it appears later it's silently ignored (matches legacy ns
    ``RuleLoader.load`` behavior).

    Empty / ``%``-comment lines are skipped. Lines that fail to parse
    are skipped silently.
    """
    specs: List[RuleSpec] = []
    var_to_domain: "OrderedDict[str, str]" = OrderedDict()
    if not os.path.isfile(path):
        return specs, var_to_domain
    seen_rule = False
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            # var2domain preamble must precede any rule.
            if not seen_rule and stripped.startswith("var2domain"):
                tokens = stripped.split()[1:]
                for i in range(0, len(tokens) - 1, 2):
                    var_to_domain[tokens[i]] = tokens[i + 1]
                continue
            spec = parse_prolog_rule(stripped, uppercase_args=uppercase_args)
            if spec is not None:
                specs.append(spec)
                seen_rule = True
    return specs, var_to_domain


# ============================================================================
# Probabilistic facts file
# ============================================================================


def load_probabilistic_facts_file(
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
            atom = parse_atom_str(parts[0])
            if atom is None or atom in seen:
                continue
            seen.add(atom)
            out.append(atom)
    return out


# ============================================================================
# Domain file parser
# ============================================================================


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


# ============================================================================
# Depth file parser (per-query depth annotations)
# ============================================================================


def load_depth_file(path: str) -> List[int]:
    """Parse a ``<query> <depth>`` sidecar into a flat list of depths.

    File format: one entry per line, ``<query> <depth>`` (depth as the
    last whitespace-separated token); empty / ``%``-comment lines are
    skipped. Lines without a trailing integer get depth ``-1`` so callers
    can distinguish "annotated depth=k" from "no depth given".

    Used by ns's per-test-query depth metric breakdowns.
    """
    depths: List[int] = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.rsplit(None, 1)
            if len(parts) == 2:
                try:
                    depths.append(int(parts[1]))
                    continue
                except ValueError:
                    pass
            depths.append(-1)
    return depths


def _iter_queries_with_depth(path: str):
    """Yield ``(query_string, depth)`` pairs from a sidecar file.

    The file format is one entry per line, ``<query> <depth>`` (depth as
    the last whitespace-separated token), with empty / ``%``-comment
    lines skipped. Lines without a trailing integer yield ``-1`` for
    depth so callers can distinguish "annotated depth=k" from "no depth
    given".

    Internal helper for :class:`KnowledgeBase` query loading. Consumers
    that only need the depth column should use :func:`load_depth_file`.
    """
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.rsplit(None, 1)
            if len(parts) == 2:
                try:
                    yield parts[0], int(parts[1])
                    continue
                except ValueError:
                    pass
            yield stripped, -1


# ============================================================================
# Path resolution for the standard <root>/<dataset>/<split> layout
# ============================================================================


def _resolve_dataset_split_path(
    data_root: str,
    dataset_name: str,
    split_filename: str,
) -> str:
    """Resolve ``<data_root>/<dataset_name>/<split_filename>``.

    Internal helper used by :func:`resolve_train_path` and
    :func:`resolve_split_path`. Not exported.
    """
    path = os.path.join(data_root, dataset_name, split_filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find split '{split_filename}' for dataset "
            f"'{dataset_name}' at {path}"
        )
    return path


def resolve_train_path(
    train_path: Optional[str],
    dataset: Optional[str],
    data_root: str,
    train_split: str,
) -> str:
    """Pick an explicit train path or resolve via dataset/split convention."""
    if train_path:
        return train_path
    if dataset:
        return _resolve_dataset_split_path(data_root, dataset, train_split)
    raise ValueError("Provide either train_path or dataset")


def resolve_split_path(
    *,
    split_name: str,
    explicit_path: Optional[str],
    dataset: Optional[str],
    data_root: str,
    split_filename: Optional[str],
) -> Optional[str]:
    """Resolve an optional eval split; returns ``None`` if not available."""
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(
                f"Provided {split_name} path '{explicit_path}' does not exist"
            )
        return explicit_path
    if dataset and split_filename:
        try:
            return _resolve_dataset_split_path(data_root, dataset, split_filename)
        except FileNotFoundError as err:
            print(
                f"Warning: {split_name} split not found ({err}); continuing without it."
            )
    return None


__all__ = [
    "RuleSpec",
    "TripleExample",
    "detect_triple_format",
    "encode_split_triples",
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
