"""Shared dataset handler base class for KGE consumers — and the
generic split-tensor bundle :class:`MaterializedSplit` consumers use to
represent one materialized split.

The same canonical KGE dataset shape — train / valid / test / facts file
quartet, plus an optional ``domain2constants.txt`` — feeds tkk standalone
training, ns reasoner training, and DpRL RL training. Without a shared
loader each repo grew its own orchestration that did the same five steps
in slightly different ways:

1. Resolve ``<root>/<dataset>/<split>`` paths.
2. ``load_triples_with_mappings(train, extras=[valid, test, fact])`` to
   get alphabetical ``entity2id`` / ``relation2id``.
3. ``encode_split_triples`` for valid / test / known so all splits share
   the same id space.
4. ``load_domain_file`` (string mode for pre-id consumers, indexed mode
   for tkk's training pipeline).
5. ``build_filter_maps`` + ``build_relation_domains_from_file`` for
   filtered ranking and domain-aware corruption.

:class:`KGEDatasetHandler` runs all five steps in ``__init__`` and exposes
the result as plain attributes. Consumers inherit and bolt their own
specialized representation on top in their own ``__init__``:

- ns's ``KGCDataHandler`` builds ``KnowledgeBase`` /
  ``EntityType`` / ``Predicate`` (FOL types) over the loaded triples.
- DpRL's ``DataHandler`` converts each ``TripleExample`` to a ``Term``
  and adds rule / query / depth-annotation loading.
- tkk's ``train_model`` consumes the indexed splits + filter maps + domain
  structures directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from os.path import join
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .loaders import (
    TripleExample,
    encode_split_triples,
    load_probabilistic_facts as _load_probabilistic_facts_file,
    load_rules_file,
    load_triples,
    load_triples_with_mappings,
)
from .transforms import (
    build_filter_maps,
    build_relation_domains_from_file,
    filter_queries_by_predicates,
    iter_queries_with_depth,
    load_domain_file,
)


@dataclass
class MaterializedSplit:
    """Generic tensor bundle for one materialized dataset split.

    The shape of ``queries`` is consumer-dependent:

    - SL (ns / tkk): ``[N, 3]`` — single ``(r, h, t)`` triples.
    - RL (DpRL): ``[N, L, max_arity + 1]`` — proof-state padded, with
      ``L`` atom slots per query and ``max_arity + 1`` columns
      (predicate id + argument ids).

    ``labels`` and ``depths`` are always ``[N]``. ``depths`` is ``-1`` for
    queries without a depth annotation.
    """

    queries: torch.LongTensor
    labels: torch.LongTensor
    depths: torch.LongTensor

    def __len__(self) -> int:
        return int(self.queries.shape[0])


@dataclass
class KGEDatasetHandler:
    """Base loader for the standard tkk KGE dataset shape.

    Subclasses pass ``dataset_name``, ``base_path`` and optional file-name
    overrides to ``super().__init__()``; the base class resolves paths,
    loads every split, builds the alphabetical id maps, and (if
    ``domain_file`` is given) builds string-keyed and id-keyed domain
    structures plus per-relation domain restriction.

    Attributes are set by ``__init__``; not constructor arguments. The
    dataclass machinery is used purely for repr / equality, not the
    generated init.
    """

    # ---- vocabulary (alphabetical, stable across runs) -----------------
    entity2id: Dict[str, int] = field(default_factory=dict)
    relation2id: Dict[str, int] = field(default_factory=dict)
    num_entities: int = 0
    num_relations: int = 0
    constants: List[str] = field(default_factory=list)

    # ---- string-form triples (TripleExample) ---------------------------
    train: List[TripleExample] = field(default_factory=list)
    all_valid: List[TripleExample] = field(default_factory=list)
    test: List[TripleExample] = field(default_factory=list)
    known: List[TripleExample] = field(default_factory=list)

    # ---- string-form (relation, head, tail) tuples ---------------------
    train_facts: List[Tuple[str, str, str]] = field(default_factory=list)
    all_valid_facts: List[Tuple[str, str, str]] = field(default_factory=list)
    valid_facts: List[Tuple[str, str, str]] = field(default_factory=list)
    test_facts: List[Tuple[str, str, str]] = field(default_factory=list)
    known_facts: List[Tuple[str, str, str]] = field(default_factory=list)
    train_facts_set: Set[Tuple[str, str, str]] = field(default_factory=set)
    valid_facts_set: Set[Tuple[str, str, str]] = field(default_factory=set)
    test_facts_set: Set[Tuple[str, str, str]] = field(default_factory=set)
    known_facts_set: Set[Tuple[str, str, str]] = field(default_factory=set)
    train_known_facts_set: Set[Tuple[str, str, str]] = field(default_factory=set)
    ground_facts_set: Set[Tuple[str, str, str]] = field(default_factory=set)

    # ---- integer-indexed (r, h, t) triples -----------------------------
    train_idx: List[Tuple[int, int, int]] = field(default_factory=list)
    valid_idx: List[Tuple[int, int, int]] = field(default_factory=list)
    test_idx: List[Tuple[int, int, int]] = field(default_factory=list)
    known_idx: List[Tuple[int, int, int]] = field(default_factory=list)
    ground_facts_idx_set: Set[Tuple[int, int, int]] = field(default_factory=set)

    # ---- domains (str-keyed and id-keyed views) ------------------------
    domain2entities: Dict[str, List[str]] = field(default_factory=dict)
    entity2domain: Dict[str, str] = field(default_factory=dict)
    domain2idx: Dict[str, List[int]] = field(default_factory=dict)
    entity2domain_idx: Dict[int, str] = field(default_factory=dict)
    head_domain: Optional[Dict[int, Set[int]]] = None
    tail_domain: Optional[Dict[int, Set[int]]] = None
    use_domain_eval: bool = False

    # ---- filter maps (filtered ranking) --------------------------------
    head_filter: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)
    tail_filter: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        base_path: str = "data",
        *,
        train_file: str = "train.txt",
        valid_file: str = "valid.txt",
        test_file: str = "test.txt",
        fact_file: Optional[str] = "facts.txt",
        domain_file: Optional[str] = None,
        permissive: bool = False,
        valid_size: Optional[int] = None,
    ) -> None:
        self._reset()
        if dataset_name is None:
            # Lazy-init: subclass will call ``self.load_dataset(...)`` (or
            # equivalently :meth:`_load_dataset`) after its own setup.
            # Used by DpRL's ``DataHandler`` which separates container
            # init from file I/O.
            return
        self._load_dataset(
            dataset_name=dataset_name, base_path=base_path,
            train_file=train_file, valid_file=valid_file, test_file=test_file,
            fact_file=fact_file, domain_file=domain_file,
            permissive=permissive, valid_size=valid_size,
        )

    def _reset(self) -> None:
        """Initialize all bundle attributes to empty defaults.

        Called from ``__init__`` so lazy-loading subclasses can construct
        ``super().__init__()`` and still see well-defined empty containers
        before they trigger an actual load.
        """
        self.entity2id = {}
        self.relation2id = {}
        self.num_entities = 0
        self.num_relations = 0
        self.constants = []
        self.train = []
        self.all_valid = []
        self.test = []
        self.known = []
        self.train_facts = []
        self.all_valid_facts = []
        self.valid_facts = []
        self.test_facts = []
        self.known_facts = []
        self.train_facts_set = set()
        self.valid_facts_set = set()
        self.test_facts_set = set()
        self.known_facts_set = set()
        self.train_known_facts_set = set()
        self.ground_facts_set = set()
        self.train_idx = []
        self.valid_idx = []
        self.test_idx = []
        self.known_idx = []
        self.ground_facts_idx_set = set()
        self.domain2entities = {}
        self.entity2domain = {}
        self.domain2idx = {}
        self.entity2domain_idx = {}
        self.head_domain = None
        self.tail_domain = None
        self.use_domain_eval = False
        self.head_filter = {}
        self.tail_filter = {}

    def _load_dataset(
        self,
        dataset_name: str,
        base_path: str = "data",
        *,
        train_file: str = "train.txt",
        valid_file: str = "valid.txt",
        test_file: str = "test.txt",
        fact_file: Optional[str] = "facts.txt",
        domain_file: Optional[str] = None,
        permissive: bool = False,
        valid_size: Optional[int] = None,
    ) -> None:
        """Run the full canonical loading pipeline.

        Public so lazy-init subclasses (e.g. DpRL's ``DataHandler``) can
        invoke it from their own ``load_dataset()`` after setting up
        subclass-specific containers.
        """
        # Path resolution.
        base = join(base_path, dataset_name)
        train_path = join(base, train_file)
        valid_path = join(base, valid_file)
        test_path = join(base, test_file)
        fact_path = join(base, fact_file) if fact_file else None

        # ── 1. Vocabulary + indexed triples (alphabetical id assignment) ──
        extras = [
            p for p in (valid_path, test_path, fact_path)
            if p and os.path.isfile(p)
        ]
        train_idx, ent2id, rel2id = load_triples_with_mappings(
            train_path, extra_paths=extras, permissive=permissive,
        )
        self.entity2id = ent2id
        self.relation2id = rel2id
        self.num_entities = len(ent2id)
        self.num_relations = len(rel2id)
        self.constants = sorted(ent2id.keys())
        self.train_idx = train_idx
        self.valid_idx = (
            encode_split_triples(valid_path, ent2id, rel2id, "valid")
            if os.path.isfile(valid_path) else []
        )
        self.test_idx = (
            encode_split_triples(test_path, ent2id, rel2id, "test")
            if os.path.isfile(test_path) else []
        )
        self.known_idx = (
            encode_split_triples(fact_path, ent2id, rel2id, "facts")
            if fact_path and os.path.isfile(fact_path) else []
        )

        # ── 2. String-form triples (TripleExample + (r,h,t) tuples) ──
        def _load_str(path: Optional[str]) -> List[TripleExample]:
            if path is None or not os.path.isfile(path):
                return []
            return load_triples(path, permissive=permissive)

        def _to_tuples(triples: List[TripleExample]) -> List[Tuple[str, str, str]]:
            return [(t.relation, t.head, t.tail) for t in triples]

        self.train = _load_str(train_path)
        self.all_valid = _load_str(valid_path)
        self.test = _load_str(test_path)
        self.known = _load_str(fact_path) if fact_path else []

        self.train_facts = _to_tuples(self.train)
        self.all_valid_facts = _to_tuples(self.all_valid)
        self.test_facts = _to_tuples(self.test)
        self.known_facts = _to_tuples(self.known)
        self.valid_facts = (
            self.all_valid_facts[:valid_size]
            if valid_size is not None else self.all_valid_facts
        )

        self.train_facts_set = set(self.train_facts)
        self.valid_facts_set = set(self.valid_facts)
        self.test_facts_set = set(self.test_facts)
        self.known_facts_set = set(self.known_facts)
        self.train_known_facts_set = set(self.train_facts + self.known_facts)
        self.ground_facts_set = set(
            self.train_facts + self.all_valid_facts +
            self.test_facts + self.known_facts
        )
        self.ground_facts_idx_set = set(
            self.train_idx + self.valid_idx + self.test_idx + self.known_idx
        )

        # ── 3. Filter maps (for filtered MRR / Hits@K) ──
        self.head_filter, self.tail_filter = build_filter_maps(
            self.train_idx, self.valid_idx, self.test_idx, self.known_idx,
        )

        # ── 4. Domain mappings + per-relation domain restriction ──
        if domain_file is not None:
            domain_path = join(base, domain_file)
            # String-mode: keys are entity / domain names. Used by
            # consumers that need this BEFORE any further id remapping
            # (e.g. ns's KnowledgeBase construction).
            d2e_str, e2d_str = load_domain_file(domain_path, entity2id=None)
            self.domain2entities = {k: list(v) for k, v in d2e_str.items()}
            self.entity2domain = dict(e2d_str)
            # Indexed-mode: keys are entity ids. Used by the sampler /
            # filtered eval. Already aligned with self.entity2id.
            self.domain2idx, self.entity2domain_idx = load_domain_file(
                domain_path, entity2id=ent2id,
            )
            if self.domain2idx:
                self.use_domain_eval = True
                self.head_domain, self.tail_domain = build_relation_domains_from_file(
                    self.train_idx + self.valid_idx + self.test_idx,
                    self.entity2domain_idx, self.domain2idx,
                )

    # ---- factory hooks (subclass overrides for typed objects) ---------

    def make_atom(self, atom_tuple: Tuple[str, ...]) -> Any:
        """Convert a ``(predicate, *args)`` tuple to a typed atom.

        Default: returns the tuple unchanged. Subclasses override to
        construct their domain-specific atom type — e.g. DpRL returns a
        ``Term(predicate, args)``.
        """
        return atom_tuple

    def make_rule(self, head: Any, body: List[Any]) -> Any:
        """Combine a head atom + body atoms into a typed rule.

        Default: returns ``(head, body)``. Subclasses override to construct
        their domain-specific rule type.
        """
        return (head, body)

    # ---- generic loaders that compose primitives + factory hooks ------

    def load_facts(
        self,
        *,
        filepath: Optional[str] = None,
        skip_predicate_prefixes: Tuple[str, ...] = (),
    ) -> List[Any]:
        """Convert background facts to typed atoms via :meth:`make_atom`.

        With no ``filepath``: pulls from ``self.known`` (already loaded
        by :meth:`_load_dataset`). With explicit ``filepath``: re-parses
        that file via :func:`load_triples` (Prolog, permissive). Triples
        whose relation starts with any prefix in ``skip_predicate_prefixes``
        are dropped — useful for dataset-specific preamble lines like
        DpRL's ``one_step``.
        """
        if filepath is None:
            triples = self.known
        else:
            triples = load_triples(filepath, format_hint="prolog", permissive=True)
        out: List[Any] = []
        for triple in triples:
            if any(triple.relation.startswith(p) for p in skip_predicate_prefixes):
                continue
            out.append(self.make_atom((triple.relation, triple.head, triple.tail)))
        return out

    def load_rules(
        self,
        filepath: str,
        *,
        uppercase_args: bool = False,
    ) -> List[Any]:
        """Load rules and convert via :meth:`make_atom` + :meth:`make_rule`.

        File parsing — Prolog ``:-`` and ``->`` formats, optional
        ``rN:weight:`` prefix, atom tokenization — lives in
        :func:`load_rules_file`. The factory hooks decide the typed
        output: default ``(head_tuple, body_list)`` tuple; DpRL returns
        ``Rule(Term, [Term, ...])``.
        """
        out: List[Any] = []
        for head_tuple, body_list in load_rules_file(filepath, uppercase_args=uppercase_args):
            head = self.make_atom(head_tuple)
            body = [self.make_atom(b) for b in body_list]
            out.append(self.make_rule(head, body))
        return out

    def load_queries_with_depth(
        self,
        filepath: str,
        *,
        limit: Optional[int] = None,
        depth_filter: Optional[Set[int]] = None,
    ) -> Tuple[List[Any], List[int]]:
        """Load queries with depth annotations; return ``(atoms, depths)``.

        Wraps :func:`iter_queries_with_depth` (line iterator) and applies
        the depth filter and atom-count limit. Each query string is
        parsed into a ``(predicate, *args)`` tuple via the same atom
        parser used by rule loading (lines that fail to parse are
        silently skipped). The tuple is then routed through
        :meth:`make_atom` so subclasses get their typed atom.
        """
        from .loaders import _parse_atom_str

        atoms: List[Any] = []
        depths: List[int] = []
        for query_str, query_depth in iter_queries_with_depth(filepath):
            if depth_filter is not None and query_depth not in depth_filter:
                continue
            atom_tuple = _parse_atom_str(query_str)
            if atom_tuple is None:
                continue
            atoms.append(self.make_atom(atom_tuple))
            depths.append(query_depth)
            if limit is not None and len(atoms) >= limit:
                break
        return atoms, depths

    def load_probabilistic_facts(
        self,
        dataset_name: str,
        *,
        base_dir: str,
        patterns: Optional[List[str]] = None,
        topk_limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Any]:
        """Discover the probabilistic-facts file for ``dataset_name`` and load it.

        Searches ``base_dir`` for the first file that matches one of
        ``patterns`` (defaulting to a few common conventions). Returns an
        empty list if no file is found. Atom tuples from
        :func:`load_probabilistic_facts` are passed through
        :meth:`make_atom`.
        """
        if patterns is None:
            patterns = [
                f"soft_top_{dataset_name}.txt",
                f"kge_top_{dataset_name}_facts.txt",
                f"kge_top5_{dataset_name}_facts.txt",
                f"kge_top_{dataset_name}.txt",
            ]
        path = next(
            (
                os.path.join(base_dir, p)
                for p in patterns
                if os.path.isfile(os.path.join(base_dir, p))
            ),
            None,
        )
        if path is None:
            return []
        tuples = _load_probabilistic_facts_file(
            path, topk_limit=topk_limit, score_threshold=score_threshold,
        )
        return [self.make_atom(t) for t in tuples]

    def filter_queries_by_rule_heads(
        self,
        query_tuples: List[Tuple[str, ...]],
        rule_head_predicates: Set[str],
    ) -> Tuple[List[Tuple[str, ...]], List[int]]:
        """Drop queries whose predicate is not a rule head.

        Thin convenience over :func:`filter_queries_by_predicates` —
        callers re-align their parallel arrays (atoms, depths, labels)
        using the returned ``kept_indices``.
        """
        return filter_queries_by_predicates(query_tuples, rule_head_predicates)

    def discover_vocabulary(
        self,
        *,
        rules_str: List[Tuple[Tuple[str, ...], List[Tuple[str, ...]]]] = (),
        queries_str: List[Tuple[str, ...]] = (),
        constants: Optional[Set[str]] = None,
        predicates: Optional[Set[str]] = None,
    ) -> Tuple[Set[str], Set[str]]:
        """Populate ``constants`` / ``predicates`` Sets from rule + query atoms.

        Seeds from base's ``entity2id`` / ``relation2id`` (already filled
        from facts by :meth:`_load_dataset`) and extends with rule heads,
        rule bodies, and query atoms — those don't flow through the
        base's id-assignment path so they're added explicitly.

        Args:
            rules_str: List of ``(head_tuple, body_list)`` rule tuples.
            queries_str: Flat list of query ``(predicate, *args)`` tuples
                (concatenate all splits before passing).
            constants: Optional Set to update in place. If ``None``, a
                fresh set is created.
            predicates: Same shape as ``constants``.

        Returns:
            ``(constants, predicates)`` with seed + rule + query vocab.
        """
        if constants is None:
            constants = set()
        if predicates is None:
            predicates = set()
        constants.update(self.entity2id.keys())
        predicates.update(self.relation2id.keys())
        for (head_pred, *_head_args), body in rules_str:
            predicates.add(head_pred)
            for body_pred, *_body_args in body:
                predicates.add(body_pred)
        for pred, *args in queries_str:
            predicates.add(pred)
            constants.update(args)
        return constants, predicates

    # ---- convenience accessors ----------------------------------------

    def build_sampler(
        self,
        *,
        default_mode: str = "both",
        seed: int = 0,
        device: Optional[Any] = None,
        order_negatives: bool = False,
    ) -> "Sampler":
        """Construct a tkk ``Sampler`` from the loaded id space + filters + domains.

        Single source of truth for sampler instantiation across consumers.
        Reads only attributes already populated by :meth:`_load_dataset`:
        ``ground_facts_idx_set`` (filter index), ``num_entities`` /
        ``num_relations`` (id space), and the optional indexed-domain
        structures ``domain2idx`` / ``entity2domain_idx``.

        Args:
            default_mode: Fallback corruption mode for ``Sampler.corrupt``
                when no per-call mode is given.
            seed: RNG seed for the underlying torch generator.
            device: Optional device override; defaults to CPU.
            order_negatives: Forwarded to ``Sampler.from_data``.

        Returns:
            A configured :class:`kge_kernels.scoring.Sampler`.
        """
        import torch

        from ..scoring import Sampler

        device = device or torch.device("cpu")
        if self.ground_facts_idx_set:
            all_known = torch.tensor(
                sorted(self.ground_facts_idx_set), dtype=torch.long,
            )
        else:
            all_known = torch.empty((0, 3), dtype=torch.long)
        return Sampler.from_data(
            all_known_triples_idx=all_known,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            device=device,
            default_mode=default_mode,
            seed=seed,
            domain2idx=(self.domain2idx or None),
            entity2domain=(self.entity2domain_idx or None),
            order_negatives=order_negatives,
            min_entity_idx=0,
        )

    def split_idx(self, split: str) -> List[Tuple[int, int, int]]:
        """Return integer-indexed triples for ``"train" | "valid" | "test" | "known"``."""
        if split == "train":
            return self.train_idx
        if split == "valid":
            return self.valid_idx
        if split == "test":
            return self.test_idx
        if split == "known":
            return self.known_idx
        raise ValueError(f"Unknown split: {split!r}")

    def split_facts(self, split: str) -> List[Tuple[str, str, str]]:
        """Return string-form facts for ``"train" | "valid" | "test" | "known"``."""
        if split == "train":
            return self.train_facts
        if split == "valid":
            return self.valid_facts
        if split == "test":
            return self.test_facts
        if split == "known":
            return self.known_facts
        raise ValueError(f"Unknown split: {split!r}")


__all__ = ["KGEDatasetHandler", "MaterializedSplit"]
