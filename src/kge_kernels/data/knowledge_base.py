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

:class:`KnowledgeBase` runs all five steps in ``__init__`` and exposes
the result as plain attributes. Consumers inherit and bolt their own
specialized representation on top in their own ``__init__``:

- ns's ``KGCDataHandler`` builds ``KnowledgeBase`` /
  ``EntityType`` / ``Predicate`` (FOL types) over the loaded triples.
- DpRL's ``DataHandler`` converts each ``TripleExample`` to a ``Term``
  and adds rule / query / depth-annotation loading.
- tkk's ``pipeline`` consumes the indexed splits + filter maps + domain
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
    _iter_queries_with_depth,
    encode_split_triples,
    load_domain_file,
    load_probabilistic_facts as _load_probabilistic_facts_file,
    load_rules_file,
    load_triples,
    load_triples_with_mappings,
    parse_atom_str,
)
from .transforms import (
    build_filter_maps,
    build_relation_domains_from_file,
    filter_queries_by_predicates,
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
class KnowledgeBase:
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
    # ``padding_idx`` is the single source of truth for the id-space
    # convention; defaults to ``0`` (id 0 reserved, real ids ``1..N``)
    # to match DpRL / SB3 / ns. Pass ``padding_idx=None`` to
    # ``__init__`` for dense 0-based ids ``0..N-1`` (used by tkk's
    # standalone training pipeline).
    entity2id: Dict[str, int] = field(default_factory=dict)
    relation2id: Dict[str, int] = field(default_factory=dict)
    num_entities: int = 0
    num_relations: int = 0
    padding_idx: int = 0
    # ``constants`` / ``predicates`` are alphabetical sorted entity / relation
    # lists for deterministic iteration. ``constants_set`` /
    # ``predicates_set`` are frozensets for O(1) membership. After
    # ``_load_dataset`` only the fact-vocabulary is in here;
    # ``discover_vocabulary`` extends both with rule + query atoms.
    constants: List[str] = field(default_factory=list)
    constants_set: frozenset = field(default_factory=frozenset)
    predicates: List[str] = field(default_factory=list)
    predicates_set: frozenset = field(default_factory=frozenset)
    # Typed-data placeholders populated by load_facts / load_rules /
    # load_queries_with_depth (kept here so subclass __init__ doesn't
    # have to redeclare them).
    rule_names: List[Optional[str]] = field(default_factory=list)
    rule_weights: List[float] = field(default_factory=list)
    rules_str: List[Tuple[Tuple[str, ...], List[Tuple[str, ...]]]] = field(default_factory=list)
    facts_str: List[Tuple[str, ...]] = field(default_factory=list)
    train_queries_str: List[Tuple[str, ...]] = field(default_factory=list)
    valid_queries_str: List[Tuple[str, ...]] = field(default_factory=list)
    test_queries_str: List[Tuple[str, ...]] = field(default_factory=list)
    train_depths: List[int] = field(default_factory=list)
    valid_depths: List[int] = field(default_factory=list)
    test_depths: List[int] = field(default_factory=list)
    train_labels: List[int] = field(default_factory=list)
    valid_labels: List[int] = field(default_factory=list)
    test_labels: List[int] = field(default_factory=list)

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
        padding_idx: Optional[int] = 0,
        use_reciprocal: bool = False,
    ) -> None:
        self._reset()
        self.padding_idx = padding_idx if padding_idx is not None else 0
        # ``padding_idx=None`` means "no padding sentinel; dense 0-based
        # ids". We still expose ``self.padding_idx`` (defaulted to 0 in
        # that case) so downstream code can read a single attribute, but
        # internally the loader uses ``_padding_idx`` to drive the shift.
        self._padding_idx = padding_idx
        self.use_reciprocal = use_reciprocal
        self.num_relations_orig = 0   # populated after load if reciprocal applied
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
        if use_reciprocal:
            self._apply_reciprocal_relations()

    def _apply_reciprocal_relations(self) -> None:
        """Augment training triples + relation vocab with reciprocal relations.

        For each ``(r, h, t)`` adds ``(r + num_relations_orig, t, h)``
        where the inverse relation is named ``<r>__inv``. Updates
        ``relation2id``, ``predicates``, ``num_relations``, and
        ``train_idx``. Validation/test are NOT augmented (only training
        sees the reciprocal pairs).
        """
        from .transforms import add_reciprocal_triples
        self.num_relations_orig = self.num_relations
        new_train_idx, new_relation2id, new_num_relations = add_reciprocal_triples(
            self.train_idx, self.relation2id, inv_suffix="__inv",
        )
        self.train_idx = new_train_idx
        self.relation2id = new_relation2id
        self.num_relations = new_num_relations
        self.predicates = sorted(new_relation2id.keys())
        self.predicates_set = frozenset(self.predicates)

    def _reset(self) -> None:
        """Initialize all bundle attributes to empty defaults.

        Called from ``__init__`` so lazy-loading subclasses can construct
        ``super().__init__()`` and still see well-defined empty containers
        before they trigger an actual load. ``padding_idx`` is set by
        ``__init__`` after ``_reset`` and is not touched here.
        """
        self.entity2id = {}
        self.relation2id = {}
        self.num_entities = 0
        self.num_relations = 0
        self.constants = []
        self.constants_set = frozenset()
        self.predicates = []
        self.predicates_set = frozenset()
        self.rule_names = []
        self.rule_weights = []
        self.rules_str = []
        self.facts_str = []
        self.train_queries_str = []
        self.valid_queries_str = []
        self.test_queries_str = []
        self.train_depths = []
        self.valid_depths = []
        self.test_depths = []
        self.train_labels = []
        self.valid_labels = []
        self.test_labels = []
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
            padding_idx=getattr(self, "_padding_idx", 0),
        )
        self.entity2id = ent2id
        self.relation2id = rel2id
        self.num_entities = len(ent2id)
        self.num_relations = len(rel2id)
        self.constants = sorted(ent2id.keys())
        self.constants_set = frozenset(self.constants)
        # Predicates list / frozenset start with relation2id keys; extended
        # by discover_vocabulary later if rule / query loading adds more.
        self.predicates = sorted(rel2id.keys())
        self.predicates_set = frozenset(self.predicates)
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
        # Note: NOT auto-detected by file existence. Some datasets (family)
        # ship a ``domain2constants.txt`` but downstream parity references
        # (SB3) don't use it; auto-detecting would diverge from those
        # consumers. Callers must opt in explicitly via ``domain_file=``.
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

    # ---- typed-data loaders -------------------------------------------

    def load_facts(
        self,
        *,
        filepath: Optional[str] = None,
        skip_predicate_prefixes: Tuple[str, ...] = (),
    ) -> List[Tuple[str, ...]]:
        """Return background facts as ``(predicate, head, tail)`` tuples.

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
        return [
            (triple.relation, triple.head, triple.tail)
            for triple in triples
            if not any(triple.relation.startswith(p) for p in skip_predicate_prefixes)
        ]

    def load_rules(
        self,
        filepath: str,
        *,
        uppercase_args: bool = False,
        limit: Optional[int] = None,
        force_hard_rules: bool = False,
    ) -> List[Tuple[Tuple[str, ...], List[Tuple[str, ...]]]]:
        """Load rules into ``self.rules_str`` + parallel name/weight lists.

        Populates these handler attributes (in order):

        - ``self.rules_str: List[(head_tuple, body_list)]``
        - ``self.rule_names: List[Optional[str]]`` — rule id from
          ``rN:weight:`` prefix when present, else ``None``.
        - ``self.rule_weights: List[float]`` — float from prefix, or 1.0.
        - ``self.rule_var_domains: OrderedDict[str, str]`` — variable
          domains from ``var2domain`` preamble (empty if no preamble).

        Args:
            filepath: Path to the rule file.
            uppercase_args: Pass through to the parser (uppercase every
                atom argument). SB3-parity in DpRL.
            limit: Stop reading after ``limit`` rules (matches legacy ns
                ``RuleLoader.load(filepath, num_rules)``).
            force_hard_rules: Override every parsed weight with 1.0.

        Returns:
            ``rules_str`` — list of ``(head_tuple, body_list)`` pairs.
            Same shape as :attr:`rules_str`.
        """
        specs, var_to_domain = load_rules_file(
            filepath, uppercase_args=uppercase_args,
        )
        if limit is not None:
            specs = specs[:limit]
        rules_str: List[Tuple[Tuple[str, ...], List[Tuple[str, ...]]]] = []
        rule_names: List[Optional[str]] = []
        rule_weights: List[float] = []
        for spec in specs:
            rules_str.append((spec.head, spec.body))
            rule_names.append(spec.name)
            rule_weights.append(1.0 if force_hard_rules else spec.weight)
        self.rules_str = rules_str
        self.rule_names = rule_names
        self.rule_weights = rule_weights
        self.rule_var_domains = var_to_domain
        return rules_str

    def load_queries_with_depth(
        self,
        filepath: str,
        *,
        limit: Optional[int] = None,
        depth_filter: Optional[Set[int]] = None,
    ) -> Tuple[List[Tuple[str, ...]], List[int]]:
        """Load queries with depth annotations; return ``(atoms, depths)``.

        Iterates the ``<query> <depth>`` sidecar file, applies the depth
        filter and atom-count limit, and parses each query string into a
        ``(predicate, *args)`` tuple. Lines that fail to parse are
        silently skipped.
        """
        atoms: List[Tuple[str, ...]] = []
        depths: List[int] = []
        for query_str, query_depth in _iter_queries_with_depth(filepath):
            if depth_filter is not None and query_depth not in depth_filter:
                continue
            atom_tuple = parse_atom_str(query_str)
            if atom_tuple is None:
                continue
            atoms.append(atom_tuple)
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
    ) -> List[Tuple[str, ...]]:
        """Discover the probabilistic-facts file for ``dataset_name`` and load it.

        Searches ``base_dir`` for the first file that matches one of
        ``patterns`` (defaulting to a few common conventions). Returns an
        empty list if no file is found.
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
        return _load_probabilistic_facts_file(
            path, topk_limit=topk_limit, score_threshold=score_threshold,
        )

    def discover_vocabulary(
        self,
        *,
        rules_str: List[Tuple[Tuple[str, ...], List[Tuple[str, ...]]]] = (),
        queries_str: List[Tuple[str, ...]] = (),
        constants: Optional[Set[str]] = None,
        predicates: Optional[Set[str]] = None,
    ) -> Tuple[Set[str], Set[str]]:
        """Populate ``constants_set`` / ``predicates_set`` from rule + query atoms.

        Seeds from base's ``entity2id`` / ``relation2id`` (already filled
        from facts by :meth:`_load_dataset`) and extends with rule heads,
        rule bodies, and query atoms — those don't flow through the
        base's id-assignment path so they're added explicitly.

        After this method runs, the canonical alphabetical iteration is
        ``self.constants`` / ``sorted(self.predicates_set)``; the O(1)
        membership views are ``self.constants_set`` / ``self.predicates_set``.

        Args:
            rules_str: List of ``(head_tuple, body_list)`` rule tuples.
            queries_str: Flat list of query ``(predicate, *args)`` tuples
                (concatenate all splits before passing).
            constants: Optional Set to update in place. If ``None``, a
                fresh set is created (seeded from ``entity2id``).
            predicates: Same shape as ``constants`` (seeded from ``relation2id``).

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
        # Finalize the canonical handler views (alphabetical List + frozenset).
        self.constants = sorted(constants)
        self.constants_set = frozenset(constants)
        self.predicates = sorted(predicates)
        self.predicates_set = frozenset(predicates)
        return constants, predicates

    def load_dataset(
        self,
        dataset_name: str,
        base_path: str = "data",
        *,
        train_file: str = "train.txt",
        valid_file: str = "valid.txt",
        test_file: str = "test.txt",
        fact_file: Optional[str] = "facts.txt",
        rules_file: Optional[str] = None,
        domain_file: Optional[str] = None,
        permissive: bool = False,
        valid_size: Optional[int] = None,
        # Typed-data switches (opt-in for consumers that want typed atoms/rules)
        load_typed_facts: bool = False,
        skip_predicate_prefixes: Tuple[str, ...] = (),
        load_typed_rules: bool = False,
        uppercase_rule_args: bool = False,
        rule_limit: Optional[int] = None,
        force_hard_rules: bool = False,
        load_typed_queries: bool = False,
        n_train_queries: Optional[int] = None,
        n_eval_queries: Optional[int] = None,
        n_test_queries: Optional[int] = None,
        train_depth: Optional[Set[int]] = None,
        valid_depth: Optional[Set[int]] = None,
        test_depth: Optional[Set[int]] = None,
        load_depth_info: bool = True,
        # Filter / vocab / KG / materialize
        filter_queries_by_rule_heads: bool = False,
        discover_vocab: bool = True,
        build_kg: bool = True,
        default_domain_name: str = "default",
        materialize: bool = False,
        sort_data: bool = True,
        device: Optional[Any] = None,
    ) -> None:
        """One canonical KGE dataset load for all consumers.

        Pipeline (each typed-data step is opt-in):

        1. Low-level: vocabulary, splits, filter maps, domains via
           :meth:`_load_dataset` (always).
        2. Typed facts via :meth:`load_facts` (if ``load_typed_facts``).
        3. Typed rules via :meth:`load_rules` (if ``load_typed_rules`` and
           ``rules_file`` is provided).
        4. Typed queries with depth via :meth:`load_queries_with_depth`
           for train / valid / test (if ``load_typed_queries``).
        5. Filter train queries by rule heads (if requested + rules loaded).
        6. ``sort_data`` — alphabetical sort of ``facts_str`` + ``rules_str``
           (and its parallel ``rule_names`` / ``rule_weights`` arrays) +
           per-split ``{split}_queries_str`` (with parallel labels/depths).
        7. Vocabulary discovery — populate ``constants`` /
           ``constants_set`` / ``predicates_set`` (if ``discover_vocab``).
        8. ``build_kg`` — default-domain catch-all + id-view rebuild +
           per-relation domain restriction (if ``build_kg`` is True).
        9. ``materialize`` — tensor versions of facts/rules/queries
           (if ``materialize`` is True).

        Subclasses override this method only to add consumer-specific
        extras (DpRL: probabilistic-facts merge, IndexManager-shifted
        materialize). The default flag values match the SL consumer
        (just the canonical pipeline, no typed lists, no materialize);
        DpRL passes ``load_typed_*=True`` and ``materialize=False``
        (uses its own IndexManager-aware materialize override).
        """
        # 1. Low-level loader.
        self._load_dataset(
            dataset_name=dataset_name, base_path=base_path,
            train_file=train_file, valid_file=valid_file, test_file=test_file,
            fact_file=fact_file, domain_file=domain_file,
            permissive=permissive, valid_size=valid_size,
        )
        dataset_path = join(base_path, dataset_name)

        # 2. Typed facts (str form populated alongside).
        if load_typed_facts:
            self.facts = self.load_facts(skip_predicate_prefixes=skip_predicate_prefixes)
            # Populate facts_str alongside the typed list — useful for
            # downstream materialize() and consumers that want str-form.
            self.facts_str = [
                (t.relation, t.head, t.tail) for t in self.known
                if not any(t.relation.startswith(p) for p in skip_predicate_prefixes)
            ]

        # 3. Typed rules.
        if load_typed_rules and rules_file:
            rules_path = join(dataset_path, rules_file)
            if os.path.isfile(rules_path):
                self.rules = self.load_rules(
                    rules_path,
                    uppercase_args=uppercase_rule_args,
                    limit=rule_limit,
                    force_hard_rules=force_hard_rules,
                )

        # 4. Typed queries with depth (per split).
        if load_typed_queries:
            train_limit = None if filter_queries_by_rule_heads else n_train_queries
            for split, fname, limit, depth_set in (
                ("train", train_file, train_limit, train_depth),
                ("valid", valid_file, n_eval_queries, valid_depth),
                ("test", test_file, n_test_queries, test_depth),
            ):
                full_path = join(dataset_path, fname)
                if os.path.isfile(full_path):
                    queries, depths = self.load_queries_with_depth(
                        full_path, limit=limit, depth_filter=depth_set,
                    )
                    if not load_depth_info:
                        depths = [-1] * len(queries)
                else:
                    queries, depths = [], []
                # ``load_queries_with_depth`` already returns parsed
                # ``(predicate, *args)`` tuples in file order, so the
                # str-form list is the same payload — no second pass.
                setattr(self, f"{split}_queries", queries)
                setattr(self, f"{split}_queries_str", list(queries))
                setattr(self, f"{split}_depths", list(depths))
                setattr(self, f"{split}_labels", [1] * len(queries))

        # 5. Filter queries by rule heads.
        if filter_queries_by_rule_heads and self.rules_str:
            rule_head_preds = {head[0] for head, _body in self.rules_str}
            filtered_str, kept = filter_queries_by_predicates(
                self.train_queries_str, rule_head_preds,
            )
            if len(kept) < len(self.train_queries_str):
                self.train_queries_str = filtered_str
                self.train_depths = [self.train_depths[i] for i in kept]
                self.train_labels = [self.train_labels[i] for i in kept]
                if hasattr(self, "train_queries") and self.train_queries:
                    self.train_queries = [self.train_queries[i] for i in kept]
            # Always re-apply ``n_train_queries`` after the filter — even
            # when the filter removed nothing, the loader pre-bypassed
            # the limit (so it could see all candidates) and we must trim
            # back here to honor the cap. Matches DpRL's pre-Phase-A
            # behavior where SB3 parity expects a hard limit.
            if n_train_queries is not None:
                self.train_queries_str = self.train_queries_str[:n_train_queries]
                self.train_depths = self.train_depths[:n_train_queries]
                self.train_labels = self.train_labels[:n_train_queries]
                if hasattr(self, "train_queries") and self.train_queries:
                    self.train_queries = self.train_queries[:n_train_queries]

        # 6. Sort (deterministic ordering for parity).
        if sort_data:
            self._apply_sort_data()

        # 7. Vocabulary discovery.
        if discover_vocab:
            self.discover_vocabulary(
                rules_str=self.rules_str,
                queries_str=(
                    self.train_queries_str
                    + self.valid_queries_str
                    + self.test_queries_str
                ),
            )

        # 8. Build KG (default-domain catch-all + id-view rebuild).
        if build_kg:
            self.build_kg(default_domain_name=default_domain_name)

        # 9. Materialize tensors.
        if materialize:
            self.materialize(device=device)

    def _apply_sort_data(self) -> None:
        """Alphabetical sort of facts + rules for deterministic ordering.

        Sorts ``facts_str`` (natural) plus parallel ``self.facts`` if
        populated, and ``rules_str`` (by head atom) with parallel
        ``rule_names`` / ``rule_weights`` / ``self.rules`` re-aligned.

        Per-split queries are NOT sorted — file order matters for parity
        references (notably DpRL's SB3 baseline) which iterate queries in
        source-file order. Sort the typed parallel lists alongside the
        str-form so the two views stay aligned. Idempotent.
        """
        if self.facts_str:
            order = sorted(range(len(self.facts_str)),
                           key=lambda i: self.facts_str[i])
            self.facts_str = [self.facts_str[i] for i in order]
            facts_typed = getattr(self, "facts", None)
            if facts_typed and len(facts_typed) == len(order):
                self.facts = [facts_typed[i] for i in order]
        if self.rules_str:
            # Sort by HEAD only (head_tuple). Stable sort preserves the
            # original (file) order across rules with the same head — DpRL's
            # SB3 parity reference relies on this for deterministic rule
            # selection in the env. Including body in the key would shuffle
            # same-head rules.
            order = sorted(range(len(self.rules_str)),
                           key=lambda i: self.rules_str[i][0])
            self.rules_str = [self.rules_str[i] for i in order]
            self.rule_names = [self.rule_names[i] for i in order]
            self.rule_weights = [self.rule_weights[i] for i in order]
            rules_typed = getattr(self, "rules", None)
            if rules_typed and len(rules_typed) == len(order):
                self.rules = [rules_typed[i] for i in order]

    def materialize(
        self,
        *,
        device: Optional[Any] = None,
        entity_id_fn: Optional[Any] = None,
        relation_id_fn: Optional[Any] = None,
        include_rules: bool = True,
    ) -> None:
        """Build tensor versions of facts, rules, and queries.

        Defaults use ``self.entity2id`` / ``self.relation2id`` directly.
        Subclasses (e.g. DpRL with ``IndexManager``) can pass alternate
        id-mapping callbacks to remap into a shifted id space (DpRL
        reserves id 0 for padding atoms).

        Populates these attributes on ``self``:

        - ``facts_t: LongTensor[F, 3]``
        - ``rules_t: LongTensor[R, max_body, 3]`` (body atoms; only when
          ``include_rules`` is True — rule atoms with variable arguments
          can't be encoded by ``entity_id_fn``, so consumers with variable
          rules pass ``include_rules=False`` and tensorize rules
          themselves).
        - ``rule_lens_t: LongTensor[R]``
        - ``rules_heads_t: LongTensor[R, 3]``
        - per-split: ``train_queries_t``, ``valid_queries_t``, ``test_queries_t``
          all shape ``[N, 3]``
        - per-split: ``train_labels_t``, ``valid_labels_t``, ``test_labels_t``
          shape ``[N]``
        - per-split: ``train_depths_t``, ``valid_depths_t``, ``test_depths_t``
          shape ``[N]``

        Source-of-truth for the string-form is ``facts_str`` / ``rules_str`` /
        ``{split}_queries_str`` / ``{split}_depths`` / ``{split}_labels``;
        these must be populated before calling ``materialize`` (typically
        by ``load_facts`` / ``load_rules`` / ``load_queries_with_depth``).
        """
        device = device or torch.device("cpu")
        ent_fn = entity_id_fn or (lambda e: self.entity2id[e])
        rel_fn = relation_id_fn or (lambda r: self.relation2id[r])

        def _atom_to_row(atom: Tuple[str, ...]) -> Tuple[int, int, int]:
            # Public format is (predicate, head, tail). Tensor row is
            # (relation_id, head_id, tail_id) — same column order.
            return (rel_fn(atom[0]), ent_fn(atom[1]), ent_fn(atom[2]))

        # Facts
        if self.facts_str:
            self.facts_t = torch.tensor(
                [_atom_to_row(a) for a in self.facts_str],
                dtype=torch.long, device=device,
            )
        else:
            self.facts_t = torch.empty((0, 3), dtype=torch.long, device=device)

        # Rules: pack body atoms into [R, max_body, 3]; head into [R, 3].
        # Skipped when ``include_rules=False`` — for consumers whose rules
        # contain variables (DpRL) the entity_id_fn can't encode them; those
        # consumers materialize rules separately via their own machinery
        # (IndexManager.rules_to_tensor in DpRL's case).
        if self.rules_str and include_rules:
            max_body = max((len(body) for _, body in self.rules_str), default=0)
            max_body = max(1, max_body)
            R = len(self.rules_str)
            rules_t = torch.zeros((R, max_body, 3), dtype=torch.long, device=device)
            rule_lens = torch.zeros((R,), dtype=torch.long, device=device)
            heads_t = torch.zeros((R, 3), dtype=torch.long, device=device)
            for r, (head, body) in enumerate(self.rules_str):
                heads_t[r] = torch.tensor(_atom_to_row(head), dtype=torch.long)
                for b, atom in enumerate(body):
                    rules_t[r, b] = torch.tensor(_atom_to_row(atom), dtype=torch.long)
                rule_lens[r] = len(body)
            self.rules_t = rules_t
            self.rule_lens_t = rule_lens
            self.rules_heads_t = heads_t
        else:
            self.rules_t = torch.empty((0, 1, 3), dtype=torch.long, device=device)
            self.rule_lens_t = torch.empty((0,), dtype=torch.long, device=device)
            self.rules_heads_t = torch.empty((0, 3), dtype=torch.long, device=device)

        # Per-split queries / labels / depths.
        for split in ("train", "valid", "test"):
            queries = getattr(self, f"{split}_queries_str")
            labels = getattr(self, f"{split}_labels")
            depths = getattr(self, f"{split}_depths")
            if queries:
                q_t = torch.tensor(
                    [_atom_to_row(a) for a in queries],
                    dtype=torch.long, device=device,
                )
            else:
                q_t = torch.empty((0, 3), dtype=torch.long, device=device)
            l_t = torch.as_tensor(labels, dtype=torch.long, device=device)
            d_t = torch.as_tensor(depths, dtype=torch.long, device=device)
            setattr(self, f"{split}_queries_t", q_t)
            setattr(self, f"{split}_labels_t", l_t)
            setattr(self, f"{split}_depths_t", d_t)

    def build_kg(self, default_domain_name: str = "default") -> None:
        """Default-domain catch-all over a loaded domain file.

        Only runs when a real domain file was loaded (i.e.
        ``self.domain2entities`` is non-empty). For any constant not
        assigned by the file, appends it to the ``default_domain_name``
        bucket and updates ``self.entity2domain``. Then rebuilds
        ``self.domain2idx`` / ``self.entity2domain_idx`` and
        ``self.head_domain`` / ``self.tail_domain`` from the updated
        string-keyed maps.

        For datasets WITHOUT a domain file, no-op: domain2entities stays
        empty, the sampler sees no domain restriction (KGE-standard
        behavior), and parity references that don't load domains stay
        consistent with the canonical id space.
        """
        if not self.domain2entities:
            # No real domain file loaded — nothing to catch up with.
            return

        # Catch-all over alphabetical self.constants for determinism.
        self.domain2entities.setdefault(default_domain_name, [])
        for c in self.constants:
            if c not in self.entity2domain:
                self.domain2entities[default_domain_name].append(c)
                self.entity2domain[c] = default_domain_name

        # Rebuild id-keyed views from the updated string-keyed maps.
        self.domain2idx = {
            dom: [self.entity2id[e] for e in entities if e in self.entity2id]
            for dom, entities in self.domain2entities.items()
        }
        self.entity2domain_idx = {
            self.entity2id[e]: d
            for e, d in self.entity2domain.items()
            if e in self.entity2id
        }
        # Per-relation domain restriction reflects the catch-all additions.
        if self.use_domain_eval and self.domain2idx:
            self.head_domain, self.tail_domain = build_relation_domains_from_file(
                self.train_idx + self.valid_idx + self.test_idx,
                self.entity2domain_idx, self.domain2idx,
            )

    def build_domain_idx(self, entity_id_map: Optional[Dict[str, int]] = None) -> None:
        """Rebuild id-keyed domain views from the string-keyed mapping.

        ``self.domain2entities`` (str → str list) is the canonical source.
        This method populates ``self.domain2idx`` (str → int list) and
        ``self.entity2domain_idx`` (int → str) using either ``self.entity2id``
        (default) or a custom ``entity_id_map`` (e.g. DpRL's
        ``IndexManager.constant_str2idx`` which has a +1 padding shift).

        Skip entities not present in the id map.
        """
        ent_map = self.entity2id if entity_id_map is None else entity_id_map
        self.domain2idx = {
            dom: [ent_map[e] for e in entities if e in ent_map]
            for dom, entities in self.domain2entities.items()
        }
        # Drop empty buckets (caller-friendly).
        self.domain2idx = {k: v for k, v in self.domain2idx.items() if v}
        self.entity2domain_idx = {
            ent_map[e]: d
            for e, d in self.entity2domain.items()
            if e in ent_map
        }

    # ---- convenience accessors ----------------------------------------

    def build_sampler(
        self,
        *,
        default_mode: str = "both",
        seed: int = 0,
        device: Optional[Any] = None,
        order_negatives: bool = False,
        kind: str = "regular",
        train_triples_for_bern: Optional[Any] = None,
        with_domain: bool = True,
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
            kind: ``"regular"`` (the default :class:`Sampler`) or
                ``"bernoulli"`` (the :class:`BernoulliSampler` with
                per-relation head/tail-corrupt probabilities). Bernoulli
                is used by tkk's ``train_model`` when
                ``corruption_scheme == "both"``.
            train_triples_for_bern: When ``kind="bernoulli"``, the
                training triples used to compute Bernoulli probabilities.
                Falls back to ``self.train_idx``.
            with_domain: When ``False``, drop the domain restriction on
                this sampler (used to build a no-domain training sampler
                from a kb that carries domain info for eval).

        Returns:
            A configured :class:`kge_kernels.scoring.Sampler` (or
            :class:`BernoulliSampler` when ``kind="bernoulli"``).
        """
        import torch

        from ..scoring import BernoulliSampler, Sampler

        device = device or torch.device("cpu")
        if self.ground_facts_idx_set:
            all_known = torch.tensor(
                sorted(self.ground_facts_idx_set), dtype=torch.long,
            )
        else:
            all_known = torch.empty((0, 3), dtype=torch.long)

        # ``min_entity_idx`` is derived from the padding convention:
        # id 0 reserved → entities start at 1; ``padding_idx=None`` (dense
        # 0-based ids) → entities start at 0. Read the private
        # ``_padding_idx`` because the public ``padding_idx`` defaults
        # to 0 in the dense-id case for backwards compat.
        min_entity_idx = 1 if self._padding_idx == 0 else 0
        d2i = (self.domain2idx or None) if with_domain else None
        e2d = (self.entity2domain_idx or None) if with_domain else None

        if kind == "bernoulli":
            train_t = train_triples_for_bern
            if train_t is None:
                train_t = torch.tensor(self.train_idx, dtype=torch.long)
            return BernoulliSampler.from_data(
                all_known_triples_idx=all_known,
                num_entities=self.num_entities,
                num_relations=self.num_relations,
                device=device,
                bern_probs=BernoulliSampler.compute_probs(
                    train_t, self.num_relations,
                ),
                min_entity_idx=min_entity_idx,
                domain2idx=d2i,
                entity2domain=e2d,
            )
        if kind != "regular":
            raise ValueError(f"Unknown sampler kind: {kind!r}")
        return Sampler.from_data(
            all_known_triples_idx=all_known,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            device=device,
            default_mode=default_mode,
            seed=seed,
            domain2idx=d2i,
            entity2domain=e2d,
            order_negatives=order_negatives,
            min_entity_idx=min_entity_idx,
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


__all__ = ["KnowledgeBase", "MaterializedSplit"]
