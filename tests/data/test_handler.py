"""Tests for KnowledgeBase — base loader + convenience methods."""
from __future__ import annotations

import torch

from kge_kernels.data import KnowledgeBase
from kge_kernels.scoring import Sampler


def _write_tiny_dataset(tmp_path):
    """Build a minimal dataset folder with train/valid/test/facts."""
    base = tmp_path / "tinykg"
    base.mkdir()
    (base / "train.txt").write_text("alice\tknows\tbob\nbob\tknows\tcarol\n")
    (base / "valid.txt").write_text("alice\tknows\tcarol\n")
    (base / "test.txt").write_text("carol\tknows\talice\n")
    (base / "facts.txt").write_text("dave\tknows\teve\n")
    return str(tmp_path), "tinykg"


def test_handler_loads_canonical_pipeline(tmp_path):
    base_path, name = _write_tiny_dataset(tmp_path)
    h = KnowledgeBase(name, base_path)
    # Alphabetical id assignment: alice=0, bob=1, carol=2, dave=3, eve=4
    assert h.entity2id == {"alice": 0, "bob": 1, "carol": 2, "dave": 3, "eve": 4}
    assert h.relation2id == {"knows": 0}
    assert h.num_entities == 5
    assert h.num_relations == 1
    assert len(h.train_idx) == 2
    assert len(h.valid_idx) == 1
    assert len(h.test_idx) == 1
    assert len(h.known_idx) == 1


def test_handler_split_idx_returns_indexed_triples(tmp_path):
    base_path, name = _write_tiny_dataset(tmp_path)
    h = KnowledgeBase(name, base_path)
    assert h.split_idx("train") == h.train_idx
    assert h.split_idx("valid") == h.valid_idx
    assert h.split_idx("test") == h.test_idx
    assert h.split_idx("known") == h.known_idx


def test_build_sampler_uses_loaded_id_space(tmp_path):
    base_path, name = _write_tiny_dataset(tmp_path)
    h = KnowledgeBase(name, base_path)
    sampler = h.build_sampler()
    assert isinstance(sampler, Sampler)
    assert sampler.num_entities == h.num_entities
    assert sampler.num_relations == h.num_relations
    # Filter index covers ALL ground triples (train + valid + test + facts)
    pos = torch.tensor([[0, 0, 1]], dtype=torch.long)  # knows(alice, bob)
    neg, valid = sampler.corrupt_with_mask(pos, num_negatives=10, mode="tail")
    rows = [tuple(t.tolist()) for t, k in zip(neg[0], valid[0]) if k]
    # Filter should remove every known (knows, alice, *) triple
    for r, h_idx, t_idx in rows:
        assert (r, h_idx, t_idx) not in h.ground_facts_idx_set


def test_build_sampler_picks_up_domain_info(tmp_path):
    base_path, name = _write_tiny_dataset(tmp_path)
    base = tmp_path / name
    (base / "domain.txt").write_text("people alice bob carol\nothers dave eve\n")
    h = KnowledgeBase(name, base_path, domain_file="domain.txt")
    sampler = h.build_sampler()
    # Domain-restricted corruption: tail of (knows, alice, bob) stays in 'people'.
    pos = torch.tensor([[0, 0, 1]], dtype=torch.long)
    neg, valid = sampler.corrupt_with_mask(pos, num_negatives=10, mode="tail")
    rows = [tuple(t.tolist()) for t, k in zip(neg[0], valid[0]) if k]
    people_ids = {h.entity2id[n] for n in ("alice", "bob", "carol")}
    for _r, _h, t in rows:
        assert t in people_ids


# ---------------------------------------------------------------------------
# explicit domain file (no auto-detect — see _load_dataset comments)
# ---------------------------------------------------------------------------


def test_domain_file_must_be_explicit(tmp_path):
    """A ``domain2constants.txt`` in the dataset folder is NOT auto-detected.
    Callers must pass ``domain_file=`` explicitly. Some downstream parity
    references (SB3) don't use the domain file even when present, so
    auto-detecting would diverge from them."""
    base_path, name = _write_tiny_dataset(tmp_path)
    base = tmp_path / name
    (base / "domain2constants.txt").write_text("people alice bob carol\nothers dave eve\n")
    # Without explicit arg: not loaded.
    h = KnowledgeBase(name, base_path)
    assert not h.use_domain_eval
    assert h.domain2entities == {}
    # Explicit: loaded.
    h2 = KnowledgeBase(name, base_path, domain_file="domain2constants.txt")
    assert h2.use_domain_eval
    assert "people" in h2.domain2entities


# ---------------------------------------------------------------------------
# build_kg — default-domain catch-all + id-view rebuild
# ---------------------------------------------------------------------------


def test_build_kg_catch_all_when_no_domain_file(tmp_path):
    """Without a domain file, every constant lands in the default-domain
    bucket and id-keyed views are populated."""
    base_path, name = _write_tiny_dataset(tmp_path)
    h = KnowledgeBase(name, base_path)
    h.build_kg(default_domain_name="default")
    assert "default" in h.domain2entities
    # All 5 constants in the catch-all
    assert set(h.domain2entities["default"]) == set(h.constants)
    # Id-keyed views populated
    assert "default" in h.domain2idx
    assert len(h.domain2idx["default"]) == 5
    # Per-relation domain restriction NOT activated (no real domain file)
    assert not h.use_domain_eval


def test_build_kg_appends_missing_to_default_when_partial_domain_file(tmp_path):
    """Constants present in domain file keep their domain; missing
    constants get default-domain assignment."""
    base_path, name = _write_tiny_dataset(tmp_path)
    base = tmp_path / name
    # Only alice and bob are typed; carol/dave/eve are not.
    (base / "domain2constants.txt").write_text("people alice bob\n")
    h = KnowledgeBase(name, base_path, domain_file="domain2constants.txt")
    h.build_kg(default_domain_name="default")
    assert h.entity2domain["alice"] == "people"
    assert h.entity2domain["bob"] == "people"
    assert h.entity2domain["carol"] == "default"
    assert h.entity2domain["dave"] == "default"
    assert h.entity2domain["eve"] == "default"
    # Id-keyed view in sync after catch-all
    assert h.entity2id["carol"] in h.domain2idx["default"]


# ---------------------------------------------------------------------------
# materialize — tensor versions of facts / rules / queries
# ---------------------------------------------------------------------------


def test_materialize_facts_tensor(tmp_path):
    base_path, name = _write_tiny_dataset(tmp_path)
    h = KnowledgeBase(name, base_path)
    h.facts_str = list(h.known_facts)  # convert from base str-tuples
    h.materialize()
    assert h.facts_t.shape == (1, 3)  # one fact: (knows, dave, eve)
    r, hd, tl = h.facts_t[0].tolist()
    assert (r, hd, tl) == (
        h.relation2id["knows"], h.entity2id["dave"], h.entity2id["eve"],
    )


def test_materialize_rules_tensor(tmp_path):
    """Rules tensor packs body atoms; rule_lens gives unpadded count."""
    base_path, name = _write_tiny_dataset(tmp_path)
    h = KnowledgeBase(name, base_path)
    h.rules_str = [
        # head: knows(alice, bob); body: knows(alice, carol)
        (("knows", "alice", "bob"), [("knows", "alice", "carol")]),
        # head: knows(bob, carol); body: knows(alice, bob), knows(bob, carol)
        (("knows", "bob", "carol"),
         [("knows", "alice", "bob"), ("knows", "bob", "carol")]),
    ]
    h.materialize()
    assert h.rules_t.shape == (2, 2, 3)
    assert h.rule_lens_t.tolist() == [1, 2]
    assert h.rules_heads_t.shape == (2, 3)


def test_materialize_queries_tensor(tmp_path):
    base_path, name = _write_tiny_dataset(tmp_path)
    h = KnowledgeBase(name, base_path)
    h.train_queries_str = [("knows", "alice", "bob"), ("knows", "bob", "carol")]
    h.train_labels = [1, 0]
    h.train_depths = [-1, 2]
    h.materialize()
    assert h.train_queries_t.shape == (2, 3)
    assert h.train_labels_t.tolist() == [1, 0]
    assert h.train_depths_t.tolist() == [-1, 2]


def test_materialize_with_id_fn_callbacks(tmp_path):
    """Custom id-fn callbacks remap into a different id space (e.g. shifted)."""
    base_path, name = _write_tiny_dataset(tmp_path)
    h = KnowledgeBase(name, base_path)
    h.facts_str = list(h.known_facts)
    # Shift entity ids by +1 (DpRL-style: id 0 reserved for padding)
    h.materialize(
        entity_id_fn=lambda e: h.entity2id[e] + 1,
        relation_id_fn=lambda r: h.relation2id[r],
    )
    r, hd, tl = h.facts_t[0].tolist()
    assert hd == h.entity2id["dave"] + 1
    assert tl == h.entity2id["eve"] + 1


# ---------------------------------------------------------------------------
# discover_vocabulary populates Set + List + frozenset views
# ---------------------------------------------------------------------------


def test_discover_vocabulary_finalizes_set_views(tmp_path):
    base_path, name = _write_tiny_dataset(tmp_path)
    h = KnowledgeBase(name, base_path)
    # Pretend a rule introduced a new predicate "ancestor" not in any fact
    rules_str = [(("ancestor", "X", "Y"), [("knows", "X", "Y")])]
    h.discover_vocabulary(rules_str=rules_str, queries_str=[], constants=set(), predicates=set())
    assert "ancestor" in h.predicates_set
    assert "knows" in h.predicates_set
    # constants finalized as alphabetical List + frozenset
    assert h.constants == sorted(h.entity2id.keys())
    assert isinstance(h.constants_set, frozenset)
