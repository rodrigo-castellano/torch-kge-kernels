"""Tests for kge_kernels.data (triple file loading + filter maps)."""
from __future__ import annotations

import pytest

from kge_kernels.data import (
    TripleExample,
    add_reciprocal_triples,
    build_filter_maps,
    build_relation_domains,
    detect_triple_format,
    encode_split_triples,
    filter_queries_by_predicates,
    load_domain_file,
    load_probabilistic_facts,
    load_rules_file,
    load_triples,
    load_triples_with_mappings,
)

# ═══════════════════════════════════════════════════════════════════════
# detect_triple_format
# ═══════════════════════════════════════════════════════════════════════


def test_detect_tsv():
    assert detect_triple_format("alice\tknows\tbob") == "tsv"


def test_detect_csv():
    assert detect_triple_format("alice,knows,bob") == "csv"


def test_detect_prolog():
    assert detect_triple_format("knows(alice, bob).") == "prolog"


def test_detect_unknown_empty():
    assert detect_triple_format("") == "unknown"


# ═══════════════════════════════════════════════════════════════════════
# load_triples
# ═══════════════════════════════════════════════════════════════════════


def test_load_triples_tsv(tmp_path):
    path = tmp_path / "triples.tsv"
    path.write_text("alice\tknows\tbob\ncarol\tknows\tdave\n")
    triples = load_triples(str(path))
    assert len(triples) == 2
    assert triples[0] == TripleExample("alice", "knows", "bob")
    assert triples[1] == TripleExample("carol", "knows", "dave")


def test_load_triples_csv(tmp_path):
    path = tmp_path / "triples.csv"
    path.write_text("alice,knows,bob\ncarol,knows,dave\n")
    triples = load_triples(str(path))
    assert len(triples) == 2


def test_load_triples_prolog(tmp_path):
    path = tmp_path / "triples.pl"
    path.write_text("knows(alice, bob).\n# a comment\nknows(carol, dave).\n")
    triples = load_triples(str(path))
    assert len(triples) == 2
    assert triples[0] == TripleExample("alice", "knows", "bob")


def test_load_triples_prolog_rejects_non_binary(tmp_path):
    path = tmp_path / "ternary.pl"
    path.write_text("between(alice, bob, carol).\n")
    with pytest.raises(ValueError, match="binary"):
        load_triples(str(path))


def test_load_triples_with_mappings(tmp_path):
    path = tmp_path / "triples.tsv"
    path.write_text("alice\tknows\tbob\nbob\tknows\tcarol\n")
    triple_ids, e2id, r2id = load_triples_with_mappings(str(path))
    assert len(triple_ids) == 2
    # Entity ids assigned in discovery order: alice=0, bob=1, carol=2
    assert e2id == {"alice": 0, "bob": 1, "carol": 2}
    assert r2id == {"knows": 0}
    # Triple format is (r, h, t)
    assert triple_ids[0] == (0, 0, 1)
    assert triple_ids[1] == (0, 1, 2)


# ═══════════════════════════════════════════════════════════════════════
# encode_split_triples
# ═══════════════════════════════════════════════════════════════════════


def test_encode_split_triples_known_vocab(tmp_path):
    path = tmp_path / "valid.tsv"
    path.write_text("alice\tknows\tbob\n")
    e2id = {"alice": 0, "bob": 1}
    r2id = {"knows": 0}
    out = encode_split_triples(str(path), e2id, r2id, "valid")
    assert out == [(0, 0, 1)]


def test_encode_split_triples_extends_vocab_on_unseen(tmp_path, capsys):
    path = tmp_path / "valid.tsv"
    path.write_text("alice\tlikes\tbob\ncarol\tknows\tdave\n")
    e2id = {"alice": 0, "bob": 1}
    r2id = {"knows": 0}
    out = encode_split_triples(str(path), e2id, r2id, "valid")
    # All 4 entities + 2 relations should now be in the vocab
    assert len(e2id) == 4
    assert len(r2id) == 2
    assert len(out) == 2


# ═══════════════════════════════════════════════════════════════════════
# add_reciprocal_triples
# ═══════════════════════════════════════════════════════════════════════


def test_add_reciprocal_doubles_triple_count():
    triples = [(0, 1, 2), (0, 3, 4), (1, 5, 6)]
    relation2id = {"r0": 0, "r1": 1}
    out, expanded, new_num = add_reciprocal_triples(triples, relation2id)
    assert new_num == 4
    assert len(out) == 6
    # Reciprocals swap h and t and shift relation by original num_relations=2
    assert (2, 2, 1) in out
    assert (2, 4, 3) in out
    assert (3, 6, 5) in out
    assert expanded == {"r0": 0, "r1": 1, "r0__inv": 2, "r1__inv": 3}


# ═══════════════════════════════════════════════════════════════════════
# build_filter_maps
# ═══════════════════════════════════════════════════════════════════════


def test_build_filter_maps_single_collection():
    triples = [(0, 1, 2), (0, 3, 2), (0, 1, 4)]
    head_filter, tail_filter = build_filter_maps(triples)
    # For (?, r=0, t=2), heads {1, 3} are known
    assert head_filter[(0, 2)] == {1, 3}
    # For (h=1, r=0, ?), tails {2, 4} are known
    assert tail_filter[(1, 0)] == {2, 4}


def test_build_filter_maps_multiple_collections():
    train = [(0, 1, 2)]
    valid = [(0, 1, 3)]
    test = [(0, 1, 4)]
    _, tail_filter = build_filter_maps(train, valid, test)
    assert tail_filter[(1, 0)] == {2, 3, 4}


def test_build_filter_maps_empty_collection_ignored():
    result = build_filter_maps([], [(0, 1, 2)])
    head_filter, tail_filter = result
    assert head_filter[(0, 2)] == {1}


# ═══════════════════════════════════════════════════════════════════════
# build_relation_domains
# ═══════════════════════════════════════════════════════════════════════


def test_build_relation_domains():
    triples = [(0, 1, 2), (0, 3, 4), (1, 5, 6), (0, 1, 7)]
    head_domain, tail_domain = build_relation_domains(triples)
    assert head_domain[0] == {1, 3}
    assert tail_domain[0] == {2, 4, 7}
    assert head_domain[1] == {5}
    assert tail_domain[1] == {6}


# ═══════════════════════════════════════════════════════════════════════
# load_domain_file
# ═══════════════════════════════════════════════════════════════════════


def test_load_domain_file_basic(tmp_path):
    p = tmp_path / "domain.txt"
    p.write_text("person alice bob carol\nplace paris tokyo\n")
    e2id = {"alice": 0, "bob": 1, "carol": 2, "paris": 10, "tokyo": 11}
    d2i, e2d = load_domain_file(str(p), e2id)
    assert d2i == {"person": [0, 1, 2], "place": [10, 11]}
    assert e2d == {0: "person", 1: "person", 2: "person", 10: "place", 11: "place"}


def test_load_domain_file_skips_unknown_entities(tmp_path):
    p = tmp_path / "domain.txt"
    p.write_text("person alice unknown_entity bob\n")
    e2id = {"alice": 0, "bob": 1}
    d2i, e2d = load_domain_file(str(p), e2id)
    assert d2i == {"person": [0, 1]}
    assert e2d == {0: "person", 1: "person"}


def test_load_domain_file_skips_empty_domains(tmp_path):
    p = tmp_path / "domain.txt"
    # domain "other" has only unknown entities — must be skipped entirely
    p.write_text("person alice bob\nother x y z\n")
    e2id = {"alice": 0, "bob": 1}
    d2i, e2d = load_domain_file(str(p), e2id)
    assert "other" not in d2i
    assert d2i == {"person": [0, 1]}


def test_load_domain_file_missing_path_returns_empty(tmp_path):
    d2i, e2d = load_domain_file(str(tmp_path / "does_not_exist.txt"), {})
    assert d2i == {}
    assert e2d == {}


def test_load_domain_file_entity_keeps_first_domain(tmp_path):
    # If an entity appears in multiple domains, entity2domain keeps the first.
    p = tmp_path / "domain.txt"
    p.write_text("person alice\nfriend alice\n")
    e2id = {"alice": 0}
    d2i, e2d = load_domain_file(str(p), e2id)
    assert d2i == {"person": [0], "friend": [0]}
    assert e2d == {0: "person"}  # first-wins


# ═══════════════════════════════════════════════════════════════════════
# load_rules_file
# ═══════════════════════════════════════════════════════════════════════


def test_load_rules_file_parses_horn_clauses(tmp_path):
    p = tmp_path / "rules.txt"
    p.write_text(
        "% comment\n"
        "ancestor(X, Y) :- parent(X, Y).\n"
        "ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).\n"
        "\n"
    )
    specs, var_to_domain = load_rules_file(str(p))
    assert var_to_domain == {}
    assert len(specs) == 2
    assert specs[0].head == ("ancestor", "X", "Y")
    assert specs[0].body == [("parent", "X", "Y")]
    assert specs[0].name is None and specs[0].weight == 1.0
    assert specs[1].head == ("ancestor", "X", "Z")
    assert specs[1].body == [("parent", "X", "Y"), ("ancestor", "Y", "Z")]


def test_load_rules_file_uppercase_args(tmp_path):
    p = tmp_path / "rules.txt"
    p.write_text("ancestor(x, y) :- parent(x, y).\n")
    specs, _ = load_rules_file(str(p), uppercase_args=True)
    assert specs[0].head == ("ancestor", "X", "Y")
    assert specs[0].body == [("parent", "X", "Y")]


def test_load_rules_file_arrow_format(tmp_path):
    """`body, body -> head` direction (DPL / ProbLog convention)."""
    p = tmp_path / "rules.txt"
    p.write_text("parent(X, Y), parent(Y, Z) -> grandparent(X, Z)\n")
    specs, _ = load_rules_file(str(p))
    assert specs[0].head == ("grandparent", "X", "Z")
    assert specs[0].body == [("parent", "X", "Y"), ("parent", "Y", "Z")]


def test_load_rules_file_probabilistic_prefix(tmp_path):
    """Strip the `rN:weight:` rule prefix; capture name + weight."""
    p = tmp_path / "rules.txt"
    p.write_text(
        "r0:0.747:brother(a,h), mother(b,h) -> son(a,b)\n"
        "r1:1:locatedInCS(X,W), locatedInSR(W,Z) -> locatedInCR(X,Z)\n"
    )
    specs, _ = load_rules_file(str(p), uppercase_args=True)
    assert len(specs) == 2
    assert specs[0].name == "r0"
    assert specs[0].weight == 0.747
    assert specs[0].head == ("son", "A", "B")
    assert specs[0].body == [("brother", "A", "H"), ("mother", "B", "H")]
    assert specs[1].name == "r1"
    assert specs[1].weight == 1.0
    assert specs[1].head == ("locatedInCR", "X", "Z")


def test_load_rules_file_var2domain_preamble(tmp_path):
    """Parse the `var2domain X dom1 Y dom2 ...` preamble line."""
    p = tmp_path / "rules.txt"
    p.write_text(
        "var2domain X person Y location Z thing\n"
        "located(X, Y) :- visits(X, Y).\n"
    )
    specs, var_to_domain = load_rules_file(str(p))
    assert var_to_domain == {"X": "person", "Y": "location", "Z": "thing"}
    assert len(specs) == 1


def test_load_rules_file_standalone_fact(tmp_path):
    p = tmp_path / "rules.txt"
    p.write_text("base_fact(a, b)\n")
    specs, _ = load_rules_file(str(p))
    assert len(specs) == 1
    assert specs[0].head == ("base_fact", "a", "b")
    assert specs[0].body == []


def test_load_rules_file_skips_malformed_silently(tmp_path):
    p = tmp_path / "rules.txt"
    p.write_text("not_a_rule\nancestor(X, Y) :- parent(X, Y).\nbroken :-\n")
    specs, _ = load_rules_file(str(p))
    # ``not_a_rule`` is a standalone-fact-like atom; ``broken :-`` is malformed.
    # Behavior: ``not_a_rule`` parses as standalone (no parens → returns None).
    # So only the well-formed rule survives.
    assert len(specs) == 1


def test_load_rules_file_missing_returns_empty(tmp_path):
    specs, var_to_domain = load_rules_file(str(tmp_path / "nope.txt"))
    assert specs == []
    assert var_to_domain == {}


# ═══════════════════════════════════════════════════════════════════════
# load_probabilistic_facts
# ═══════════════════════════════════════════════════════════════════════


def test_load_probabilistic_facts_basic(tmp_path):
    p = tmp_path / "kge_top.txt"
    p.write_text(
        "# header\n"
        "father(alice,bob) 0.95 1\n"
        "mother(alice,carol) 0.80 2\n"
        "\n"
    )
    facts = load_probabilistic_facts(str(p))
    assert facts == [
        ("father", "alice", "bob"),
        ("mother", "alice", "carol"),
    ]


def test_load_probabilistic_facts_topk_filter(tmp_path):
    p = tmp_path / "kge_top.txt"
    p.write_text(
        "father(alice,bob) 0.95 1\n"
        "father(alice,carol) 0.50 5\n"
    )
    facts = load_probabilistic_facts(str(p), topk_limit=3)
    assert facts == [("father", "alice", "bob")]


def test_load_probabilistic_facts_score_threshold(tmp_path):
    p = tmp_path / "kge_top.txt"
    p.write_text(
        "father(alice,bob) 0.95\n"
        "father(alice,carol) 0.30\n"
    )
    facts = load_probabilistic_facts(str(p), score_threshold=0.5)
    assert facts == [("father", "alice", "bob")]


def test_load_probabilistic_facts_dedupes(tmp_path):
    p = tmp_path / "kge_top.txt"
    p.write_text(
        "father(alice,bob) 0.95\n"
        "father(alice,bob) 0.80\n"
    )
    facts = load_probabilistic_facts(str(p))
    assert facts == [("father", "alice", "bob")]


# ═══════════════════════════════════════════════════════════════════════
# filter_queries_by_predicates
# ═══════════════════════════════════════════════════════════════════════


def test_filter_queries_by_predicates_keeps_matching():
    queries = [
        ("father", "alice", "bob"),
        ("mother", "alice", "carol"),
        ("father", "carol", "dave"),
        ("sibling", "bob", "carol"),
    ]
    allowed = {"father", "sibling"}
    filtered, kept = filter_queries_by_predicates(queries, allowed)
    assert filtered == [
        ("father", "alice", "bob"),
        ("father", "carol", "dave"),
        ("sibling", "bob", "carol"),
    ]
    assert kept == [0, 2, 3]


def test_filter_queries_by_predicates_empty_allow_drops_all():
    queries = [("father", "alice", "bob")]
    filtered, kept = filter_queries_by_predicates(queries, set())
    assert filtered == []
    assert kept == []


def test_filter_queries_by_predicates_skips_empty_tuples():
    queries = [(), ("father", "alice", "bob")]
    filtered, kept = filter_queries_by_predicates(queries, {"father"})
    assert filtered == [("father", "alice", "bob")]
    assert kept == [1]
