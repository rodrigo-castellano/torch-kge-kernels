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
