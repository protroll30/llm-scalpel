"""Unit tests for discovery.circuit_graphviz helpers (no model forward)."""

import io

from discovery.circuit_graphviz import dot_escape_label, score_to_fillcolor, write_bipartite_sae_dot


def test_dot_escape_label() -> None:
    assert dot_escape_label('a"b') == 'a\\"b'
    assert dot_escape_label("x\\y") == "x\\\\y"


def test_score_to_fillcolor_bounds() -> None:
    assert score_to_fillcolor(1.0, 1.0).startswith("#")
    assert score_to_fillcolor(-1.0, 1.0).startswith("#")


def test_write_bipartite_sae_dot_smoke() -> None:
    buf = io.StringIO()
    write_bipartite_sae_dot(
        out=buf,
        src_feature_ids=[1, 2],
        dst_feature_ids=[9],
        edge_weight={(1, 9): 0.5, (2, 9): -0.25},
        src_taylor={1: 0.1, 2: -0.2},
        dst_taylor={9: 0.3},
        min_abs_edge=0.0,
    )
    text = buf.getvalue()
    assert "digraph sae_circuit" in text
    assert "src_1" in text and "dst_9" in text
    assert "#27ae60" in text or "#c0392b" in text
