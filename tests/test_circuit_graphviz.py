"""Unit tests for discovery.circuit_graphviz helpers (no model forward)."""

import io

from discovery.circuit_graphviz import (
    _edge_label,
    dot_escape_label,
    score_to_fillcolor,
    write_bipartite_sae_dot,
    write_tripartite_sae_head_dot,
)


def test_dot_escape_label() -> None:
    assert dot_escape_label('a"b') == 'a\\"b'
    assert dot_escape_label("x\\y") == "x\\\\y"
    # Real Python newline becomes the DOT line-break escape (literal "\n"),
    # not "\\n" — otherwise nodes render with a visible "\n" instead of wrapping.
    assert dot_escape_label("a\nb") == "a\\nb"


def test_score_to_fillcolor_bounds() -> None:
    assert score_to_fillcolor(1.0, 1.0).startswith("#")
    assert score_to_fillcolor(-1.0, 1.0).startswith("#")


def test_edge_label_suppresses_near_zero() -> None:
    # Sub-threshold weights drop their label; structure (color/penwidth) is
    # still drawn by the writer, so this only declutters the readout.
    assert _edge_label(0.0) == ""
    assert _edge_label(1e-6) == ""
    assert _edge_label(-1e-6) == ""
    # Above threshold: signed, 3 sig figs.
    assert _edge_label(0.286) == "+0.286"
    assert _edge_label(-0.0725) == "-0.0725"


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


def test_write_tripartite_sae_head_dot_smoke() -> None:
    buf = io.StringIO()
    write_tripartite_sae_head_dot(
        out=buf,
        src_feature_ids=[1],
        dst_feature_ids=[99],
        middle_heads=[(8, 11), (9, 8)],
        edge_src_to_mid={(1, (8, 11)): 0.2, (1, (9, 8)): -0.1},
        edge_mid_to_dst={((8, 11), 99): 0.4, ((9, 8), 99): -0.05},
        src_taylor={1: 0.05},
        dst_taylor={99: -0.07},
        min_abs_edge=0.0,
    )
    text = buf.getvalue()
    assert "digraph sae_three_node" in text
    assert "src_1" in text and "mid_9_8" in text and "dst_99" in text
