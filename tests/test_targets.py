"""Tests for ``PatchTarget`` and hook name resolution."""

import pytest
from transformer_lens import utilities as utils

from causal_patcher.targets import PatchTarget, patch_hook_name


@pytest.mark.parametrize(
    ("kind", "layer", "expected_suffix"),
    [
        ("resid_pre", 2, "blocks.2.hook_resid_pre"),
        ("resid_mid", 0, "blocks.0.hook_resid_mid"),
        ("resid_post", 1, "blocks.1.hook_resid_post"),
        ("mlp_out", 4, "blocks.4.hook_mlp_out"),
    ],
)
def test_patch_hook_name_matches_utils(kind, layer, expected_suffix):
    assert patch_hook_name(kind, layer) == expected_suffix
    assert patch_hook_name(kind, layer) == utils.get_act_name(kind, layer)


def test_attn_head_z_uses_hook_z():
    layer = 3
    assert patch_hook_name("attn_head_z", layer) == utils.get_act_name("z", layer)


def test_patch_target_hook_name():
    t = PatchTarget("mlp_out", 5)
    assert t.hook_name() == utils.get_act_name("mlp_out", 5)


def test_attn_head_z_requires_head():
    with pytest.raises(ValueError, match="head"):
        PatchTarget("attn_head_z", 0)
    t = PatchTarget("attn_head_z", 0, head=1)
    assert t.head == 1


def test_non_head_target_rejects_head():
    with pytest.raises(ValueError, match="head"):
        PatchTarget("resid_pre", 0, head=0)


def test_pos_pair_tuple():
    t = PatchTarget("resid_pre", 0, pos=(1, 2))
    assert t.pos == (1, 2)


def test_pos_tuple_validates_length():
    with pytest.raises(ValueError, match="pos tuple"):
        PatchTarget("resid_pre", 0, pos=(1, 2, 3))
