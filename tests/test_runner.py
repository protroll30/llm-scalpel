"""Tests for ``ExperimentRunner``."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig, utilities as utils

from causal_patcher.runner import ExperimentRunner
from causal_patcher.targets import PatchTarget


def test_mock_model_patch_invokes_hook_with_get_act_name():
    hook = utils.get_act_name("resid_pre", 0)
    model = MagicMock()
    model.cfg = SimpleNamespace(n_layers=1, n_heads=2)
    model.to_tokens = MagicMock(return_value=torch.ones(1, 3, dtype=torch.long))

    def run_with_cache(tokens, names_filter=None, return_type="logits", return_cache_object=False, **kwargs):
        logits = torch.zeros(1, 3, 10)
        cache = {hook: torch.ones(1, 3, 4) * 2.0}
        return logits, cache

    model.run_with_cache = run_with_cache
    captured: list = []

    def run_with_hooks(tokens, fwd_hooks, return_type="logits", **kw):
        captured.extend(fwd_hooks)
        act = torch.ones(1, 3, 4) * 9.0
        for name, fn in fwd_hooks:
            act = fn(act, SimpleNamespace(name=name))
        return torch.zeros(1, 3, 10)

    model.run_with_hooks = run_with_hooks

    r = ExperimentRunner(model, "a", "b", clean_answer_token=2, corrupt_answer_token=3, names_filter=[hook])
    out = r.patch_clean_into_corrupt(PatchTarget("resid_pre", 0))
    assert len(captured) == 1
    assert captured[0][0] == hook
    assert out.shape == (1, 3, 10)


def _tiny_model():
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_head=16,
        d_mlp=64,
        n_ctx=64,
        d_vocab=50257,
        act_fn="gelu",
        normalization_type="LN",
        default_prepend_bos=False,
        tokenizer_name="gpt2",
    )
    return HookedTransformer(cfg)


def test_logit_diff_shape():
    model = _tiny_model()
    logits = torch.randn(1, 5, model.cfg.d_vocab)
    r = ExperimentRunner(
        model,
        "ab",
        "cd",
        clean_answer_token=10,
        corrupt_answer_token=20,
        run_baselines=False,
    )
    d = r.logit_diff(logits)
    assert d.shape == torch.Size([])


def test_patch_clean_into_corrupt_runs():
    torch.manual_seed(0)
    model = _tiny_model()
    names = ExperimentRunner.all_patch_hook_names(model.cfg.n_layers)
    r = ExperimentRunner(
        model,
        "a a",
        "b b",
        clean_answer_token=1,
        corrupt_answer_token=2,
        names_filter=names,
    )
    hook = utils.get_act_name("resid_pre", 0)
    logits = r.patch_clean_into_corrupt(PatchTarget("resid_pre", 0))
    assert logits.shape == r.corrupt_logits.shape
    assert hook in r.clean_cache


def test_shape_mismatch_raises():
    model = MagicMock()
    model.cfg = SimpleNamespace(n_layers=1, n_heads=1)
    model.to_tokens = MagicMock(
        side_effect=[
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 2]]),
        ]
    )
    model.run_with_cache = MagicMock()
    with pytest.raises(ValueError, match="same shape"):
        ExperimentRunner(
            model,
            "a",
            "b",
            clean_answer_token=0,
            corrupt_answer_token=1,
            run_baselines=True,
            names_filter=None,
        )


def test_missing_cache_key_raises():
    torch.manual_seed(1)
    model = _tiny_model()
    r = ExperimentRunner(
        model,
        "x x",
        "y y",
        clean_answer_token=3,
        corrupt_answer_token=4,
        names_filter=[utils.get_act_name("resid_pre", 0)],
    )
    missing = utils.get_act_name("resid_post", 1)
    with pytest.raises(KeyError, match=missing):
        r.patch_clean_into_corrupt(PatchTarget("resid_post", 1))


def test_head_patch_changes_logits():
    torch.manual_seed(2)
    model = _tiny_model()
    names = ExperimentRunner.all_patch_hook_names(model.cfg.n_layers)
    r = ExperimentRunner(
        model,
        "c c",
        "d d",
        clean_answer_token=5,
        corrupt_answer_token=6,
        names_filter=names,
    )
    base = r.logit_diff(r.corrupt_logits)
    patched = r.logit_diff(
        r.patch_clean_into_corrupt(PatchTarget("attn_head_z", 0, head=0), positions=None)
    )
    assert patched.shape == base.shape
    assert torch.isfinite(torch.as_tensor(patched))
    assert torch.isfinite(torch.as_tensor(base))
