"""Tests for causal_patcher.head_patch_rank."""

import math

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils

from causal_patcher.head_patch_rank import hook_z_names_filter, marginal_head_patch_effects, metric_tensor
from causal_patcher.runner import ExperimentRunner


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


@pytest.mark.parametrize("layer", [0])
def test_marginal_head_patch_effect_finite(layer: int):
    torch.manual_seed(0)
    model = _tiny_model()
    hook_z = utils.get_act_name("z", layer)
    filt = hook_z_names_filter([layer])

    r = ExperimentRunner(
        model,
        "a a",
        "b b",
        clean_answer_token=1,
        corrupt_answer_token=2,
        names_filter=filt,
        prepend_bos=False,
    )
    scores = marginal_head_patch_effects(r, layers=[layer], patch_positions=-1, metric_seq_pos=-1)
    assert len(scores) == model.cfg.n_heads
    for v in scores.values():
        assert math.isfinite(v)


def test_metric_tensor_matches_logit_diff_runner():
    model = _tiny_model()
    logits = torch.randn(1, 5, model.cfg.d_vocab)
    r = ExperimentRunner(model, "x", "y", clean_answer_token=10, corrupt_answer_token=20, run_baselines=False)
    a = metric_tensor(r, logits, metric="logit_diff", seq_pos=-1)
    b = r.logit_diff(logits, seq_pos=-1)
    assert torch.allclose(a, b)


def test_prepend_bos_propagates_to_to_tokens():
    hook = utils.get_act_name("resid_pre", 0)
    calls = []

    class _M(torch.nn.Module):
        cfg = type("C", (), {"n_layers": 1, "n_heads": 2})()

        def to_tokens(self, s, **kw):
            calls.append(kw)
            return torch.ones(1, 3, dtype=torch.long)

        def run_with_cache(self, tokens, names_filter=None, return_type="logits", **kwargs):
            logits = torch.zeros(1, 3, 10)
            cache = {hook: torch.ones(1, 3, 32)}
            return logits, cache

    model = _M()
    ExperimentRunner(
        model,
        "a",
        "b",
        clean_answer_token=0,
        corrupt_answer_token=1,
        names_filter=[hook],
        prepend_bos=False,
    )
    assert calls and calls[0].get("prepend_bos") is False
