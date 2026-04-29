"""Tests for patching heatmaps."""

import matplotlib

matplotlib.use("Agg")

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig, utilities as utils

from causal_patcher.runner import ExperimentRunner
from causal_patcher import viz


def _tiny_model():
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=24,
        n_heads=2,
        d_head=12,
        d_mlp=48,
        n_ctx=32,
        d_vocab=50257,
        act_fn="gelu",
        normalization_type="LN",
        default_prepend_bos=False,
        tokenizer_name="gpt2",
    )
    return HookedTransformer(cfg)


def test_sweep_layer_position_grid_shape():
    torch.manual_seed(0)
    model = _tiny_model()
    names = ExperimentRunner.all_patch_hook_names(model.cfg.n_layers)
    r = ExperimentRunner(
        model,
        "e e",
        "f f",
        clean_answer_token=1,
        corrupt_answer_token=2,
        names_filter=names,
    )
    g = viz.sweep_layer_position_logit_diff(r, kind="resid_mid")
    assert g.shape == (model.cfg.n_layers, r.corrupt_tokens.shape[-1])


def test_sweep_layer_head_grid_shape():
    torch.manual_seed(1)
    model = _tiny_model()
    names = ExperimentRunner.all_patch_hook_names(model.cfg.n_layers)
    r = ExperimentRunner(
        model,
        "g g",
        "h h",
        clean_answer_token=3,
        corrupt_answer_token=4,
        names_filter=names,
    )
    g = viz.sweep_layer_head_logit_diff(r, positions=0)
    assert g.shape == (model.cfg.n_layers, model.cfg.n_heads)


def test_plot_functions_return_figure():
    torch.manual_seed(2)
    model = _tiny_model()
    names = [utils.get_act_name("resid_pre", L) for L in range(model.cfg.n_layers)]
    r = ExperimentRunner(
        model,
        "i i",
        "j j",
        clean_answer_token=5,
        corrupt_answer_token=6,
        names_filter=names,
    )
    fig, ax, im, grid = viz.plot_layer_position_patching(r, kind="resid_pre")
    assert grid.ndim == 2
    fig.savefig(io := __import__("io").BytesIO(), format="png")
    assert io.tell() > 0
