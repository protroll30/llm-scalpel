"""Layer-by-position and layer-by-head patching heatmaps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from causal_patcher.targets import PatchKind, PatchTarget

if TYPE_CHECKING:
    from causal_patcher.runner import ExperimentRunner


def sweep_layer_position_logit_diff(
    runner: "ExperimentRunner", *, kind: PatchKind = "resid_pre"
) -> np.ndarray:
    """Patch at each (layer, position); cell value is patched corrupt logit difference."""
    n_layers = runner.model.cfg.n_layers
    seq_len = int(runner.corrupt_tokens.shape[-1])  # type: ignore[union-attr]
    grid = np.zeros((n_layers, seq_len))
    if kind == "attn_head_z":
        raise ValueError("Use sweep_layer_head_logit_diff for attn_head_z")
    for layer in range(n_layers):
        for pos in range(seq_len):
            target = PatchTarget(kind, layer)
            logits = runner.patch_clean_into_corrupt(target, positions=pos)
            grid[layer, pos] = float(runner.logit_diff(logits).detach().cpu())
    return grid


def sweep_layer_head_logit_diff(
    runner: "ExperimentRunner",
    *,
    positions: int | slice | None = None,
) -> np.ndarray:
    """Patch each attention head's ``hook_z`` slice; cell is patched corrupt logit difference."""
    n_layers = runner.model.cfg.n_layers
    n_heads = runner.model.cfg.n_heads
    grid = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        for head in range(n_heads):
            target = PatchTarget("attn_head_z", layer, head=head)
            logits = runner.patch_clean_into_corrupt(target, positions=positions)
            grid[layer, head] = float(runner.logit_diff(logits).detach().cpu())
    return grid


def plot_heatmap(
    data: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str = "",
    figsize: tuple[float, float] = (8.0, 4.0),
    cmap: str = "RdBu_r",
    center_zero: bool = True,
):
    """Render a simple ``imshow`` heatmap; returns ``(fig, ax, im)``."""
    fig, ax = plt.subplots(figsize=figsize)
    kwargs: dict = {"cmap": cmap, "aspect": "auto"}
    if center_zero:
        lim = float(np.nanmax(np.abs(data))) or 1.0
        kwargs["vmin"], kwargs["vmax"] = -lim, lim
    im = ax.imshow(data, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax, im


def plot_layer_position_patching(
    runner: "ExperimentRunner",
    *,
    kind: PatchKind = "resid_pre",
    title: str | None = None,
):
    """Run ``sweep_layer_position_logit_diff`` and plot. Returns ``(fig, ax, im, grid)``."""
    grid = sweep_layer_position_logit_diff(runner, kind=kind)
    t = title or f"Logit diff (patched corrupt): {kind}"
    fig, ax, im = plot_heatmap(
        grid,
        xlabel="position",
        ylabel="layer",
        title=t,
    )
    return fig, ax, im, grid


def plot_layer_head_patching(
    runner: "ExperimentRunner",
    *,
    positions: int | slice | None = None,
    title: str | None = None,
):
    """Run ``sweep_layer_head_logit_diff`` and plot. Returns ``(fig, ax, im, grid)``."""
    grid = sweep_layer_head_logit_diff(runner, positions=positions)
    t = title or "Logit diff (patched corrupt, attn head z)"
    fig, ax, im = plot_heatmap(
        grid,
        xlabel="head",
        ylabel="layer",
        title=t,
    )
    return fig, ax, im, grid
