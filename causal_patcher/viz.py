"""Layer-by-position and layer-by-head patching heatmaps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np

from causal_patcher.targets import PatchKind, PatchPos, PatchTarget

if TYPE_CHECKING:
    from causal_patcher.runner import ExperimentRunner

AxisTokenSource = Literal["corrupt", "clean"]


def position_tick_labels(
    runner: "ExperimentRunner", which: AxisTokenSource = "corrupt"
) -> list[str]:
    """Decode one label per **prompt position** for axis ticks (model tokenizer, batch row 0).

    Use the **corrupt** prompt (default) when the heatmap x-axis is corrupt sequence position, which
    is the usual case for patched-forward runs. Use **clean** when the grid is indexed by clean
    positions.
    """
    toks = runner.corrupt_tokens if which == "corrupt" else runner.clean_tokens
    if toks is None:
        raise ValueError("Runner has no tokenized prompts; run baselines first.")
    if not hasattr(runner.model, "tokenizer") or runner.model.tokenizer is None:
        raise ValueError("The model has no tokenizer; pass labels manually to plot_heatmap().")

    labels: list[str] = []
    for tid in toks[0].cpu().tolist():
        s = runner.model.tokenizer.decode([int(tid)])
        labels.append(s if s else f"<id {tid}>")
    return labels


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
    positions: PatchPos = None,
) -> np.ndarray:
    """Patch each attention head's ``hook_z`` slice; cell is patched corrupt logit difference.

    ``positions`` is passed through to :meth:`~causal_patcher.runner.ExperimentRunner.patch_clean_into_corrupt`
    (e.g. a single index, or ``(clean_index, corrupt_index)`` for explicit alignment).
    """
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
    x_tick_labels: list[str] | None = None,
    x_tick_rotation: float | None = None,
    x_tick_fontsize: float | None = None,
) -> tuple:
    """Render a simple ``imshow`` heatmap; returns ``(fig, ax, im)``.

    For logit-difference data with the default colormap, **red / warm** typically indicates the
    patch *increased* clean-minus-corrupt log odds (**recovery** toward the clean answer); **blue
    / cool** indicates a *decrease* (**suppression**). See ``docs/heatmap.md`` in the repository.
    """
    fig, ax = plt.subplots(figsize=figsize)
    kwargs: dict = {"cmap": cmap, "aspect": "auto"}
    if center_zero:
        lim = float(np.nanmax(np.abs(data))) or 1.0
        kwargs["vmin"], kwargs["vmax"] = -lim, lim
    n_x = int(data.shape[1])
    im = ax.imshow(data, **kwargs)
    if x_tick_labels is not None:
        if len(x_tick_labels) != n_x:
            raise ValueError(
                f"x_tick_labels length {len(x_tick_labels)} != number of x bins {n_x}."
            )
        ax.set_xticks(np.arange(n_x))
        fs = 8.0 if x_tick_fontsize is None else x_tick_fontsize
        rot = 45.0 if x_tick_rotation is None else x_tick_rotation
        ax.set_xticklabels(
            x_tick_labels, rotation=rot, ha="right" if rot else "center", fontsize=fs
        )
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
    figsize: tuple[float, float] = (8.0, 4.0),
    label_x_with_tokens: bool = True,
    x_token_source: AxisTokenSource = "corrupt",
    xlabel: str | None = None,
) -> tuple:
    """Run ``sweep_layer_position_logit_diff`` and plot. Returns ``(fig, ax, im, grid)``.

    By default, the x-axis is labeled with **decoded subwords** for each position (from
    :func:`position_tick_labels`), so you do not have to hand-build tick strings.
    Set ``label_x_with_tokens=False`` to use a plain index axis.
    """
    grid = sweep_layer_position_logit_diff(runner, kind=kind)
    t = title or f"Logit diff (patched corrupt): {kind}"
    if label_x_with_tokens:
        x_tick_labels = position_tick_labels(runner, x_token_source)
        xl = (
            xlabel
            or f"Position (tokens from {x_token_source} prompt)"
        )
    else:
        x_tick_labels = None
        xl = xlabel or "Position index"
    fig, ax, im = plot_heatmap(
        grid,
        xlabel=xl,
        ylabel="layer",
        title=t,
        figsize=figsize,
        x_tick_labels=x_tick_labels,
    )
    return fig, ax, im, grid


def plot_layer_head_patching(
    runner: "ExperimentRunner",
    *,
    positions: PatchPos = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 4.0),
) -> tuple:
    """Run ``sweep_layer_head_logit_diff`` and plot. Returns ``(fig, ax, im, grid)``."""
    grid = sweep_layer_head_logit_diff(runner, positions=positions)
    t = title or "Logit diff (patched corrupt, attn head z)"
    fig, ax, im = plot_heatmap(
        grid,
        xlabel="head",
        ylabel="layer",
        title=t,
        figsize=figsize,
    )
    return fig, ax, im, grid
