"""Rank attention heads via marginal activation patching on ``hook_z`` (clean → corrupt).

Each score is ``metric(patched_logits) - metric(corrupt_baseline)``.

For ``metric='logit_diff'``, a **positive** marginal means patching that head's clean slice into
the corrupt forward **increases** ``logit(clean_answer) - logit(corrupt_answer)`` at
``metric_seq_pos`` versus the corrupt baseline (activation patching indirect effect).
"""

from __future__ import annotations

from typing import Callable, Literal, Sequence

import torch
from transformer_lens import utils as tl_utils

from causal_patcher.runner import ExperimentRunner
from causal_patcher.targets import PatchPos, PatchTarget

Metric = Literal["logit_diff", "clean_logit", "corrupt_logit"]


def hook_z_names_filter(layers: Sequence[int]) -> Callable[[str], bool]:
    """Return ``names_filter`` caching only ``hook_z`` for the listed transformer layers."""

    wanted = frozenset(tl_utils.get_act_name("z", int(L)) for L in layers)
    return lambda name: name in wanted


def metric_tensor(
    runner: ExperimentRunner,
    logits: torch.Tensor,
    *,
    metric: Metric,
    seq_pos: int,
) -> torch.Tensor:
    """Scalar tensor for ``metric`` at ``seq_pos`` (supports negative ``seq_pos``)."""

    if logits.dim() == 3:
        row = logits[0]
    elif logits.dim() == 2:
        row = logits
    else:
        raise ValueError(f"Expected logits rank 2 or 3, got shape {tuple(logits.shape)}")
    pos = int(seq_pos)
    if pos < 0:
        pos += int(row.shape[0])
    if metric == "logit_diff":
        return runner.logit_diff(logits, seq_pos=seq_pos)
    if metric == "clean_logit":
        return row[pos, runner.clean_answer_token]
    if metric == "corrupt_logit":
        return row[pos, runner.corrupt_answer_token]
    raise ValueError(f"unknown metric: {metric!r}")


def marginal_head_patch_effects(
    runner: ExperimentRunner,
    *,
    layers: Sequence[int],
    patch_positions: PatchPos = -1,
    metric: Metric = "logit_diff",
    metric_seq_pos: int = -1,
) -> dict[tuple[int, int], float]:
    """Marginal patch effect per ``(layer, head)`` over ``layers``.

    ``patch_positions`` is forwarded to :meth:`ExperimentRunner.patch_clean_into_corrupt`
    (aligned index, ``None`` for full sequence, or ``(clean_pos, corrupt_pos)``).
    """

    if runner.corrupt_logits is None:
        raise RuntimeError("runner.corrupt_logits missing; run baselines first.")

    base = metric_tensor(
        runner,
        runner.corrupt_logits,
        metric=metric,
        seq_pos=metric_seq_pos,
    )
    base_f = float(base.detach().cpu().item())

    out: dict[tuple[int, int], float] = {}
    n_heads = int(runner.model.cfg.n_heads)

    for layer in layers:
        L = int(layer)
        for head in range(n_heads):
            target = PatchTarget("attn_head_z", L, head=head)
            patched_logits = runner.patch_clean_into_corrupt(target, positions=patch_positions)
            v = metric_tensor(
                runner,
                patched_logits,
                metric=metric,
                seq_pos=metric_seq_pos,
            )
            out[(L, head)] = float(v.detach().cpu().item()) - base_f

    return out


def rank_heads(
    scores: dict[tuple[int, int], float],
    *,
    top_k: int | None = None,
    by_abs: bool = True,
) -> list[tuple[tuple[int, int], float]]:
    """Sort ``(layer, head) -> score`` descending by ``|score|`` (default) or signed score."""

    items = list(scores.items())
    if by_abs:
        items.sort(key=lambda x: abs(x[1]), reverse=True)
    else:
        items.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        items = items[: int(top_k)]
    return items
