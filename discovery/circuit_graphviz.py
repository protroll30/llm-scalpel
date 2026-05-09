"""Graphviz (DOT) export for bipartite SAE latent “circuits” across two layers.

Edges combine (1) a **forward causal effect** of intervening on one source latent at layer A
(at ``src_seq_pos``) with (2) the **loss gradient** on destination latents at layer B at
``dst_seq_pos``. Positions may differ (e.g. country token → later bottleneck after attention).
The scalar loss (e.g. logit-diff) uses whatever index the caller passes into
``logits_to_scalar_loss``—typically the prediction position, independent of ``src_seq_pos``.

For source *i* and destination *j*:

    weight[i→j] ≈ (Δ f_j) · (∂ℒ / ∂ f_j)

where Δ f_j is the change in the layer-B SAE latent *j* when only latent *i* is perturbed at
layer A (decode reconstruction hook), and ∂ℒ/∂ f_j comes from the corrupt forward of the same
scalar loss used in :func:`discovery.attribution.feature_attribution_pass`.

Positive weights are drawn as **promoter** (green) strokes; negative as **suppressor** (red).
This is a **first-order local** summary, not a full Jacobian—use it for visualization and
hypothesis generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, IO, Mapping, Sequence, Union

import torch

from causal_patcher.runner import _patch_fn, _resolve_patch_pos
from causal_patcher.targets import PatchTarget
from discovery.attribution import feature_attribution_pass
from transformer_lens import utils as tl_utils

PathLike = Union[str, Path]


def dot_escape_label(text: str) -> str:
    """Escape text for use inside DOT double-quoted labels.

    Real newlines in ``text`` (Python ``"\\n"``, one char) become the DOT
    line-break escape ``\\n`` (two chars). User-supplied backslashes and
    quotes are escaped first so they survive verbatim.
    """

    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _edge_label(weight: float, hide_below: float = 5e-3) -> str:
    """Edge label: empty string for near-zero weights, otherwise ``+0.286`` style.

    ``{:.3g}`` rounded sub-millisigned weights to ``"0"`` / ``"-0"`` which
    cluttered the graph without conveying anything; suppress those labels and
    keep the edge itself (color + penwidth still encode magnitude).
    """

    if abs(float(weight)) < float(hide_below):
        return ""
    return f"{float(weight):+.3g}"


def _interp_rgb(c0: tuple[int, int, int], c1: tuple[int, int, int], t: float) -> str:
    """Linear blend between two RGB 0–255 triples → #rrggbb."""

    t = max(0.0, min(1.0, float(t)))
    r = int(round(c0[0] + (c1[0] - c0[0]) * t))
    g = int(round(c0[1] + (c1[1] - c0[1]) * t))
    b = int(round(c0[2] + (c1[2] - c0[2]) * t))
    return f"#{r:02x}{g:02x}{b:02x}"


def score_to_fillcolor(score: float, max_abs: float) -> str:
    """White → green (promoter, score > 0) or white → red (suppressor), scaled by |score|."""

    den = max(float(max_abs), 1e-12)
    t = min(1.0, abs(float(score)) / den)
    if score >= 0.0:
        return _interp_rgb((255, 255, 255), (46, 204, 113), t)
    return _interp_rgb((255, 255, 255), (231, 76, 60), t)


@dataclass(frozen=True)
class CrossLayerEdgeBuild:
    """Result of :func:`build_cross_layer_edges`."""

    edge_weight: dict[tuple[int, int], float]
    """(src_feature_id, dst_feature_id) → Δf_dst · (∂ℒ/∂f_dst)."""

    dst_gradient_g: torch.Tensor
    """Full ∂ℒ/∂f at the destination hook (corrupt latents), shape ``[n_features_dst]``."""

    src_taylor: dict[int, float]
    """Per-source ``(Δf)·(∂ℒ/∂f)`` from Taylor attribution at the source hook."""

    dst_taylor: dict[int, float]
    """Per-destination ``(Δf)·(∂ℒ/∂f)`` at the destination hook (same pass metadata)."""

    src_seq_pos_resolved: int
    """0-based index used for source attribution and intervention."""

    dst_seq_pos_resolved: int
    """0-based index used for destination attribution and Δf readout."""


@dataclass(frozen=True)
class ThreeNodeEdgeBuild:
    """Latent → ``hook_z`` head slice → latent tripartite scores."""

    edge_src_to_mid: dict[tuple[int, tuple[int, int]], float]
    """(src_feature_id, (layer, head)) → Δz·∂ℒ/∂z on corrupt run."""

    edge_mid_to_dst: dict[tuple[tuple[int, int], int], float]
    """((layer, head), dst_feature_id) → Δf·∂ℒ/∂f when patching clean→corrupt on that head only."""

    bipartite: CrossLayerEdgeBuild
    """Direct latent→latent build (same Taylors / dst gradient); bipartite edges kept for comparison."""

    z_seq_pos_resolved: int
    """Token index where ``z`` slices and (by default) head patching are read."""

    middle_heads: tuple[tuple[int, int], ...]


def _resolve_pos(seq_pos: int, seq_len: int) -> int:
    """Map possibly-negative token index to ``0 .. seq_len-1``."""

    pos = int(seq_pos)
    raw = pos
    if pos < 0:
        pos += int(seq_len)
    if pos < 0 or pos >= int(seq_len):
        raise IndexError(
            f"seq_pos {raw!r} resolves to {pos}, invalid for seq_len={seq_len} "
            f"(valid indices: 0..{seq_len - 1}, or negatives down to -{seq_len}). "
            "Indices are 0-based; if you used a 1-based token number or forgot BOS shift, subtract one."
        )
    return pos


def _encode_f_at_pos(
    *,
    model: Any,
    tokens: torch.Tensor,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    seq_pos: int,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name], return_type="logits")
        if hook_name not in cache:
            raise KeyError(f"Cache missing {hook_name!r}")
        act = cache[hook_name]
        f_raw = encode_fn(act)
        if f_raw.dim() == 2:
            f_raw = f_raw.unsqueeze(0)
        if f_raw.dim() < 3:
            raise ValueError(
                f"encode_fn must return [pos, n_features] or [batch, pos, n_features]; got {tuple(f_raw.shape)}"
            )
        p = _resolve_pos(seq_pos, int(f_raw.shape[1]))
        return f_raw[0, p, :].detach().clone()


@torch.no_grad()
def _run_single_src_intervention(
    *,
    model: Any,
    tokens: torch.Tensor,
    src_hook: str,
    dst_hook: str,
    encode_src: Callable[[torch.Tensor], torch.Tensor],
    decode_src: Callable[[torch.Tensor], torch.Tensor],
    encode_dst: Callable[[torch.Tensor], torch.Tensor],
    src_pos_eff: int,
    dst_pos_eff: int,
    src_feature_id: int,
    mode: str,
    value: float,
    counterfactual_scale: float,
    base_f_src_at_pos: torch.Tensor,
    debug_zero_act: bool,
) -> torch.Tensor:
    """Apply one source intervention at ``src_pos_eff``; return dst latents at ``dst_pos_eff``."""

    from discovery.sae_scout import reconstruct_activation

    inject_idx = torch.tensor([int(src_feature_id)], device=tokens.device, dtype=torch.long)
    base_inject = base_f_src_at_pos.index_select(0, inject_idx).detach()
    cf_scale = float(counterfactual_scale)

    def _hook_src(act: torch.Tensor, hook) -> torch.Tensor:
        if debug_zero_act:
            return torch.zeros_like(act)

        f = encode_src(act)
        if f.dim() == 2:
            f = f.unsqueeze(0)
        f_full = f
        f_work = f_full.clone()
        if inject_idx.numel() > 0:
            if cf_scale > 0.0:
                if mode == "set":
                    target = base_inject.to(device=f_work.device, dtype=f_work.dtype) * cf_scale
                    f_work[0, src_pos_eff, inject_idx] = target
                else:
                    delta_inj = base_inject.to(device=f_work.device, dtype=f_work.dtype) * (cf_scale - 1.0)
                    f_work[0, src_pos_eff, inject_idx] = (
                        f_work[0, src_pos_eff, inject_idx] + delta_inj
                    )
            else:
                if mode == "set":
                    f_work[0, src_pos_eff, inject_idx] = float(value)
                else:
                    f_work[0, src_pos_eff, inject_idx] = f_work[0, src_pos_eff, inject_idx] + float(
                        value
                    )

        return reconstruct_activation(
            f_patched=f_work,
            x_corrupt=act,
            f_corrupt=f_full,
            decode_fn=decode_src,
        )

    act_dst: torch.Tensor | None = None

    def _capture_dst(act: torch.Tensor, hook) -> torch.Tensor:
        nonlocal act_dst
        act_dst = act.detach()
        return act

    model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (str(src_hook), _hook_src),
            (str(dst_hook), _capture_dst),
        ],
        return_type="logits",
    )

    if act_dst is None:
        raise RuntimeError(f"Failed to capture activation at {dst_hook!r}.")

    f_dst_raw = encode_dst(act_dst)
    if f_dst_raw.dim() == 2:
        f_dst_raw = f_dst_raw.unsqueeze(0)
    return f_dst_raw[0, dst_pos_eff, :].detach().clone()


def build_cross_layer_edges(
    *,
    model: Any,
    prompt_clean: str,
    prompt_corrupt: str,
    src_hook: str,
    dst_hook: str,
    encode_src: Callable[[torch.Tensor], torch.Tensor],
    decode_src: Callable[[torch.Tensor], torch.Tensor],
    encode_dst: Callable[[torch.Tensor], torch.Tensor],
    decode_dst: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
    src_feature_ids: Sequence[int],
    dst_feature_ids: Sequence[int],
    metric: str = "logit_diff",
    seq_pos: int = -1,
    src_seq_pos: int | None = None,
    dst_seq_pos: int | None = None,
    prepend_bos: bool | None = False,
    device: torch.device | None = None,
    intervention_mode: str = "set",
    intervention_value: float = 5.0,
    counterfactual_scale: float = 0.0,
    debug_zero_act: bool = False,
) -> CrossLayerEdgeBuild:
    """Interventions on ``src_hook`` latents × gradient on ``dst_hook`` latents (corrupt run).

    Interventions are evaluated on **prompt_corrupt** so they align with the corrupt forward
    used to obtain ∂ℒ/∂f at the destination hook inside :func:`feature_attribution_pass`.

    Use ``src_seq_pos`` / ``dst_seq_pos`` to read attributions and Δf at different token indices
    (e.g. layer-8 country token vs layer-9 ``is`` token). When omitted, both fall back to ``seq_pos``.
    """

    to_tokens_kw: dict = {}
    if prepend_bos is not None:
        to_tokens_kw["prepend_bos"] = bool(prepend_bos)
    corrupt_tokens = model.to_tokens(prompt_corrupt, **to_tokens_kw)
    if device is not None:
        corrupt_tokens = corrupt_tokens.to(device)

    seq_len = int(corrupt_tokens.shape[-1])
    src_seq_raw = int(src_seq_pos) if src_seq_pos is not None else int(seq_pos)
    dst_seq_raw = int(dst_seq_pos) if dst_seq_pos is not None else int(seq_pos)
    src_pos_eff = _resolve_pos(src_seq_raw, seq_len)
    dst_pos_eff = _resolve_pos(dst_seq_raw, seq_len)

    pass_dst = feature_attribution_pass(
        model=model,
        prompt_clean=prompt_clean,
        prompt_corrupt=prompt_corrupt,
        hook_name=str(dst_hook),
        encode_fn=encode_dst,
        decode_fn=decode_dst,
        logits_to_scalar_loss=logits_to_scalar_loss,
        metric=metric,
        seq_pos=int(dst_seq_raw),
        prepend_bos=prepend_bos,
        device=device,
    )
    g_dst = pass_dst.components.gradient_g
    dst_taylor = {int(j): float(pass_dst.scores[j].detach().cpu().item()) for j in dst_feature_ids}

    pass_src = feature_attribution_pass(
        model=model,
        prompt_clean=prompt_clean,
        prompt_corrupt=prompt_corrupt,
        hook_name=str(src_hook),
        encode_fn=encode_src,
        decode_fn=decode_src,
        logits_to_scalar_loss=logits_to_scalar_loss,
        metric=metric,
        seq_pos=int(src_seq_raw),
        prepend_bos=prepend_bos,
        device=device,
    )
    src_taylor = {int(i): float(pass_src.scores[i].detach().cpu().item()) for i in src_feature_ids}

    base_f_src_at_pos = _encode_f_at_pos(
        model=model,
        tokens=corrupt_tokens,
        hook_name=str(src_hook),
        encode_fn=encode_src,
        seq_pos=int(src_seq_raw),
    )
    base_f_dst = _encode_f_at_pos(
        model=model,
        tokens=corrupt_tokens,
        hook_name=str(dst_hook),
        encode_fn=encode_dst,
        seq_pos=int(dst_seq_raw),
    )

    edge_weight: dict[tuple[int, int], float] = {}
    g_cpu = g_dst.detach().float()

    for i in src_feature_ids:
        int_f_dst = _run_single_src_intervention(
            model=model,
            tokens=corrupt_tokens,
            src_hook=str(src_hook),
            dst_hook=str(dst_hook),
            encode_src=encode_src,
            decode_src=decode_src,
            encode_dst=encode_dst,
            src_pos_eff=src_pos_eff,
            dst_pos_eff=dst_pos_eff,
            src_feature_id=int(i),
            mode=str(intervention_mode),
            value=float(intervention_value),
            counterfactual_scale=float(counterfactual_scale),
            base_f_src_at_pos=base_f_src_at_pos,
            debug_zero_act=bool(debug_zero_act),
        )
        delta = (int_f_dst - base_f_dst).detach().float()
        for j in dst_feature_ids:
            jj = int(j)
            w = float((delta[jj] * g_cpu[jj]).detach().cpu().item())
            edge_weight[(int(i), jj)] = w

    return CrossLayerEdgeBuild(
        edge_weight=edge_weight,
        dst_gradient_g=g_dst.detach(),
        src_taylor=src_taylor,
        dst_taylor=dst_taylor,
        src_seq_pos_resolved=src_pos_eff,
        dst_seq_pos_resolved=dst_pos_eff,
    )


def _grad_z_corrupt(
    *,
    model: Any,
    corrupt_tokens: torch.Tensor,
    z_hook_name: str,
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """∂ℒ/∂z on corrupt forward at ``z_hook_name`` (full tensor, batch=0)."""

    z_leaf: torch.Tensor | None = None

    def repl(act: torch.Tensor, hook) -> torch.Tensor:
        nonlocal z_leaf
        z_r = act.detach().clone().requires_grad_(True)
        z_leaf = z_r
        return z_r

    model.zero_grad(set_to_none=True)
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(str(z_hook_name), repl)],
                return_type="logits",
            )
            loss = logits_to_scalar_loss(logits)
            loss.backward()
    finally:
        model.train(was_training)

    if z_leaf is None or z_leaf.grad is None:
        raise RuntimeError(f"No gradient at z hook {z_hook_name!r} (graph blocked?).")
    return z_leaf.grad.detach()


@torch.no_grad()
def _corrupt_z_snapshot(
    *,
    model: Any,
    corrupt_tokens: torch.Tensor,
    z_hook_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    _, cache = model.run_with_cache(
        corrupt_tokens,
        names_filter=list(z_hook_names),
        return_type="logits",
    )
    out: dict[str, torch.Tensor] = {}
    for name in z_hook_names:
        if name not in cache:
            raise KeyError(f"Corrupt cache missing {name!r}")
        out[name] = cache[name].detach()
    return out


@torch.no_grad()
def _run_src_intervention_capture_z_hooks(
    *,
    model: Any,
    corrupt_tokens: torch.Tensor,
    src_hook: str,
    encode_src: Callable[[torch.Tensor], torch.Tensor],
    decode_src: Callable[[torch.Tensor], torch.Tensor],
    src_pos_eff: int,
    src_feature_id: int,
    mode: str,
    value: float,
    counterfactual_scale: float,
    base_f_src_at_pos: torch.Tensor,
    debug_zero_act: bool,
    z_hook_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Same layer-8 style intervention as :func:`_run_single_src_intervention`; snapshot listed ``hook_z`` tensors."""

    from discovery.sae_scout import reconstruct_activation

    inject_idx = torch.tensor([int(src_feature_id)], device=corrupt_tokens.device, dtype=torch.long)
    base_inject = base_f_src_at_pos.index_select(0, inject_idx).detach()
    cf_scale = float(counterfactual_scale)

    def _hook_src(act: torch.Tensor, hook) -> torch.Tensor:
        if debug_zero_act:
            return torch.zeros_like(act)

        f = encode_src(act)
        if f.dim() == 2:
            f = f.unsqueeze(0)
        f_full = f
        f_work = f_full.clone()
        if inject_idx.numel() > 0:
            if cf_scale > 0.0:
                if mode == "set":
                    target = base_inject.to(device=f_work.device, dtype=f_work.dtype) * cf_scale
                    f_work[0, src_pos_eff, inject_idx] = target
                else:
                    delta_inj = base_inject.to(device=f_work.device, dtype=f_work.dtype) * (cf_scale - 1.0)
                    f_work[0, src_pos_eff, inject_idx] = (
                        f_work[0, src_pos_eff, inject_idx] + delta_inj
                    )
            else:
                if mode == "set":
                    f_work[0, src_pos_eff, inject_idx] = float(value)
                else:
                    f_work[0, src_pos_eff, inject_idx] = f_work[0, src_pos_eff, inject_idx] + float(
                        value
                    )

        return reconstruct_activation(
            f_patched=f_work,
            x_corrupt=act,
            f_corrupt=f_full,
            decode_fn=decode_src,
        )

    snaps: dict[str, torch.Tensor] = {}

    def _cap(name: str):
        def _hook(act: torch.Tensor, hook) -> torch.Tensor:
            snaps[name] = act.detach()
            return act

        return _hook

    fwd_hooks = [(str(src_hook), _hook_src)] + [(zh, _cap(zh)) for zh in z_hook_names]
    model.run_with_hooks(corrupt_tokens, fwd_hooks=fwd_hooks, return_type="logits")

    for zh in z_hook_names:
        if zh not in snaps:
            raise RuntimeError(f"Failed to capture {zh!r} during src intervention.")
    return snaps


def build_three_node_edges(
    *,
    model: Any,
    prompt_clean: str,
    prompt_corrupt: str,
    src_hook: str,
    dst_hook: str,
    encode_src: Callable[[torch.Tensor], torch.Tensor],
    decode_src: Callable[[torch.Tensor], torch.Tensor],
    encode_dst: Callable[[torch.Tensor], torch.Tensor],
    decode_dst: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
    src_feature_ids: Sequence[int],
    dst_feature_ids: Sequence[int],
    middle_heads: Sequence[tuple[int, int]],
    metric: str = "logit_diff",
    seq_pos: int = -1,
    src_seq_pos: int | None = None,
    dst_seq_pos: int | None = None,
    z_seq_pos: int | None = None,
    head_patch_positions: int | tuple[int, int] | slice | None = None,
    prepend_bos: bool | None = False,
    device: torch.device | None = None,
    intervention_mode: str = "set",
    intervention_value: float = 5.0,
    counterfactual_scale: float = 0.0,
    debug_zero_act: bool = False,
) -> ThreeNodeEdgeBuild:
    """Tripartite edges: src latent → ``hook_z`` head slice → dst latent.

    ``edge_src_to_mid`` uses Δz·∂ℒ/∂z after an src SAE intervention. ``edge_mid_to_dst`` patches one head’s
    clean ``z`` into the corrupt run (same mechanism as ``causal_patcher`` ``attn_head_z`` patching) and
    applies Δf·∂ℒ/∂f at ``dst_hook``.

    Also returns :class:`CrossLayerEdgeBuild` as ``bipartite`` for the direct latent→latent baseline.
    """

    if not middle_heads:
        raise ValueError("middle_heads must be non-empty for three-node discovery.")

    mh_tuple = tuple((int(L), int(H)) for L, H in middle_heads)
    n_layers = int(getattr(model.cfg, "n_layers", 0))
    n_heads_cfg = int(getattr(model.cfg, "n_heads", 0))
    for L, H in mh_tuple:
        if not (0 <= int(L) < n_layers):
            raise IndexError(
                f"middle-head layer {int(L)} is out of range for this model (n_layers={n_layers}, use 0..{n_layers - 1})."
            )
        if not (0 <= int(H) < n_heads_cfg):
            raise IndexError(
                f"middle-head index {int(H)} is out of range (n_heads={n_heads_cfg}, use 0..{n_heads_cfg - 1}). "
                "Head indices are 0-based like TransformerLens."
            )
    layers_mid = sorted({L for L, _ in mh_tuple})
    z_hooks = [tl_utils.get_act_name("z", L) for L in layers_mid]

    bipartite = build_cross_layer_edges(
        model=model,
        prompt_clean=prompt_clean,
        prompt_corrupt=prompt_corrupt,
        src_hook=str(src_hook),
        dst_hook=str(dst_hook),
        encode_src=encode_src,
        decode_src=decode_src,
        encode_dst=encode_dst,
        decode_dst=decode_dst,
        logits_to_scalar_loss=logits_to_scalar_loss,
        src_feature_ids=src_feature_ids,
        dst_feature_ids=dst_feature_ids,
        metric=metric,
        seq_pos=int(seq_pos),
        src_seq_pos=src_seq_pos,
        dst_seq_pos=dst_seq_pos,
        prepend_bos=prepend_bos,
        device=device,
        intervention_mode=intervention_mode,
        intervention_value=intervention_value,
        counterfactual_scale=counterfactual_scale,
        debug_zero_act=debug_zero_act,
    )

    to_tokens_kw: dict = {}
    if prepend_bos is not None:
        to_tokens_kw["prepend_bos"] = bool(prepend_bos)
    corrupt_tokens = model.to_tokens(prompt_corrupt, **to_tokens_kw)
    clean_tokens = model.to_tokens(prompt_clean, **to_tokens_kw)
    if device is not None:
        corrupt_tokens = corrupt_tokens.to(device)
        clean_tokens = clean_tokens.to(device)

    seq_len = int(corrupt_tokens.shape[-1])
    src_seq_raw = int(src_seq_pos) if src_seq_pos is not None else int(seq_pos)
    dst_seq_raw = int(dst_seq_pos) if dst_seq_pos is not None else int(seq_pos)
    if z_seq_pos is None:
        z_pos_eff = int(bipartite.dst_seq_pos_resolved)
    else:
        z_pos_eff = _resolve_pos(int(z_seq_pos), seq_len)

    if z_pos_eff < 0 or z_pos_eff >= seq_len:
        raise IndexError(
            f"z_seq_pos resolves to {z_pos_eff}, invalid for corrupt seq_len={seq_len} (indices 0..{seq_len - 1})."
        )
    dsi = int(bipartite.dst_seq_pos_resolved)
    if dsi < 0 or dsi >= seq_len:
        raise IndexError(
            f"bipartite dst_seq_pos_resolved={dsi} vs three-node corrupt seq_len={seq_len}. "
            "Usually a prepend_bos / tokenization mismatch between builds—align flags with your prompt."
        )

    z_baselines = _corrupt_z_snapshot(model=model, corrupt_tokens=corrupt_tokens, z_hook_names=z_hooks)

    grad_z_by_layer: dict[int, torch.Tensor] = {}
    for L in layers_mid:
        zh = tl_utils.get_act_name("z", L)
        grad_z_by_layer[L] = _grad_z_corrupt(
            model=model,
            corrupt_tokens=corrupt_tokens,
            z_hook_name=zh,
            logits_to_scalar_loss=logits_to_scalar_loss,
        )

    base_f_src_at_pos = _encode_f_at_pos(
        model=model,
        tokens=corrupt_tokens,
        hook_name=str(src_hook),
        encode_fn=encode_src,
        seq_pos=int(src_seq_raw),
    )

    base_f_dst = _encode_f_at_pos(
        model=model,
        tokens=corrupt_tokens,
        hook_name=str(dst_hook),
        encode_fn=encode_dst,
        seq_pos=int(dst_seq_raw),
    )

    g_cpu = bipartite.dst_gradient_g.detach().float()
    nf_dst = int(base_f_dst.numel())
    nf_src = int(base_f_src_at_pos.numel())
    for j in dst_feature_ids:
        jj = int(j)
        if jj < 0 or jj >= nf_dst:
            raise IndexError(
                f"--dst-features includes {jj} but destination SAE width is {nf_dst} (valid 0..{nf_dst - 1})."
            )
    if int(g_cpu.numel()) != nf_dst:
        raise ValueError(
            f"Destination gradient length {int(g_cpu.numel())} does not match latent width {nf_dst}; "
            "check attribution vs encode_dst."
        )
    for i in src_feature_ids:
        ii = int(i)
        if ii < 0 or ii >= nf_src:
            raise IndexError(
                f"--src-features includes {ii} but source SAE width is {nf_src} (valid 0..{nf_src - 1})."
            )

    edge_src_to_mid: dict[tuple[int, tuple[int, int]], float] = {}
    for i in src_feature_ids:
        snaps = _run_src_intervention_capture_z_hooks(
            model=model,
            corrupt_tokens=corrupt_tokens,
            src_hook=str(src_hook),
            encode_src=encode_src,
            decode_src=decode_src,
            src_pos_eff=int(bipartite.src_seq_pos_resolved),
            src_feature_id=int(i),
            mode=str(intervention_mode),
            value=float(intervention_value),
            counterfactual_scale=float(counterfactual_scale),
            base_f_src_at_pos=base_f_src_at_pos,
            debug_zero_act=bool(debug_zero_act),
            z_hook_names=z_hooks,
        )
        for L, H in mh_tuple:
            zh = tl_utils.get_act_name("z", L)
            zb = z_baselines[zh]
            gz_t = grad_z_by_layer[L]
            try:
                z0 = zb[0, z_pos_eff, int(H), :].float()
                z1 = snaps[zh][0, z_pos_eff, int(H), :].float()
                gz = gz_t[0, z_pos_eff, int(H), :].float()
            except IndexError as err:
                raise IndexError(
                    f"Bad z slice at query_pos={z_pos_eff}, head={int(H)}, layer={int(L)} "
                    f"(corrupt seq_len={seq_len}; z shape={tuple(zb.shape)}; grad_z shape={tuple(gz_t.shape)})."
                ) from err
            dz = z1 - z0
            edge_src_to_mid[(int(i), (L, int(H)))] = float((dz * gz).sum().detach().cpu().item())

    _, clean_cache = model.run_with_cache(
        clean_tokens,
        names_filter=z_hooks,
        return_type="logits",
    )

    hp_spec = head_patch_positions if head_patch_positions is not None else z_pos_eff

    edge_mid_to_dst: dict[tuple[tuple[int, int], int], float] = {}
    for L, H in mh_tuple:
        zh = tl_utils.get_act_name("z", L)
        if zh not in clean_cache:
            raise KeyError(f"Clean cache missing {zh!r}")
        clean_z = clean_cache[zh]
        target = PatchTarget("attn_head_z", int(L), head=int(H))
        pos_spec = _resolve_patch_pos(hp_spec, seq_len)
        hook_patch = _patch_fn(clean_z, target, pos_spec)
        act_dst_cap: torch.Tensor | None = None

        def _cap_dst(act: torch.Tensor, hook) -> torch.Tensor:
            nonlocal act_dst_cap
            act_dst_cap = act.detach()
            return act

        model.run_with_hooks(
            corrupt_tokens,
            fwd_hooks=[
                (zh, hook_patch),
                (str(dst_hook), _cap_dst),
            ],
            return_type="logits",
        )
        if act_dst_cap is None:
            raise RuntimeError(f"Failed to capture {dst_hook!r} after patching {zh!r}.")

        f_raw = encode_dst(act_dst_cap)
        if f_raw.dim() == 2:
            f_raw = f_raw.unsqueeze(0)
        dst_i = int(bipartite.dst_seq_pos_resolved)
        try:
            f_patch = f_raw[0, dst_i, :].detach().float()
        except IndexError as err:
            raise IndexError(
                f"encode_dst output shape {tuple(f_raw.shape)} cannot index position dst_seq_pos_resolved={dst_i} "
                f"(seq_len={seq_len})."
            ) from err
        delta = f_patch - base_f_dst.detach().float()
        for j in dst_feature_ids:
            jj = int(j)
            try:
                edge_mid_to_dst[((int(L), int(H)), jj)] = float((delta[jj] * g_cpu[jj]).detach().cpu().item())
            except IndexError as err:
                raise IndexError(
                    f"mid→dst edge failed for dst feature {jj} (delta dim={int(delta.numel())})."
                ) from err

    return ThreeNodeEdgeBuild(
        edge_src_to_mid=edge_src_to_mid,
        edge_mid_to_dst=edge_mid_to_dst,
        bipartite=bipartite,
        z_seq_pos_resolved=z_pos_eff,
        middle_heads=mh_tuple,
    )


def write_tripartite_sae_head_dot(
    *,
    out: PathLike | IO[str],
    src_feature_ids: Sequence[int],
    dst_feature_ids: Sequence[int],
    middle_heads: Sequence[tuple[int, int]],
    edge_src_to_mid: Mapping[tuple[int, tuple[int, int]], float],
    edge_mid_to_dst: Mapping[tuple[tuple[int, int], int], float],
    src_taylor: Mapping[int, float] | None = None,
    dst_taylor: Mapping[int, float] | None = None,
    src_cluster_label: str = "Source SAE latents",
    mid_cluster_label: str = "Attention heads (hook_z)",
    dst_cluster_label: str = "Destination SAE latents",
    min_abs_edge: float = 0.0,
    penwidth_scale: float = 8.0,
    title: str | None = None,
) -> None:
    """Tripartite ``digraph``: src latent → head → dst latent."""

    src_ids = [int(x) for x in src_feature_ids]
    dst_ids = [int(x) for x in dst_feature_ids]
    mids = [(int(L), int(H)) for L, H in middle_heads]
    src_scores = dict(src_taylor or {})
    dst_scores = dict(dst_taylor or {})

    max_node = 1e-12
    for i in src_ids:
        max_node = max(max_node, abs(float(src_scores.get(i, 0.0))))
    for j in dst_ids:
        max_node = max(max_node, abs(float(dst_scores.get(j, 0.0))))

    max_edge = 1e-12
    for w in edge_src_to_mid.values():
        max_edge = max(max_edge, abs(float(w)))
    for w in edge_mid_to_dst.values():
        max_edge = max(max_edge, abs(float(w)))

    lines = [
        "digraph sae_three_node {",
        "  rankdir=LR;",
        "  graph [fontsize=11, nodesep=0.45, ranksep=0.85];",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [arrowhead=vee, fontsize=9];',
    ]
    if title:
        lines.append(f'  label="{dot_escape_label(title)}";')
        lines.append("  labelloc=t;")

    lines.append(f'  subgraph cluster_src {{ label="{dot_escape_label(src_cluster_label)}"; style=dashed; color=gray;')
    for i in src_ids:
        sc = float(src_scores.get(i, 0.0))
        lab = f"{i}\nscore={sc:+.3g}" if i in src_scores else str(i)
        fill = score_to_fillcolor(sc, max_node) if i in src_scores else "#ecf0f1"
        lines.append(f'    src_{i} [label="{dot_escape_label(lab)}", fillcolor="{fill}"];')
    lines.append("  }")

    lines.append(f'  subgraph cluster_mid {{ label="{dot_escape_label(mid_cluster_label)}"; style=dashed; color=gray;')
    for L, H in mids:
        nid = f"mid_{L}_{H}"
        lab = f"L{L}H{H}"
        lines.append(f'    {nid} [label="{dot_escape_label(lab)}", fillcolor="#d6eaf8"];')
    lines.append("  }")

    lines.append(f'  subgraph cluster_dst {{ label="{dot_escape_label(dst_cluster_label)}"; style=dashed; color=gray;')
    for j in dst_ids:
        sc = float(dst_scores.get(j, 0.0))
        lab = f"{j}\nscore={sc:+.3g}" if j in dst_scores else str(j)
        fill = score_to_fillcolor(sc, max_node) if j in dst_scores else "#ecf0f1"
        lines.append(f'    dst_{j} [label="{dot_escape_label(lab)}", fillcolor="{fill}"];')
    lines.append("  }")

    for (i, mid_k), w in edge_src_to_mid.items():
        if int(i) not in src_ids or mid_k not in mids:
            continue
        L, H = mid_k
        aw = abs(float(w))
        if aw < float(min_abs_edge):
            continue
        col = "#27ae60" if w >= 0.0 else "#c0392b"
        pw = 0.35 + float(penwidth_scale) * (aw / max_edge if max_edge > 0 else 0.0)
        pw = min(float(penwidth_scale) + 2.0, max(0.35, pw))
        lines.append(
            f'  src_{int(i)} -> mid_{L}_{H} [label="{dot_escape_label(_edge_label(w))}", color="{col}", '
            f'fontcolor="{col}", penwidth={pw:.3f}];'
        )

    for (mid_k, j), w in edge_mid_to_dst.items():
        if mid_k not in mids or int(j) not in dst_ids:
            continue
        L, H = mid_k
        aw = abs(float(w))
        if aw < float(min_abs_edge):
            continue
        col = "#2980b9" if w >= 0.0 else "#8e44ad"
        pw = 0.35 + float(penwidth_scale) * (aw / max_edge if max_edge > 0 else 0.0)
        pw = min(float(penwidth_scale) + 2.0, max(0.35, pw))
        lines.append(
            f'  mid_{L}_{H} -> dst_{int(j)} [label="{dot_escape_label(_edge_label(w))}", color="{col}", '
            f'fontcolor="{col}", penwidth={pw:.3f}];'
        )

    lines.append("}")
    text = "\n".join(lines) + "\n"

    if isinstance(out, (str, Path)):
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    out.write(text)


def write_bipartite_sae_dot(
    *,
    out: PathLike | IO[str],
    src_feature_ids: Sequence[int],
    dst_feature_ids: Sequence[int],
    edge_weight: Mapping[tuple[int, int], float],
    src_taylor: Mapping[int, float] | None = None,
    dst_taylor: Mapping[int, float] | None = None,
    src_cluster_label: str = "Layer A (source latents)",
    dst_cluster_label: str = "Layer B (bottleneck latents)",
    min_abs_edge: float = 0.0,
    penwidth_scale: float = 8.0,
    title: str | None = None,
) -> None:
    """Write a ``digraph`` with ``rankdir=LR`` and promoter (green) / suppressor (red) edges."""

    src_ids = [int(x) for x in src_feature_ids]
    dst_ids = [int(x) for x in dst_feature_ids]
    src_scores = dict(src_taylor or {})
    dst_scores = dict(dst_taylor or {})

    max_node = 1e-12
    for i in src_ids:
        max_node = max(max_node, abs(float(src_scores.get(i, 0.0))))
    for j in dst_ids:
        max_node = max(max_node, abs(float(dst_scores.get(j, 0.0))))

    max_edge = 1e-12
    for (a, b), w in edge_weight.items():
        if a in src_ids and b in dst_ids:
            max_edge = max(max_edge, abs(float(w)))

    lines = [
        "digraph sae_circuit {",
        "  rankdir=LR;",
        "  graph [fontsize=11, nodesep=0.45, ranksep=0.85];",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [arrowhead=vee, fontsize=9];',
    ]
    if title:
        lines.append(f'  label="{dot_escape_label(title)}";')
        lines.append("  labelloc=t;")

    lines.append(f'  subgraph cluster_src {{ label="{dot_escape_label(src_cluster_label)}"; style=dashed; color=gray;')
    for i in src_ids:
        nid = f"src_{i}"
        sc = float(src_scores.get(i, 0.0))
        lab = f"{i}\nscore={sc:+.3g}" if i in src_scores else str(i)
        fill = score_to_fillcolor(sc, max_node) if i in src_scores else "#ecf0f1"
        lines.append(
            f'    {nid} [label="{dot_escape_label(lab)}", fillcolor="{fill}"];'
        )
    lines.append("  }")

    lines.append(f'  subgraph cluster_dst {{ label="{dot_escape_label(dst_cluster_label)}"; style=dashed; color=gray;')
    for j in dst_ids:
        nid = f"dst_{j}"
        sc = float(dst_scores.get(j, 0.0))
        lab = f"{j}\nscore={sc:+.3g}" if j in dst_scores else str(j)
        fill = score_to_fillcolor(sc, max_node) if j in dst_scores else "#ecf0f1"
        lines.append(
            f'    {nid} [label="{dot_escape_label(lab)}", fillcolor="{fill}"];'
        )
    lines.append("  }")

    for (i, j), w in edge_weight.items():
        if int(i) not in src_ids or int(j) not in dst_ids:
            continue
        aw = abs(float(w))
        if aw < float(min_abs_edge):
            continue
        col = "#27ae60" if w >= 0.0 else "#c0392b"
        pw = 0.35 + float(penwidth_scale) * (aw / max_edge if max_edge > 0 else 0.0)
        pw = min(float(penwidth_scale) + 2.0, max(0.35, pw))
        lines.append(
            f'  src_{int(i)} -> dst_{int(j)} [label="{dot_escape_label(_edge_label(w))}", color="{col}", fontcolor="{col}", '
            f'penwidth={pw:.3f}];'
        )

    lines.append("}")
    text = "\n".join(lines) + "\n"

    if isinstance(out, (str, Path)):
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    out.write(text)
