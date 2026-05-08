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

from discovery.attribution import feature_attribution_pass

PathLike = Union[str, Path]


def dot_escape_label(text: str) -> str:
    """Escape text for use inside DOT double-quoted labels."""

    return text.replace("\\", "\\\\").replace('"', '\\"')


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


def _resolve_pos(seq_pos: int, seq_len: int) -> int:
    pos = int(seq_pos)
    if pos < 0:
        pos += int(seq_len)
    if pos < 0 or pos >= int(seq_len):
        raise IndexError(f"seq_pos resolved to {pos}, invalid for seq_len={seq_len}")
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
        "  graph [fontsize=11];",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [arrowhead=vee];',
    ]
    if title:
        lines.append(f'  label="{dot_escape_label(title)}";')
        lines.append("  labelloc=t;")

    lines.append(f'  subgraph cluster_src {{ label="{dot_escape_label(src_cluster_label)}"; style=dashed; color=gray;')
    for i in src_ids:
        nid = f"src_{i}"
        sc = float(src_scores.get(i, 0.0))
        lab = f"{i}\\nf·∇f={sc:.4g}" if i in src_scores else str(i)
        fill = score_to_fillcolor(sc, max_node) if i in src_scores else "#ecf0f1"
        lines.append(
            f'    {nid} [label="{dot_escape_label(lab)}", fillcolor="{fill}"];'
        )
    lines.append("  }")

    lines.append(f'  subgraph cluster_dst {{ label="{dot_escape_label(dst_cluster_label)}"; style=dashed; color=gray;')
    for j in dst_ids:
        nid = f"dst_{j}"
        sc = float(dst_scores.get(j, 0.0))
        lab = f"{j}\\nf·∇f={sc:.4g}" if j in dst_scores else str(j)
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
        lab = f"{w:.3g}"
        lines.append(
            f'  src_{int(i)} -> dst_{int(j)} [label="{dot_escape_label(lab)}", color="{col}", fontcolor="{col}", '
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
