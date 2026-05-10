"""Manual latent intervention: layer-8 SAE feature → destination-layer SAE readout.

Default destination hook is **blocks.10.hook_resid_pre** so the mover head (e.g. L9H8) can run,
write into the residual stream, and layer-10 latents capture that signal (e.g. ``Berlin``-like
bottleneck features). Override ``--dst-sae-id`` if you still want a layer-9 readout.

Dual intervention (zero some latents, then inject others at the same seq position in one forward):
  Same flags as below, plus e.g. ``--erase-features ID1 ID2 --inject-features ID3`` (instead of ``--src-features``).

Single prompt (CUDA)::

  python scripts/intervene_layer8_to_layer9.py ^
    --sae-release gpt2-small-res-jb ^
    --src-sae-id blocks.8.hook_resid_pre ^
    --dst-sae-id blocks.10.hook_resid_pre ^
    --prompt \"The capital of Germany is\" ^
    --seq-pos 4 ^
    --src-features 19692 23151 20901 ^
    --dst-features 5396 17889 ^
    --mode set ^
    --value 5.0 ^
    --device cuda

Batch over a filtered benchmark JSON (Phase 3-style loop)::

  python scripts/intervene_layer8_to_layer9.py ^
    --benchmark-json benchmarks/processed/factual_recall_filtered_enriched.json ^
    --benchmark-batch ^
    --benchmark-prompt-field corrupt ^
    --seq-pos-fallback last ^
    --dynamic-src-top-k 8 ^
    ... dst-features ...

**Positions:** Prefer per-row ``metric_seq_pos`` / ``loss_seq_pos`` / ``seq_pos`` / ``subject_pos`` in the JSON.
If absent, ``--seq-pos-fallback last`` uses the last prompt token (``-1``), avoiding a fixed ``--seq-pos 4`` across
varying template lengths. **Layer-8 ids:** Optional per-row ``inject_features`` / ``src_feature_ids``; else
``--dynamic-src-top-k`` picks top-|activation| latents at the chosen position for prompt-specific features.

**Destination ids:** ``--dynamic-dst-latent`` reads per-row ``correct_answer_id`` (int or list) from the benchmark JSON;
if a row omits it, ``--dst-features`` is used as fallback when provided.

**Why layer-10 ``dst-sae-id``:** Layer 9 ``hook_resid_pre`` is *before* the L9 attention block finishes writing;
layer 10 captures post–mover-head residual state relevant to late bottlenecks (e.g. answer-side latents).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from transformer_lens import HookedTransformer

from discovery.benchmark_json import (
    add_discovery_benchmark_cli_args,
    apply_benchmark_single_prompt,
    dst_features_from_correct_answer_id,
    inject_features_from_benchmark_row,
    load_benchmark_pairs,
    prompt_field_from_row,
    select_benchmark_row,
    seq_pos_from_benchmark_row,
)
from discovery.sae_lens_bridge import assert_d_in_matches_model, discovery_encode_decode, load_pretrained_sae
from discovery.sae_scout import reconstruct_activation


def _resolve_pos(seq_pos: int, seq_len: int) -> int:
    pos = int(seq_pos)
    if pos < 0:
        pos += int(seq_len)
    if pos < 0 or pos >= int(seq_len):
        raise IndexError(f"seq_pos resolved to {pos}, invalid for seq_len={seq_len}")
    return pos


def _resolve_dst_features_for_row(
    args: Any,
    row: dict[str, Any] | None,
    dst_cli: list[int],
) -> list[int] | None:
    """Static ``dst_cli``, or per-row ``correct_answer_id`` when ``--dynamic-dst-latent`` (fallback: ``dst_cli``)."""
    if not bool(getattr(args, "dynamic_dst_latent", False)):
        return dst_cli if dst_cli else None
    if row is not None:
        got = dst_features_from_correct_answer_id(row)
        if got:
            return got
    return dst_cli if dst_cli else None


def _resolve_seq_pos_from_args(args: Any, row: dict[str, Any] | None) -> int:
    """Prefer per-row JSON keys; else ``--seq-pos-fallback`` (cli vs last=-1) then ``--seq-pos``."""
    if row is not None:
        sp = seq_pos_from_benchmark_row(row)
        if sp is not None:
            return sp
    fb = str(getattr(args, "seq_pos_fallback", "cli") or "cli").strip().lower()
    if fb == "last":
        return -1
    return int(args.seq_pos)


@torch.no_grad()
def _encode_feature_vec_at_pos(
    *,
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    seq_pos: int,
) -> torch.Tensor:
    """Return 1D feature vector at one position: shape [n_features]."""
    model.eval()
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name], return_type="logits")
    if hook_name not in cache:
        raise KeyError(f"Cache missing {hook_name!r}")
    act = cache[hook_name]
    f = encode_fn(act)
    if f.dim() == 2:
        f = f.unsqueeze(0)
    if f.dim() < 3:
        raise ValueError(f"encode_fn must return [pos, n_features] or [batch, pos, n_features]; got {tuple(f.shape)}")
    pos_eff = _resolve_pos(seq_pos, int(f.shape[1]))
    return f[0, pos_eff, :].detach().clone()


def _parse_int_list(xs: list[str]) -> list[int]:
    return [int(x) for x in xs]


def run_one_intervention(
    *,
    args: Any,
    model: HookedTransformer,
    encode8: Callable[[torch.Tensor], torch.Tensor],
    encode9: Callable[[torch.Tensor], torch.Tensor],
    decode8: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    prompt: str,
    pair_tag: str,
    seq_pos_raw: int,
    erase_features: list[int],
    inject_features: list[int],
    dst_features: list[int],
    dual_mode: bool,
    dynamic_src_top_k: int = 0,
) -> None:
    """Single forward with layer-8 hook + capture at ``args.dst_sae_id``."""
    tokens = model.to_tokens(prompt, prepend_bos=bool(args.prepend_bos)).to(device)
    seq_len = int(tokens.shape[-1])
    pos_eff = _resolve_pos(int(seq_pos_raw), seq_len)
    try:
        tok_list = model.to_str_tokens(tokens[0] if tokens.dim() == 2 else tokens)
        if 0 <= pos_eff < len(tok_list):
            print(f"[{pair_tag}] Token at pos {pos_eff}: {tok_list[pos_eff]!r}")
        else:
            print(f"[{pair_tag}] Token at pos {pos_eff}: <out of range for to_str_tokens len={len(tok_list)}>")
    except Exception as e:  # pragma: no cover
        print(f"[{pair_tag}] Token at pos {pos_eff}: <failed to decode tokens: {e}>")

    base_f9 = _encode_feature_vec_at_pos(
        model=model,
        tokens=tokens,
        hook_name=str(args.dst_sae_id),
        encode_fn=encode9,
        seq_pos=pos_eff,
    )

    base_f8 = _encode_feature_vec_at_pos(
        model=model,
        tokens=tokens,
        hook_name=str(args.src_sae_id),
        encode_fn=encode8,
        seq_pos=pos_eff,
    )
    inject_eff = list(inject_features)
    if dynamic_src_top_k > 0:
        if dual_mode:
            raise RuntimeError(f"[{pair_tag}] --dynamic-src-top-k is incompatible with --erase-features/--inject-features dual mode.")
        k = min(int(dynamic_src_top_k), int(base_f8.shape[0]))
        _, top_ix = torch.topk(base_f8.abs(), k=k)
        inject_eff = [int(i) for i in top_ix.cpu().tolist()]
        print(f"[{pair_tag}] dynamic-src-top-k={dynamic_src_top_k} -> L8 inject ids {inject_eff}")

    print(f"[{pair_tag}] Max f8 value at pos {pos_eff}: {float(base_f8.max().detach().cpu().item()):.6g}")
    if erase_features:
        ev = [float(base_f8[j].detach().cpu().item()) for j in erase_features]
        print(f"[{pair_tag}] Baseline f8 erase feature values at pos {pos_eff}: {list(zip(erase_features, ev))}")
    if inject_eff:
        iv = [float(base_f8[j].detach().cpu().item()) for j in inject_eff]
        print(f"[{pair_tag}] Baseline f8 inject feature values at pos {pos_eff}: {list(zip(inject_eff, iv))}")

    erase_idx = torch.tensor(erase_features, device=device, dtype=torch.long)
    inject_idx = torch.tensor(inject_eff, device=device, dtype=torch.long)
    base_inject = base_f8.index_select(0, inject_idx).detach() if inject_idx.numel() else torch.empty(0, device=device)
    cf_scale = float(args.counterfactual_scale)
    hook_debug_printed: list[bool] = [False]

    def _hook8(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        if not hook_debug_printed[0]:
            hook_name = getattr(hook, "name", "<unknown>")
            print(f"[{pair_tag}] DEBUG: Successfully hooked into {hook_name}")
            hook_debug_printed[0] = True
        if bool(args.debug_zero_act):
            return torch.zeros_like(act)

        f = encode8(act)
        if f.dim() == 2:
            f = f.unsqueeze(0)
        if f.dim() < 3:
            raise ValueError(f"encode_fn must return [pos, n_features] or [batch, pos, n_features]; got {tuple(f.shape)}")
        f_full = f
        f_work = f_full.clone()

        if erase_idx.numel() > 0:
            f_work[0, pos_eff, erase_idx] = 0.0
        if inject_idx.numel() > 0:
            if cf_scale > 0.0:
                if args.mode == "set":
                    target = base_inject.to(device=f_work.device, dtype=f_work.dtype) * cf_scale
                    f_work[0, pos_eff, inject_idx] = target
                else:
                    delta_inj = base_inject.to(device=f_work.device, dtype=f_work.dtype) * (cf_scale - 1.0)
                    f_work[0, pos_eff, inject_idx] = f_work[0, pos_eff, inject_idx] + delta_inj
            else:
                if args.mode == "set":
                    f_work[0, pos_eff, inject_idx] = float(args.value)
                else:
                    f_work[0, pos_eff, inject_idx] = f_work[0, pos_eff, inject_idx] + float(args.value)

        return reconstruct_activation(
            f_patched=f_work,
            x_corrupt=act,
            f_corrupt=f_full,
            decode_fn=decode8,
        )

    act9_int: torch.Tensor | None = None

    def _capture9(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        nonlocal act9_int
        act9_int = act.detach()
        return act

    with torch.no_grad():
        logits_int = model.run_with_hooks(
            tokens,
            fwd_hooks=[
                (str(args.src_sae_id), _hook8),
                (str(args.dst_sae_id), _capture9),
            ],
            return_type="logits",
        )

    if act9_int is None:
        raise RuntimeError(f"[{pair_tag}] Failed to capture activation at {str(args.dst_sae_id)!r}. Check hook name.")
    f9_int_raw = encode9(act9_int)
    if f9_int_raw.dim() == 2:
        f9_int_raw = f9_int_raw.unsqueeze(0)
    if f9_int_raw.dim() < 3:
        raise ValueError(
            f"encode_fn must return [pos, n_features] or [batch, pos, n_features]; got {tuple(f9_int_raw.shape)}"
        )
    int_f9 = f9_int_raw[0, pos_eff, :].detach().clone()
    delta = (int_f9 - base_f9).detach().cpu()

    print(f"[{pair_tag}] prompt={prompt!r} seq_pos_raw={seq_pos_raw} resolved={pos_eff} device={device}")
    if dual_mode:
        print(
            f"[{pair_tag}] src={args.src_sae_id} erase={erase_features} inject={inject_eff} "
            f"mode={args.mode} value={args.value} counterfactual_scale={cf_scale}"
        )
    elif cf_scale > 0.0:
        print(f"[{pair_tag}] src={args.src_sae_id} mode={args.mode} counterfactual_scale={cf_scale} src_features={inject_eff}")
    else:
        print(f"[{pair_tag}] src={args.src_sae_id} mode={args.mode} value={args.value} src_features={inject_eff}")
    print(f"[{pair_tag}] dst={args.dst_sae_id} dst_features={dst_features}")
    print()

    for j in dst_features:
        b = float(base_f9[j].detach().cpu().item())
        a = float(int_f9[j].detach().cpu().item())
        d = float(delta[j].item())
        print(f"[{pair_tag}] dst_feature[{j}]: baseline={b:+.6g}  intervened={a:+.6g}  delta={d:+.6g}")

    k = min(10, int(delta.numel()))
    top = torch.topk(delta.abs(), k=k)
    top_ids = [int(i) for i in top.indices.tolist()]
    print()
    print(f"[{pair_tag}] top|Δ dst_feature| (k={k}): {top_ids}")

    try:
        vocab_logits = logits_int[0, -1, :].detach()
        topk = torch.topk(vocab_logits, k=min(10, int(vocab_logits.numel())))
        print()
        print(f"[{pair_tag}] Top predictions (intervened) at last seq index ( logits[0,-1,...] ):")
        for rank, (tok_id, logit) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
            tok_str = model.tokenizer.decode([int(tok_id)]) if getattr(model, "tokenizer", None) is not None else str(tok_id)
            print(f"[{pair_tag}]   {rank:>2}. id={int(tok_id):>5} logit={float(logit):+.4f} token={tok_str!r}")
    except Exception as e:  # pragma: no cover
        print(f"[{pair_tag}] (topk decode failed: {e})")
    print()
    print("=" * 80)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Force-activate layer-8 SAE latents and read destination-layer (default L10) SAE deltas."
    )
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument("--prompt", type=str, default="The capital of Germany is")
    add_discovery_benchmark_cli_args(p)
    p.add_argument(
        "--benchmark-prompt-field",
        type=str,
        default="corrupt",
        choices=("clean", "corrupt"),
        help="When using --benchmark-json: which row field supplies the prompt for each pair.",
    )
    p.add_argument(
        "--benchmark-batch",
        action="store_true",
        help="With --benchmark-json: run intervention for every pair in 'pairs' (Phase 3-style batch).",
    )
    p.add_argument(
        "--benchmark-max-pairs",
        type=int,
        default=0,
        help="With --benchmark-batch: stop after this many pairs (0 = no limit).",
    )
    p.add_argument("--seq-pos", type=int, default=4)
    p.add_argument(
        "--seq-pos-fallback",
        type=str,
        default="cli",
        choices=("cli", "last"),
        help=(
            "When no per-row metric_seq_pos / loss_seq_pos / seq_pos / subject_pos in benchmark JSON: "
            "'cli' uses --seq-pos; 'last' uses the last prompt token index (-1)."
        ),
    )
    p.add_argument(
        "--prepend-bos",
        action="store_true",
        help="If set, prepend the BOS token when tokenizing (shifts token indices by +1 for most prompts).",
    )

    p.add_argument("--sae-release", type=str, required=True)
    p.add_argument("--src-sae-id", type=str, default="blocks.8.hook_resid_pre")
    p.add_argument(
        "--dst-sae-id",
        type=str,
        default="blocks.10.hook_resid_pre",
        help="Capture latents here after full forward (default: layer 10 resid_pre for post-mover-head signal).",
    )
    p.add_argument("--sae-dtype", type=str, default="float32")
    p.add_argument("--sae-force-download", action="store_true")

    p.add_argument(
        "--src-features",
        nargs="*",
        default=[],
        metavar="ID",
        help="Legacy: layer-8 latent ids to intervene on (same mode/value/scale for all). Do not combine with --erase-features/--inject-features.",
    )
    p.add_argument(
        "--erase-features",
        nargs="*",
        default=[],
        metavar="ID",
        help="Dual mode: set these layer-8 latents to 0.0 at seq_pos before inject.",
    )
    p.add_argument(
        "--inject-features",
        nargs="*",
        default=[],
        metavar="ID",
        help="Dual mode: apply --mode/--value/--counterfactual-scale to these latents at seq_pos (after erase).",
    )
    p.add_argument(
        "--dynamic-src-top-k",
        type=int,
        default=0,
        metavar="K",
        help=(
            "Non-dual mode: after measuring base_f8 at seq_pos, replace inject ids with top-K by |activation|. "
            "Use when batch subjects vary and fixed --src-features are wrong. Incompatible with dual erase/inject."
        ),
    )
    p.add_argument(
        "--dynamic-dst-latent",
        action="store_true",
        help=(
            "With --benchmark-json: read destination SAE feature id(s) from each row's correct_answer_id "
            "(int or list). Requires benchmark JSON; rows without the field fall back to --dst-features if set."
        ),
    )
    p.add_argument(
        "--dst-features",
        nargs="*",
        default=[],
        metavar="ID",
        help=(
            "Destination SAE feature ids to read out (e.g. bottleneck latents at layer 10). "
            "Omit when using only per-row correct_answer_id (--dynamic-dst-latent) if every row has that field."
        ),
    )
    p.add_argument("--mode", type=str, default="set", choices=("set", "add"))
    p.add_argument("--value", type=float, default=5.0, help="Inject: value to set/add for each inject latent at seq_pos.")
    p.add_argument(
        "--counterfactual-scale",
        type=float,
        default=0.0,
        help=(
            "If >0: for each inject latent, use natural A_base at seq_pos; set to A_base * scale (mode=set) "
            "or add A_base * (scale - 1) (mode=add). Ignored for --erase-features (always zero)."
        ),
    )
    p.add_argument(
        "--debug-zero-act",
        action="store_true",
        help=(
            "Debug plumbing: if set, the layer-8 hook prints a message and returns zeros_like(act) "
            "(completely destructive intervention). This should cause large downstream changes if the hook fires."
        ),
    )

    args = p.parse_args()

    dst_cli = _parse_int_list(args.dst_features)
    if bool(args.dynamic_dst_latent) and not str(args.benchmark_json or "").strip():
        raise SystemExit("--dynamic-dst-latent requires --benchmark-json.")
    if not bool(args.dynamic_dst_latent) and not dst_cli:
        raise SystemExit("Provide --dst-features (one or more ids), or use --dynamic-dst-latent with --benchmark-json.")

    if args.benchmark_batch and not str(args.benchmark_json or "").strip():
        raise SystemExit("--benchmark-batch requires --benchmark-json.")

    if not args.benchmark_batch:
        apply_benchmark_single_prompt(args, prompt_attr="prompt", field=str(args.benchmark_prompt_field))

    erase_features = _parse_int_list(args.erase_features)
    inject_features = _parse_int_list(args.inject_features)
    src_features_legacy = _parse_int_list(args.src_features)
    dynamic_src_top_k = int(args.dynamic_src_top_k)
    dual_mode = bool(args.erase_features) or bool(args.inject_features)
    if dual_mode:
        if src_features_legacy:
            raise SystemExit("Use either --src-features (legacy) OR --erase-features/--inject-features (dual), not both.")
        if dynamic_src_top_k > 0:
            raise SystemExit("--dynamic-src-top-k cannot be used with --erase-features/--inject-features (dual mode).")
    else:
        inject_features = src_features_legacy
        erase_features = []

    row_single: dict[str, Any] | None = None
    if str(args.benchmark_json or "").strip() and not args.benchmark_batch:
        pairs_one = load_benchmark_pairs(str(args.benchmark_json))
        row_single = select_benchmark_row(
            pairs_one,
            pair_index=int(args.benchmark_index),
            pair_id=getattr(args, "benchmark_id", None),
        )

    if not dual_mode:
        row_inject = inject_features_from_benchmark_row(row_single) if row_single is not None else None
        has_inject_source = bool(inject_features) or bool(row_inject) or dynamic_src_top_k > 0
        if not args.benchmark_batch and not has_inject_source:
            raise SystemExit(
                "Provide --src-features, --dynamic-src-top-k, or per-row inject_features / src_feature_ids in JSON; "
                "or use --erase-features/--inject-features for dual mode."
            )
    overlap = sorted(set(inject_features) & set(erase_features))
    if overlap:
        print(f"warning: features in both erase and inject (erase applied first, then inject): {overlap}", file=sys.stderr)

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    model = HookedTransformer.from_pretrained(
        str(args.model),
        device=device,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
    )
    model.eval()

    src_sae = load_pretrained_sae(
        release=str(args.sae_release),
        sae_id=str(args.src_sae_id),
        device=device,
        dtype=str(args.sae_dtype),
        force_download=bool(args.sae_force_download),
    )
    dst_sae = load_pretrained_sae(
        release=str(args.sae_release),
        sae_id=str(args.dst_sae_id),
        device=device,
        dtype=str(args.sae_dtype),
        force_download=bool(args.sae_force_download),
    )
    assert_d_in_matches_model(src_sae, d_model=int(model.cfg.d_model))
    assert_d_in_matches_model(dst_sae, d_model=int(model.cfg.d_model))
    encode8, decode8 = discovery_encode_decode(src_sae)
    encode9, _decode9 = discovery_encode_decode(dst_sae)

    field = str(args.benchmark_prompt_field)

    if args.benchmark_batch:
        pairs_all = load_benchmark_pairs(str(args.benchmark_json))
        limit_n = int(args.benchmark_max_pairs)
        pairs_loop = pairs_all[:limit_n] if limit_n > 0 else pairs_all
        n_ok = 0
        for idx, row in enumerate(pairs_loop):
            try:
                prompt = prompt_field_from_row(row, field)
            except Exception as e:
                print(f"[skip idx={idx}] {e}", file=sys.stderr)
                continue
            seq_raw = _resolve_seq_pos_from_args(args, row)
            inject_row = inject_features_from_benchmark_row(row)
            inject_use = list(inject_features)
            dyn_k = dynamic_src_top_k
            if inject_row is not None:
                inject_use = inject_row
                dyn_k = 0
            elif dynamic_src_top_k > 0:
                inject_use = []
                dyn_k = dynamic_src_top_k

            if not dual_mode and not inject_use and dyn_k == 0:
                print(
                    f"[skip idx={idx}] no layer-8 inject ids (empty --src-features, no row inject_features, "
                    f"--dynamic-src-top-k 0)",
                    file=sys.stderr,
                )
                continue

            dst_use = _resolve_dst_features_for_row(args, row, dst_cli)
            if not dst_use:
                print(
                    f"[skip idx={idx}] no dst features (--dst-features empty and no correct_answer_id on row)",
                    file=sys.stderr,
                )
                continue

            pid = row.get("id", idx)
            pair_tag = f"id={pid} idx={idx}"
            run_one_intervention(
                args=args,
                model=model,
                encode8=encode8,
                encode9=encode9,
                decode8=decode8,
                device=device,
                prompt=prompt,
                pair_tag=pair_tag,
                seq_pos_raw=seq_raw,
                erase_features=erase_features,
                inject_features=inject_use,
                dst_features=dst_use,
                dual_mode=dual_mode,
                dynamic_src_top_k=dyn_k,
            )
            n_ok += 1
        slice_note = f"first {limit_n} rows" if limit_n > 0 else "all rows"
        print(f"batch done: {n_ok} successful run(s) ({slice_note} of {len(pairs_all)}) source={args.benchmark_json}")
        return

    if not str(args.prompt).strip():
        raise SystemExit("Provide --prompt or --benchmark-json (without --benchmark-batch for a single row).")

    seq_raw = _resolve_seq_pos_from_args(args, row_single)
    inject_use = list(inject_features)
    dyn_k = dynamic_src_top_k
    if row_single is not None:
        ir = inject_features_from_benchmark_row(row_single)
        if ir is not None:
            inject_use = ir
            dyn_k = 0

    dst_use = _resolve_dst_features_for_row(args, row_single, dst_cli)
    if not dst_use:
        raise SystemExit(
            "No destination features: add correct_answer_id to the benchmark row, or pass --dst-features "
            "(fallback when using --dynamic-dst-latent)."
        )

    run_one_intervention(
        args=args,
        model=model,
        encode8=encode8,
        encode9=encode9,
        decode8=decode8,
        device=device,
        prompt=str(args.prompt),
        pair_tag="single",
        seq_pos_raw=seq_raw,
        erase_features=erase_features,
        inject_features=inject_use,
        dst_features=dst_use,
        dual_mode=dual_mode,
        dynamic_src_top_k=dyn_k,
    )


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    main()
