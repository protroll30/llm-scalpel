"""Manual latent intervention: layer-8 SAE feature → layer-9 SAE readout.

This script is meant for quick sanity checks like:
- pick top layer-8 features (e.g. 19692, 23151, 20901)
- force-activate one of them at a chosen token position
- measure the change in layer-9 SAE feature activations (e.g. 5396, 17889)

Example (CUDA):
  python scripts/intervene_layer8_to_layer9.py ^
    --sae-release gpt2-small-res-jb ^
    --src-sae-id blocks.8.hook_resid_pre ^
    --dst-sae-id blocks.9.hook_resid_pre ^
    --prompt "The capital of Germany is" ^
    --seq-pos 4 ^
    --src-features 19692 23151 20901 ^
    --dst-features 5396 17889 ^
    --mode set ^
    --value 5.0 ^
    --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

import torch
from transformer_lens import HookedTransformer

from discovery.sae_lens_bridge import assert_d_in_matches_model, discovery_encode_decode, load_pretrained_sae
from discovery.sae_scout import reconstruct_activation


def _resolve_pos(seq_pos: int, seq_len: int) -> int:
    pos = int(seq_pos)
    if pos < 0:
        pos += int(seq_len)
    if pos < 0 or pos >= int(seq_len):
        raise IndexError(f"seq_pos resolved to {pos}, invalid for seq_len={seq_len}")
    return pos


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


def _parse_int_list(xs: Sequence[str]) -> list[int]:
    out: list[int] = []
    for x in xs:
        out.append(int(x))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Force-activate layer-8 SAE latents and read layer-9 SAE deltas.")
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument("--prompt", type=str, default="The capital of Germany is")
    p.add_argument("--seq-pos", type=int, default=4)
    p.add_argument(
        "--prepend-bos",
        action="store_true",
        help="If set, prepend the BOS token when tokenizing (shifts token indices by +1 for most prompts).",
    )

    p.add_argument("--sae-release", type=str, required=True)
    p.add_argument("--src-sae-id", type=str, default="blocks.8.hook_resid_pre")
    p.add_argument("--dst-sae-id", type=str, default="blocks.9.hook_resid_pre")
    p.add_argument("--sae-dtype", type=str, default="float32")
    p.add_argument("--sae-force-download", action="store_true")

    p.add_argument("--src-features", nargs="+", required=True, help="Layer-8 feature ids to intervene on.")
    p.add_argument("--dst-features", nargs="+", required=True, help="Layer-9 feature ids to read out.")
    p.add_argument("--mode", type=str, default="set", choices=("set", "add"))
    p.add_argument("--value", type=float, default=5.0, help="Value to set/add for each src feature at seq_pos.")
    p.add_argument(
        "--counterfactual-scale",
        type=float,
        default=0.0,
        help=(
            "If >0: compute each src feature's natural activation A_base at seq_pos and set its value to "
            "A_base * counterfactual_scale (mode='set') or add A_base * (counterfactual_scale - 1) (mode='add'). "
            "This keeps interventions in-distribution vs a fixed constant."
        ),
    )

    args = p.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    # Match SAELens GPT-2 SAE training coords (legacy processing).
    model = HookedTransformer.from_pretrained(
        str(args.model),
        device=device,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
    )
    model.eval()

    src_features = _parse_int_list(args.src_features)
    dst_features = _parse_int_list(args.dst_features)

    # Load SAEs
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

    # Tokens
    tokens = model.to_tokens(str(args.prompt), prepend_bos=bool(args.prepend_bos)).to(device)
    seq_len = int(tokens.shape[-1])
    pos_eff = _resolve_pos(int(args.seq_pos), seq_len)
    try:
        tok_list = model.to_str_tokens(tokens[0] if tokens.dim() == 2 else tokens)
        if 0 <= pos_eff < len(tok_list):
            print(f"Token at pos {pos_eff}: {tok_list[pos_eff]!r}")
        else:
            print(f"Token at pos {pos_eff}: <out of range for to_str_tokens len={len(tok_list)}>")
    except Exception as e:  # pragma: no cover
        print(f"Token at pos {pos_eff}: <failed to decode tokens: {e}>")

    # Baseline: layer-9 features at pos.
    base_f9 = _encode_feature_vec_at_pos(model=model, tokens=tokens, hook_name=str(args.dst_sae_id), encode_fn=encode9, seq_pos=pos_eff)

    # Baseline: layer-8 features at pos (for counterfactual scaling).
    base_f8 = _encode_feature_vec_at_pos(model=model, tokens=tokens, hook_name=str(args.src_sae_id), encode_fn=encode8, seq_pos=pos_eff)
    print(f"Max f8 value at pos {pos_eff}: {float(base_f8.max().detach().cpu().item()):.6g}")
    if src_features:
        src_vals = [float(base_f8[j].detach().cpu().item()) for j in src_features]
        print(f"Baseline f8 src feature values at pos {pos_eff}: {list(zip(src_features, src_vals))}")

    # Intervention: hook at layer 8, modify f at pos, reconstruct activation with corrupt residual.
    forced = torch.tensor(src_features, device=device, dtype=torch.long)
    if forced.numel() == 0:
        raise SystemExit("--src-features must be non-empty.")
    base_forced = base_f8.index_select(0, forced).detach()
    cf_scale = float(args.counterfactual_scale)

    def _hook8(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        f = encode8(act)
        if f.dim() == 2:
            f = f.unsqueeze(0)
        if f.dim() < 3:
            raise ValueError(f"encode_fn must return [pos, n_features] or [batch, pos, n_features]; got {tuple(f.shape)}")
        f_full = f
        f_work = f_full.clone()

        # Apply intervention at [batch=0, pos_eff, feature]
        if cf_scale > 0.0:
            if args.mode == "set":
                target = base_forced.to(device=f_work.device, dtype=f_work.dtype) * cf_scale
                f_work[0, pos_eff, forced] = target
            else:  # add
                delta = base_forced.to(device=f_work.device, dtype=f_work.dtype) * (cf_scale - 1.0)
                f_work[0, pos_eff, forced] = f_work[0, pos_eff, forced] + delta
        else:
            if args.mode == "set":
                f_work[0, pos_eff, :].index_fill_(0, forced, float(args.value))
            else:  # add
                f_work[0, pos_eff, forced] = f_work[0, pos_eff, forced] + float(args.value)

        return reconstruct_activation(
            f_patched=f_work,
            x_corrupt=act,
            f_corrupt=f_full,
            decode_fn=decode8,
        )

    with torch.no_grad():
        _ = model.run_with_hooks(tokens, fwd_hooks=[(str(args.src_sae_id), _hook8)], return_type="logits")

    # Read layer-9 features under intervention.
    int_f9 = _encode_feature_vec_at_pos(model=model, tokens=tokens, hook_name=str(args.dst_sae_id), encode_fn=encode9, seq_pos=pos_eff)
    delta = (int_f9 - base_f9).detach().cpu()

    print(f"prompt={args.prompt!r} seq_pos={pos_eff} device={device}")
    if cf_scale > 0.0:
        print(
            f"src={args.src_sae_id} mode={args.mode} counterfactual_scale={cf_scale} "
            f"src_features={src_features}"
        )
    else:
        print(f"src={args.src_sae_id} mode={args.mode} value={args.value} src_features={src_features}")
    print(f"dst={args.dst_sae_id} dst_features={dst_features}")
    print()

    for j in dst_features:
        b = float(base_f9[j].detach().cpu().item())
        a = float(int_f9[j].detach().cpu().item())
        d = float(delta[j].item())
        print(f"dst_feature[{j}]: baseline={b:+.6g}  intervened={a:+.6g}  delta={d:+.6g}")

    # Also show the largest-magnitude deltas (quick scan)
    k = min(10, int(delta.numel()))
    top = torch.topk(delta.abs(), k=k)
    top_ids = [int(i) for i in top.indices.tolist()]
    print()
    print(f"top|Δ dst_feature| (k={k}): {top_ids}")


if __name__ == "__main__":
    # Ensure repo root on path when running from elsewhere.
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    main()

