"""Emit a Graphviz DOT bipartite graph: layer-A SAE latents → layer-B SAE latents.

Edge width encodes |Δf_j · ∂ℒ/∂f_j| after a **single-source** intervention on layer A;
edge color is green (positive) vs red (negative). Node fill uses per-latent Taylor scores
``(Δf)·(∂ℒ/∂f)`` at each hook for the same clean/corrupt logit-diff objective.

Dependencies: Graphviz ``dot`` to render PDF/SVG, e.g.::

  dot -Tpdf circuit.dot -o circuit.pdf

Example::

  python scripts/sae_crosslayer_circuit_dot.py ^
    --device cuda ^
    --sae-release gpt2-small-res-jb ^
    --src-sae-id blocks.8.hook_resid_pre ^
    --dst-sae-id blocks.9.hook_resid_pre ^
    --src-features 23151 19692 20901 ^
    --dst-features 5396 17889 ^
    --out runs/circuit_l8_l9.dot

Bridge different tokens (e.g. intervene at country, read bottleneck at ``is``, logit-diff at last)::

  python scripts/sae_crosslayer_circuit_dot.py ... ^
    --seq-pos -1 ^
    --loss-seq-pos -1 ^
    --src-seq-pos 4 ^
    --dst-seq-pos 5

Three-node graph (latent → hook_z head → latent); writes a **separate** DOT file so the direct
bipartite `--out` graph is unchanged::

  python scripts/sae_crosslayer_circuit_dot.py ... --out runs/circuit_l8_l9.dot \\
    --three-node --middle-head 9 8 --middle-head 8 11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from discovery.circuit_graphviz import (
    build_cross_layer_edges,
    build_three_node_edges,
    write_bipartite_sae_dot,
    write_tripartite_sae_head_dot,
)
from discovery.sae_lens_bridge import assert_d_in_matches_model, discovery_encode_decode, load_pretrained_sae


def main() -> None:
    p = argparse.ArgumentParser(
        description="Write a bipartite SAE circuit graph (DOT) for two TransformerLens hooks."
    )
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument(
        "--tl-no-processing",
        action="store_true",
        help="Load HookedTransformer without fold_ln / center_* (usually leave off for SAELens GPT-2 SAEs).",
    )
    p.add_argument("--sae-release", type=str, required=True)
    p.add_argument("--src-sae-id", type=str, default="blocks.8.hook_resid_pre")
    p.add_argument("--dst-sae-id", type=str, default="blocks.9.hook_resid_pre")
    p.add_argument("--sae-dtype", type=str, default="float32")
    p.add_argument("--sae-force-download", action="store_true")

    p.add_argument("--clean-prompt", type=str, default="The capital of France is")
    p.add_argument("--corrupt-prompt", type=str, default="The capital of Germany is")
    p.add_argument("--clean-answer", type=str, default=" Paris", help="Single tokenizer token.")
    p.add_argument("--corrupt-answer", type=str, default=" Berlin", help="Single tokenizer token.")
    p.add_argument(
        "--seq-pos",
        type=int,
        default=-1,
        help=(
            "Default token index for loss (unless --loss-seq-pos) and for src/dst hooks "
            "(unless --src-seq-pos / --dst-seq-pos). Supports negative indices."
        ),
    )
    p.add_argument(
        "--loss-seq-pos",
        type=int,
        default=None,
        help=(
            "Token index for logit-diff loss logits[0, pos, ...]. "
            "Defaults to --seq-pos (typically -1 for next-token prediction)."
        ),
    )
    p.add_argument(
        "--src-seq-pos",
        type=int,
        default=None,
        help="Layer-8 (source) attribution + intervention position; defaults to --seq-pos.",
    )
    p.add_argument(
        "--dst-seq-pos",
        type=int,
        default=None,
        help="Layer-9 (destination) attribution + Δf readout position; defaults to --seq-pos.",
    )

    p.add_argument("--prepend-bos", action="store_true")

    p.add_argument("--src-features", nargs="+", type=str, required=True, metavar="ID")
    p.add_argument("--dst-features", nargs="+", type=str, required=True, metavar="ID")

    p.add_argument("--mode", type=str, default="set", choices=("set", "add"))
    p.add_argument("--value", type=float, default=5.0)
    p.add_argument("--counterfactual-scale", type=float, default=0.0)
    p.add_argument("--debug-zero-act", action="store_true")

    p.add_argument("--out", type=str, required=True, help="Output path, e.g. circuit.dot")
    p.add_argument(
        "--min-abs-edge",
        type=float,
        default=0.0,
        help="Skip drawing edges with |Δf·∂ℒ/∂f| below this (after building the full table).",
    )
    p.add_argument("--penwidth-scale", type=float, default=8.0)

    p.add_argument(
        "--three-node",
        action="store_true",
        help="Also write tripartite DAG (src latent → hook_z head → dst latent). Uses --out-three-node or <out-stem>_three_node.dot.",
    )
    p.add_argument(
        "--middle-head",
        nargs=2,
        type=int,
        metavar=("L", "H"),
        action="append",
        help="With --three-node: repeatable layer and head index (e.g. --middle-head 9 8).",
    )
    p.add_argument(
        "--out-three-node",
        type=str,
        default="",
        help="Tripartite DOT path. Default: <out-stem>_three_node<suffix> next to --out.",
    )
    p.add_argument(
        "--z-seq-pos",
        type=int,
        default=None,
        help="Token index for Δz and ∂ℒ/∂z slices (defaults to resolved dst seq pos).",
    )
    p.add_argument("--head-patch-seq-pos", type=int, default=None, help="Aligned hook_z patch index for mid→dst edges.")
    p.add_argument("--head-patch-cross-clean", type=int, default=None, metavar="I")
    p.add_argument("--head-patch-cross-corrupt", type=int, default=None, metavar="J")

    args = p.parse_args()

    if args.three_node and not args.middle_head:
        raise SystemExit("--three-node requires at least one --middle-head L H.")
    hpc, hpr = args.head_patch_cross_clean, args.head_patch_cross_corrupt
    if (hpc is None) ^ (hpr is None):
        raise SystemExit("Provide both --head-patch-cross-clean and --head-patch-cross-corrupt, or neither.")
    head_patch_positions: int | tuple[int, int] | None
    if hpc is not None and hpr is not None:
        head_patch_positions = (int(hpc), int(hpr))
    elif args.head_patch_seq_pos is not None:
        head_patch_positions = int(args.head_patch_seq_pos)
    else:
        head_patch_positions = None

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    device = torch.device(args.device)
    load_kw: dict = dict(device=device)
    if bool(args.tl_no_processing):
        load_kw.update(
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
        )
    else:
        load_kw.update(
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
        )
    model = HookedTransformer.from_pretrained(str(args.model), **load_kw)
    model.eval()

    d_model = int(model.cfg.d_model)
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
    assert_d_in_matches_model(src_sae, d_model=d_model)
    assert_d_in_matches_model(dst_sae, d_model=d_model)
    encode_src, decode_src = discovery_encode_decode(src_sae)
    encode_dst, decode_dst = discovery_encode_decode(dst_sae)

    clean_tok = int(model.to_single_token(args.clean_answer))
    corrupt_tok = int(model.to_single_token(args.corrupt_answer))

    loss_pos_raw = args.loss_seq_pos if args.loss_seq_pos is not None else args.seq_pos

    def logits_to_scalar_loss(logits: torch.Tensor, /) -> torch.Tensor:
        pos = int(loss_pos_raw)
        if pos < 0:
            pos += int(logits.shape[1])
        score = logits[0, pos, clean_tok] - logits[0, pos, corrupt_tok]
        return -score

    src_ids = [int(x) for x in args.src_features]
    dst_ids = [int(x) for x in args.dst_features]

    built = build_cross_layer_edges(
        model=model,
        prompt_clean=str(args.clean_prompt),
        prompt_corrupt=str(args.corrupt_prompt),
        src_hook=str(args.src_sae_id),
        dst_hook=str(args.dst_sae_id),
        encode_src=encode_src,
        decode_src=decode_src,
        encode_dst=encode_dst,
        decode_dst=decode_dst,
        logits_to_scalar_loss=logits_to_scalar_loss,
        src_feature_ids=src_ids,
        dst_feature_ids=dst_ids,
        metric="logit_diff",
        seq_pos=int(args.seq_pos),
        src_seq_pos=args.src_seq_pos,
        dst_seq_pos=args.dst_seq_pos,
        prepend_bos=True if bool(args.prepend_bos) else False,
        device=device,
        intervention_mode=str(args.mode),
        intervention_value=float(args.value),
        counterfactual_scale=float(args.counterfactual_scale),
        debug_zero_act=bool(args.debug_zero_act),
    )

    out_path = Path(str(args.out))
    title = (
        f"{args.model}  {args.src_sae_id}@{built.src_seq_pos_resolved} → "
        f"{args.dst_sae_id}@{built.dst_seq_pos_resolved}\n"
        f"logit_diff @ loss_pos={loss_pos_raw}"
    )
    write_bipartite_sae_dot(
        out=out_path,
        src_feature_ids=src_ids,
        dst_feature_ids=dst_ids,
        edge_weight=built.edge_weight,
        src_taylor=built.src_taylor,
        dst_taylor=built.dst_taylor,
        src_cluster_label=f"Source latents ({args.src_sae_id})",
        dst_cluster_label=f"Bottleneck latents ({args.dst_sae_id})",
        min_abs_edge=float(args.min_abs_edge),
        penwidth_scale=float(args.penwidth_scale),
        title=title,
    )

    print(f"wrote {out_path.resolve()}")
    print(
        f"positions (0-based): src={built.src_seq_pos_resolved} dst={built.dst_seq_pos_resolved} "
        f"loss_raw_index={loss_pos_raw}"
    )
    print("render: dot -Tpdf", str(out_path), "-o circuit.pdf")
    for (i, j), w in sorted(built.edge_weight.items(), key=lambda x: -abs(x[1])):
        print(f"  edge {i}→{j}: {w:+.6g}")

    if args.three_node:
        assert args.middle_head is not None
        middle = [(int(L), int(H)) for L, H in args.middle_head]
        triple = build_three_node_edges(
            model=model,
            prompt_clean=str(args.clean_prompt),
            prompt_corrupt=str(args.corrupt_prompt),
            src_hook=str(args.src_sae_id),
            dst_hook=str(args.dst_sae_id),
            encode_src=encode_src,
            decode_src=decode_src,
            encode_dst=encode_dst,
            decode_dst=decode_dst,
            logits_to_scalar_loss=logits_to_scalar_loss,
            src_feature_ids=src_ids,
            dst_feature_ids=dst_ids,
            middle_heads=middle,
            metric="logit_diff",
            seq_pos=int(args.seq_pos),
            src_seq_pos=args.src_seq_pos,
            dst_seq_pos=args.dst_seq_pos,
            z_seq_pos=args.z_seq_pos,
            head_patch_positions=head_patch_positions,
            prepend_bos=True if bool(args.prepend_bos) else False,
            device=device,
            intervention_mode=str(args.mode),
            intervention_value=float(args.value),
            counterfactual_scale=float(args.counterfactual_scale),
            debug_zero_act=bool(args.debug_zero_act),
        )

        out_bi = Path(str(args.out))
        if str(args.out_three_node).strip():
            three_path = Path(str(args.out_three_node).strip())
        else:
            three_path = out_bi.with_name(f"{out_bi.stem}_three_node{out_bi.suffix}")

        title_trip = (
            f"{args.model} three-node z@{triple.z_seq_pos_resolved} heads={middle}\n"
            f"logit_diff @ loss_pos={loss_pos_raw}"
        )
        write_tripartite_sae_head_dot(
            out=three_path,
            src_feature_ids=src_ids,
            dst_feature_ids=dst_ids,
            middle_heads=middle,
            edge_src_to_mid=triple.edge_src_to_mid,
            edge_mid_to_dst=triple.edge_mid_to_dst,
            src_taylor=triple.bipartite.src_taylor,
            dst_taylor=triple.bipartite.dst_taylor,
            src_cluster_label=f"Source ({args.src_sae_id})",
            mid_cluster_label="Attention hook_z",
            dst_cluster_label=f"Destination ({args.dst_sae_id})",
            min_abs_edge=float(args.min_abs_edge),
            penwidth_scale=float(args.penwidth_scale),
            title=title_trip,
        )
        print(f"wrote tripartite {three_path.resolve()}")
        print("render: dot -Tpdf", str(three_path), "-o tripartite.pdf")
        for k, w in sorted(triple.edge_src_to_mid.items(), key=lambda x: -abs(x[1]))[:20]:
            print(f"  src {k[0]}→mid L{k[1][0]}H{k[1][1]}: {w:+.6g}")
        for k, w in sorted(triple.edge_mid_to_dst.items(), key=lambda x: -abs(x[1]))[:20]:
            print(f"  mid L{k[0][0]}H{k[0][1]}→dst {k[1]}: {w:+.6g}")


if __name__ == "__main__":
    main()
