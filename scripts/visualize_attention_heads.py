"""Plot attention patterns (hook_pattern) for chosen heads on one prompt.

TransformerLens ``hook_pattern`` is post-softmax weights, typically shaped
``[batch, n_heads, query_pos, key_pos]``.

Example::

  python scripts/visualize_attention_heads.py --device cuda \\
    --prompt \"The capital of Germany is\" \\
    --no-prepend-bos \\
    --layer-head 9 8 --layer-head 8 11 --layer-head 9 3 \\
    --query-pos -2 --key-pos -3 \\
    --out-dir runs/attn_viz

Uses ``--query-pos`` / ``--key-pos`` (0-based, negatives OK) to annotate the weight
``pattern[query, key]`` in the console title (e.g. `` is`` -> country token).

Benchmark JSON::

  python scripts/visualize_attention_heads.py --benchmark-json benchmarks/processed/factual_recall_filtered_enriched.json \\
    --benchmark-prompt-field corrupt --layer-head 9 8 --out-dir runs/attn_viz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer, utils as tl_utils

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from discovery.benchmark_json import (
    add_discovery_benchmark_cli_args,
    apply_benchmark_single_prompt,
)


def _normalize_tick_labels(labels: list[str] | list[list[str]]) -> list[str]:
    # TransformerLens typings can expose nested token strings in some cases.
    # Matplotlib expects a flat iterable of text, so we normalize to `list[str]`.
    out: list[str] = []
    for x in labels:
        if isinstance(x, list):
            out.append("\n".join(str(p) for p in x))
        else:
            out.append(str(x))
    return out


def _resolve_pos(idx: int, seq_len: int) -> int:
    i = int(idx)
    if i < 0:
        i += seq_len
    if i < 0 or i >= seq_len:
        raise IndexError(f"Position {idx!r} resolves to {i}, invalid for seq_len={seq_len}")
    return i


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize attention patterns for specific heads.")
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument(
        "--tl-no-processing",
        action="store_true",
        help="Skip TransformerLens fold_ln/center_*.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Corrupt or clean prompt text (required unless --benchmark-json is set).",
    )
    p.add_argument("--prepend-bos", action="store_true")
    p.add_argument("--no-prepend-bos", action="store_true")

    add_discovery_benchmark_cli_args(p)
    p.add_argument(
        "--benchmark-prompt-field",
        type=str,
        default="corrupt",
        choices=("clean", "corrupt"),
        help="When using --benchmark-json: take this row field as --prompt.",
    )

    p.add_argument(
        "--layer-head",
        nargs=2,
        type=int,
        metavar=("L", "H"),
        action="append",
        required=True,
        help="Repeatable: layer index and head index (e.g. --layer-head 9 8).",
    )
    p.add_argument(
        "--query-pos",
        type=int,
        default=None,
        help="Query token index for highlighted weight (0-based; negatives supported).",
    )
    p.add_argument(
        "--key-pos",
        type=int,
        default=None,
        help="Key token index for highlighted weight (0-based; negatives supported).",
    )

    p.add_argument("--out-dir", type=str, default="", help="If set, save one PNG per head here.")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--show", action="store_true", help="Call plt.show() (needs a display).")

    args = p.parse_args()
    apply_benchmark_single_prompt(args, prompt_attr="prompt", field=str(args.benchmark_prompt_field))

    if not str(args.prompt).strip():
        raise SystemExit("Provide --prompt or --benchmark-json (with a non-empty pair).")

    if args.prepend_bos and args.no_prepend_bos:
        raise SystemExit("Use at most one of --prepend-bos / --no-prepend-bos.")

    tk_kw: dict = {}
    if args.prepend_bos:
        tk_kw["prepend_bos"] = True
    elif args.no_prepend_bos:
        tk_kw["prepend_bos"] = False

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

    tokens = model.to_tokens(str(args.prompt), **tk_kw).to(device)
    seq_len = int(tokens.shape[-1])
    try:
        tok_labels = _normalize_tick_labels(model.to_str_tokens(tokens[0]))
    except Exception:
        tok_labels = [str(i) for i in range(seq_len)]

    pairs = [(int(L), int(H)) for L, H in args.layer_head]
    pattern_hooks = sorted({tl_utils.get_act_name("pattern", L) for L, _ in pairs})

    logits, cache = model.run_with_cache(
        tokens,
        names_filter=pattern_hooks,
        return_type="logits",
    )

    q_star = _resolve_pos(args.query_pos, seq_len) if args.query_pos is not None else None
    k_star = _resolve_pos(args.key_pos, seq_len) if args.key_pos is not None else None

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for L, H in pairs:
        hook = tl_utils.get_act_name("pattern", L)
        if hook not in cache:
            raise KeyError(f"Cache missing {hook!r}")
        pat = cache[hook]
        if pat.dim() != 4:
            raise ValueError(f"Expected rank-4 pattern tensor, got shape {tuple(pat.shape)}")
        # [batch, head, q, k]
        mat = pat[0, int(H)].detach().float().cpu().numpy()

        fig_w = max(6.0, seq_len * 0.35)
        fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
        im = ax.imshow(mat, vmin=0.0, vmax=float(mat.max()) if mat.size else 1.0, cmap="magma")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tok_labels, rotation=65, ha="right", fontsize=8)
        ax.set_yticklabels(tok_labels, fontsize=8)
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")

        subtitle_parts = [f"L{L}H{H}", hook]
        if q_star is not None and k_star is not None:
            w = float(mat[q_star, k_star])
            subtitle_parts.append(f"A[q={q_star}→k={k_star}]={w:.4f}")
            ax.axhline(q_star - 0.5, color="cyan", linewidth=0.8, alpha=0.6)
            ax.axvline(k_star - 0.5, color="cyan", linewidth=0.8, alpha=0.6)
        ax.set_title(" ".join(subtitle_parts), fontsize=10)

        plt.tight_layout()
        if out_dir is not None:
            path = out_dir / f"attn_L{L}_H{H}.png"
            fig.savefig(path, dpi=int(args.dpi))
            print(f"wrote {path.resolve()}")
        if bool(args.show):
            plt.show()
        else:
            plt.close(fig)

        if q_star is not None and k_star is not None:
            row = mat[q_star]
            print(f"L{L}H{H}: weight(q={q_star}, k={k_star})={float(mat[q_star, k_star]):.6f}")
            topk = min(5, seq_len)
            top_keys = np.argsort(-row)[:topk]
            pairs_txt = ", ".join(f"k={int(j)}:{float(row[j]):.3f}" for j in top_keys)
            print(f"         top-{topk} keys for this query: {pairs_txt}")


if __name__ == "__main__":
    main()
