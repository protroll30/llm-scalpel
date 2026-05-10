"""Rank attention heads by marginal clean→corrupt patching on ``hook_z``.

Scores are ``metric(patched_corrupt) - metric(corrupt_baseline)``. Default metric is the factual
logit margin ``logit(Paris) - logit(Berlin)`` at ``--metric-seq-pos``.

Aligned patching (single token index on both prompts):

  python scripts/find_mover_heads.py --device cuda --layers 8 9 --patch-seq-pos -1

Cross-token patching (clean hook slice from ``Germany``, write into corrupt at ``is``):

  python scripts/find_mover_heads.py --device cuda --layers 8 9 --patch-cross-clean 4 --patch-cross-corrupt 5

Match ``discovery`` scripts that omit BOS (explicit):

  python scripts/find_mover_heads.py --device cuda --no-prepend-bos ...

Use a benchmark JSON (``pairs`` with ``clean`` / ``corrupt`` / ``correct_answer`` / ``corrupt_answer``)::

  python scripts/find_mover_heads.py --device cuda --layers 8 9 --benchmark-json benchmarks/processed/factual_recall_filtered_enriched.json --benchmark-index 0

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

from causal_patcher.head_patch_rank import hook_z_names_filter, marginal_head_patch_effects, metric_tensor, rank_heads
from causal_patcher.runner import ExperimentRunner
from causal_patcher.targets import PatchPos
from discovery.benchmark_json import add_discovery_benchmark_cli_args, apply_benchmark_dual_prompts


def main() -> None:
    p = argparse.ArgumentParser(description="Rank heads by marginal hook_z patching (clean into corrupt).")
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument(
        "--tl-no-processing",
        action="store_true",
        help="Skip TransformerLens fold_ln/center_* (omit unless you know you need raw HF coords).",
    )

    p.add_argument("--clean-prompt", type=str, default="The capital of France is")
    p.add_argument("--corrupt-prompt", type=str, default="The capital of Germany is")
    p.add_argument("--clean-answer", type=str, default=" Paris", help="Tokenizer token for factual answer.")
    p.add_argument("--corrupt-answer", type=str, default=" Berlin", help="Tokenizer token for counterfactual answer.")

    p.add_argument("--layers", nargs="+", type=int, required=True, metavar="L", help="Layers to sweep (e.g. 8 9).")
    p.add_argument(
        "--patch-seq-pos",
        type=int,
        default=-1,
        help="Aligned patch index on clean and corrupt runs (supports negative indices). Ignored if --patch-cross-* set.",
    )
    p.add_argument(
        "--patch-cross-clean",
        type=int,
        default=None,
        metavar="I",
        help="If set with --patch-cross-corrupt: copy hook_z from this clean token index into corrupt patch-cross-corrupt.",
    )
    p.add_argument(
        "--patch-cross-corrupt",
        type=int,
        default=None,
        metavar="J",
        help="Corrupt-side token index receiving clean hook_z from patch-cross-clean.",
    )

    p.add_argument(
        "--metric",
        type=str,
        default="logit_diff",
        choices=("logit_diff", "clean_logit", "corrupt_logit"),
        help="Scalar read off patched logits; marginal vs corrupt baseline.",
    )
    p.add_argument(
        "--metric-seq-pos",
        type=int,
        default=-1,
        help="Sequence position for metric (often -1 with patch-cross-* mover probes).",
    )

    p.add_argument("--prepend-bos", action="store_true", help="Pass prepend_bos=True to model.to_tokens.")
    p.add_argument(
        "--no-prepend-bos",
        action="store_true",
        help="Pass prepend_bos=False to model.to_tokens (align with scripts/run_discovery_real.py defaults).",
    )

    p.add_argument("--top-k", type=int, default=20, help="How many heads to print after full ranking.")

    add_discovery_benchmark_cli_args(p)

    args = p.parse_args()
    apply_benchmark_dual_prompts(args)

    if args.prepend_bos and args.no_prepend_bos:
        raise SystemExit("Use at most one of --prepend-bos / --no-prepend-bos.")
    prepend_bos: bool | None
    if args.prepend_bos:
        prepend_bos = True
    elif args.no_prepend_bos:
        prepend_bos = False
    else:
        prepend_bos = None

    cc = args.patch_cross_clean
    cr = args.patch_cross_corrupt
    if (cc is None) ^ (cr is None):
        raise SystemExit("Provide both --patch-cross-clean and --patch-cross-corrupt, or neither.")

    patch_positions: PatchPos
    if cc is not None and cr is not None:
        patch_positions = (int(cc), int(cr))
    else:
        patch_positions = int(args.patch_seq_pos)

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

    clean_tok = int(model.to_single_token(args.clean_answer))
    corrupt_tok = int(model.to_single_token(args.corrupt_answer))

    layers_sorted = sorted({int(L) for L in args.layers})
    nfilt = hook_z_names_filter(layers_sorted)

    runner = ExperimentRunner(
        model,
        str(args.clean_prompt),
        str(args.corrupt_prompt),
        clean_tok,
        corrupt_tok,
        names_filter=nfilt,
        prepend_bos=prepend_bos,
    )

    scores = marginal_head_patch_effects(
        runner,
        layers=layers_sorted,
        patch_positions=patch_positions,
        metric=args.metric,  # type: ignore[arg-type]
        metric_seq_pos=int(args.metric_seq_pos),
    )
    ranked = rank_heads(scores, top_k=int(args.top_k), by_abs=True)

    print(f"model={args.model} device={device}")
    print(f"clean={args.clean_prompt!r} corrupt={args.corrupt_prompt!r}")
    print(f"layers={layers_sorted} patch_positions={patch_positions!r} metric={args.metric} metric_seq_pos={args.metric_seq_pos}")
    base_logits = runner.corrupt_logits
    assert base_logits is not None

    base_metric = float(
        metric_tensor(
            runner,
            base_logits,
            metric=args.metric,  # type: ignore[arg-type]
            seq_pos=int(args.metric_seq_pos),
        )
        .detach()
        .cpu()
        .item()
    )
    print(f"corrupt_baseline {args.metric}={base_metric:+.6g}")
    print(f"top-{args.top_k} heads by |marginal| (marginal = patched - baseline):")
    for (L, H), m in ranked:
        print(f"  L{L}H{H}: marginal={m:+.6g}")

    print("\nfull table (layer,head,marginal):")
    for (L, H) in sorted(scores.keys()):
        print(f"{L},{H},{scores[(L, H)]:+.8g}")


if __name__ == "__main__":
    main()
