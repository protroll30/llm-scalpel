"""
Fill per-row ``src_feature_ids`` via **single-feature causal ablation** on the **clean** prompt (Layer-8 SAE).

For each benchmark row:

1. Forward the **clean** prompt and read SAE latents at the chosen sequence position (subject-first
   resolution, same as ``enrich_source_sae_ids.py``).
2. Collect **active** latents at that position (default: ``f[j] > 0``, typical for ReLU SAEs).
3. For **each** active latent ``j``, run one patched forward that sets **only** ``f[batch, pos, j]`` to
   ``0`` at that position (residual-aware reconstruction via ``reconstruct_activation``).
4. Score ``drop_j = logit_answer(base) - logit_answer(ablated_j)`` at the last prompt position for the
   **first token** of ``correct_answer``.
5. Store the **top-K** ``j`` with the **largest drops** (features whose removal hurts the answer logit most).

This is **activation patching / causal scrubbing** at SAE feature granularity. It is slower than
``enrich_source_sae_ids_logit_attr.py`` (gradients) but matches the “biggest logit drop” definition directly.

Example::

  python benchmarks/enrich_source_sae_ids_ablation.py --in benchmarks/processed/factual_recall_filtered_enriched.json --out benchmarks/processed/factual_recall_filtered_enriched_ablation.json --device cuda --top-k 8 --skip-dropped
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from transformer_lens import HookedTransformer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from discovery.sae_lens_bridge import assert_d_in_matches_model, discovery_encode_decode, load_pretrained_sae
from discovery.sae_scout import reconstruct_activation


def _seq_pos_for_source_enrichment(row: dict[str, Any]) -> int | None:
    for key in ("subject_pos", "metric_seq_pos", "loss_seq_pos", "seq_pos"):
        if key not in row:
            continue
        v = row[key]
        if v is None:
            continue
        try:
            return int(v)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_pos(seq_pos: int, seq_len: int) -> int:
    pos = int(seq_pos)
    if pos < 0:
        pos += int(seq_len)
    if pos < 0 or pos >= int(seq_len):
        raise IndexError(f"seq_pos resolved to {pos}, invalid for seq_len={seq_len}")
    return pos


def _answer_token_id(model: HookedTransformer, answer: str) -> int:
    t = model.to_tokens(answer, prepend_bos=False)
    if t.numel() < 1:
        raise ValueError(f"Empty answer after tokenization: {answer!r}")
    return int(t[0, 0].item())


def _answer_logit(logits: torch.Tensor, answer_tid: int) -> float:
    return float(logits[0, -1, answer_tid].detach().cpu().item())


def _logits_with_masked_features_at_pos(
    *,
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_name: str,
    encode_fn,
    decode_fn,
    pos_eff: int,
    zero_indices: set[int],
) -> torch.Tensor:
    """One forward; zeros listed SAE features **only** at ``pos_eff`` before decode/reconstruction."""

    def _hook(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        f = encode_fn(act)
        if f.dim() == 2:
            f = f.unsqueeze(0)
        if f.dim() != 3:
            raise ValueError(
                f"encode_fn must return [batch, pos, n_features]; got {tuple(f.shape)}"
            )
        f_full = f
        f_masked = f_full.clone()
        if zero_indices:
            for j in zero_indices:
                f_masked[0, pos_eff, j] = 0.0
        return reconstruct_activation(
            f_patched=f_masked,
            x_corrupt=act,
            f_corrupt=f_full,
            decode_fn=decode_fn,
        )

    model.eval()
    with torch.no_grad():
        return model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, _hook)],
            return_type="logits",
        )


def _active_feature_indices(vec: torch.Tensor, *, threshold: float) -> list[int]:
    """Indices where activation is strictly above ``threshold`` (ReLU latents: > 0)."""
    v = vec.detach().float().flatten()
    mask = v > float(threshold)
    return [int(i) for i in torch.nonzero(mask, as_tuple=False).flatten().tolist()]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Write src_feature_ids = top-K SAE features by causal answer-logit drop (single-feature ablation)."
    )
    p.add_argument("--in", "-i", dest="in_path", type=Path, required=True)
    p.add_argument("--out", "-o", dest="out_path", type=Path, default=None, help="Default: overwrite --in")
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument(
        "--prepend-bos",
        action="store_true",
        help="Match tokenizer behavior used in intervene_layer8_to_layer9 (default: off).",
    )
    p.add_argument("--sae-release", type=str, default="gpt2-small-res-jb")
    p.add_argument("--src-sae-id", type=str, default="blocks.8.hook_resid_pre")
    p.add_argument("--top-k", type=int, default=8, metavar="K")
    p.add_argument("--sae-dtype", type=str, default="float32")
    p.add_argument("--sae-force-download", action="store_true")
    p.add_argument(
        "--active-threshold",
        type=float,
        default=0.0,
        help="Treat latent j as active at seq_pos if f[j] > this value (default: 0).",
    )
    p.add_argument(
        "--max-ablation-candidates",
        type=int,
        default=0,
        help="If >0, only ablate the strongest active latents by value (cap count). 0 = no cap.",
    )
    p.add_argument(
        "--skip-dropped",
        action="store_true",
        default=True,
        help="Skip rows that have drop_reason. Default: on.",
    )
    p.add_argument(
        "--no-skip-dropped",
        action="store_false",
        dest="skip_dropped",
        help="Also score dropped pairs.",
    )
    p.add_argument(
        "--only-missing",
        action="store_true",
        help="Only fill rows where src_feature_ids is absent.",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not write output file.")
    p.add_argument("--seq-pos", type=int, default=4)
    p.add_argument(
        "--seq-pos-fallback",
        type=str,
        default="last",
        choices=("cli", "last"),
        help="When JSON has no subject/metric seq keys.",
    )
    args = p.parse_args()

    out_path = args.out_path or args.in_path
    data = json.loads(args.in_path.read_text(encoding="utf-8"))
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        print("Input JSON must contain a 'pairs' array.", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; use --device cpu.", file=sys.stderr)
        return 1

    k = max(1, int(args.top_k))
    cap = int(args.max_ablation_candidates)

    print(f"Loading model {args.model!r} on {device}...")
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
    assert_d_in_matches_model(src_sae, d_model=int(model.cfg.d_model))
    encode_src, decode_src = discovery_encode_decode(src_sae)
    hook_name = str(args.src_sae_id)

    n_skipped_dropped = 0
    n_skipped_missing = 0
    n_skipped_already = 0
    n_written = 0
    n_empty_active = 0
    errors = 0

    for row in tqdm(pairs, desc="src_feature_ids_ablation"):
        if not isinstance(row, dict):
            continue
        if args.skip_dropped and row.get("drop_reason"):
            n_skipped_dropped += 1
            continue
        if args.only_missing and "src_feature_ids" in row:
            n_skipped_already += 1
            continue
        clean = row.get("clean")
        ans = row.get("correct_answer")
        if not isinstance(clean, str) or not clean.strip():
            n_skipped_missing += 1
            continue
        if not isinstance(ans, str) or not ans.strip():
            n_skipped_missing += 1
            continue

        try:
            tokens = model.to_tokens(clean, prepend_bos=bool(args.prepend_bos)).to(device)
            seq_len = int(tokens.shape[-1])
            raw_sp = _seq_pos_for_source_enrichment(row)
            if raw_sp is not None:
                pos_eff = _resolve_pos(int(raw_sp), seq_len)
                pos_src = "benchmark_json"
            elif str(args.seq_pos_fallback).strip().lower() == "last":
                pos_eff = _resolve_pos(-1, seq_len)
                pos_src = "fallback_last"
            else:
                pos_eff = _resolve_pos(int(args.seq_pos), seq_len)
                pos_src = "fallback_cli"

            tid = _answer_token_id(model, ans)

            logits_base = _logits_with_masked_features_at_pos(
                model=model,
                tokens=tokens,
                hook_name=hook_name,
                encode_fn=encode_src,
                decode_fn=decode_src,
                pos_eff=pos_eff,
                zero_indices=set(),
            )
            base_logit = _answer_logit(logits_base, tid)

            # Encode once (no grad) to list active latents at pos_eff
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[hook_name], return_type="logits")
                act = cache[hook_name]
                f_raw = encode_src(act)
                if f_raw.dim() == 2:
                    f_raw = f_raw.unsqueeze(0)
                vec = f_raw[0, pos_eff, :].float()

            active = _active_feature_indices(vec, threshold=float(args.active_threshold))
            if cap > 0 and len(active) > cap:
                active_sorted = sorted(active, key=lambda j: float(vec[j].item()), reverse=True)
                active = active_sorted[:cap]

            if not active:
                n_empty_active += 1
                continue

            drops: list[tuple[int, float]] = []
            for j in active:
                logits_ab = _logits_with_masked_features_at_pos(
                    model=model,
                    tokens=tokens,
                    hook_name=hook_name,
                    encode_fn=encode_src,
                    decode_fn=decode_src,
                    pos_eff=pos_eff,
                    zero_indices={j},
                )
                ab_logit = _answer_logit(logits_ab, tid)
                drops.append((j, base_logit - ab_logit))

            drops.sort(key=lambda x: x[1], reverse=True)
            top = drops[: min(k, len(drops))]
            ids = [int(j) for j, _ in top]

            row["src_feature_ids"] = ids
            row["src_feature_ids_seq_pos"] = pos_eff
            row["src_feature_ids_seq_pos_source"] = pos_src
            row["src_feature_ids_answer_token_id"] = tid
            row["src_feature_ids_ablation_meta"] = {
                "base_answer_logit": base_logit,
                "n_active_candidates": len(active),
                "top_drops": [{"feature_id": int(j), "logit_drop": float(d)} for j, d in top],
            }
            n_written += 1
        except Exception as e:
            errors += 1
            pid = row.get("id", "?")
            print(f"[error id={pid}] {e}", file=sys.stderr)

    gm = data.get("generator_meta")
    if isinstance(gm, dict):
        gm = dict(gm)
        gm["src_feature_ids_enrichment"] = {
            "script": "benchmarks/enrich_source_sae_ids_ablation.py",
            "rule": "top_k_single_feature_ablation_answer_logit_drop_at_seq_pos_on_clean",
            "position": "subject_pos first, then metric_seq_pos / loss_seq_pos / seq_pos; else seq-pos-fallback",
            "ablation": "zero one SAE latent at seq_pos per forward; score base_logit - ablated_logit",
            "active_threshold": float(args.active_threshold),
            "max_ablation_candidates": cap if cap > 0 else None,
            "prompt_field": "clean",
            "answer_logit": "logits[0, -1, first_token(correct_answer)]",
            "model": args.model,
            "sae_release": args.sae_release,
            "src_sae_id": args.src_sae_id,
            "top_k": k,
            "prepend_bos": bool(args.prepend_bos),
            "seq_pos_fallback": str(args.seq_pos_fallback),
            "skip_dropped": bool(args.skip_dropped),
        }
        data["generator_meta"] = gm

    print(
        f"done: written={n_written} skipped_dropped={n_skipped_dropped} "
        f"skipped_bad_row={n_skipped_missing} skipped_already_present={n_skipped_already} "
        f"skipped_no_active_latents={n_empty_active} errors={errors}"
    )

    if args.dry_run:
        print("[dry-run] not writing file.")
        return 0 if errors == 0 else 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote -> {out_path}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
