"""
Fill per-row ``src_feature_ids`` (Layer-8 SAE) from the **clean** prompt at the **subject / metric** position.

Top-|f| only at ``is`` (last token) reinforces generic template latents. Prefer per-row positions from the
benchmark JSON: ``subject_pos`` first, then ``metric_seq_pos``, ``loss_seq_pos``, ``seq_pos``.
If none are set, use ``--seq-pos-fallback``
(``last`` = old behavior at final prompt token; ``cli`` = ``--seq-pos``).

Use this when you run interventions on the **corrupt** prompt but want source latents from the clean subject
region. ``discovery.benchmark_json.inject_features_from_benchmark_row`` reads ``src_feature_ids``.

Example::

  python benchmarks/enrich_source_sae_ids.py --in benchmarks/processed/factual_recall_filtered_enriched.json --out benchmarks/processed/factual_recall_filtered_enriched.json --device cuda --top-k 8 --skip-dropped --seq-pos-fallback last

Then batch intervene **without** ``--dynamic-src-top-k`` so each row uses the stored clean-derived ids.
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


def _seq_pos_for_source_enrichment(row: dict[str, Any]) -> int | None:
    """Token index for Layer-8 source latents: prefer subject (entity) over metric/loss positions.

    Order: ``subject_pos``, then ``metric_seq_pos``, ``loss_seq_pos``, ``seq_pos`` (matches
    ``seq_pos_from_benchmark_row`` but puts subject first for subject-centric intervention).
    """
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


def _top_k_abs_feature_ids(
    *,
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_name: str,
    encode_fn,
    device: torch.device,
    k: int,
    pos_eff: int,
) -> list[int]:
    """Top-k feature indices by |f| at sequence position ``pos_eff``."""
    model.eval()
    with torch.inference_mode():
        _, cache = model.run_with_cache(tokens.to(device), names_filter=[hook_name], return_type="logits")
    if hook_name not in cache:
        raise KeyError(f"Cache missing {hook_name!r}")
    act = cache[hook_name]
    f = encode_fn(act)
    if f.dim() == 2:
        f = f.unsqueeze(0)
    if f.dim() != 3:
        raise ValueError(f"encode_fn must yield [batch, pos, n_features]; got {tuple(f.shape)}")
    seq_len = int(f.shape[1])
    pos = _resolve_pos(int(pos_eff), seq_len)
    vec = f[0, pos, :].float()
    kk = min(int(k), int(vec.numel()))
    _, top_ix = torch.topk(vec.abs(), k=kk)
    return [int(i) for i in top_ix.cpu().tolist()]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Write per-row src_feature_ids = top-|f| Layer-8 latents on CLEAN prompt at subject/metric pos."
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
    p.add_argument("--top-k", type=int, default=8, metavar="K", help="How many Layer-8 latents to store per row.")
    p.add_argument("--sae-dtype", type=str, default="float32")
    p.add_argument("--sae-force-download", action="store_true")
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
        help="Also fill dropped pairs.",
    )
    p.add_argument(
        "--only-missing",
        action="store_true",
        help="Only fill rows where src_feature_ids is absent.",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not write output file.")
    p.add_argument(
        "--seq-pos",
        type=int,
        default=4,
        help="Used only when no per-row subject_pos / metric_seq_pos / seq_pos and --seq-pos-fallback cli.",
    )
    p.add_argument(
        "--seq-pos-fallback",
        type=str,
        default="last",
        choices=("cli", "last"),
        help=(
            "When JSON has no subject/metric position: 'last' uses final prompt token (-1); "
            "'cli' uses --seq-pos."
        ),
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
    encode_src, _decode_src = discovery_encode_decode(src_sae)
    hook_name = str(args.src_sae_id)

    n_skipped_dropped = 0
    n_skipped_missing = 0
    n_skipped_already = 0
    n_written = 0
    errors = 0

    for row in tqdm(pairs, desc="src_feature_ids"):
        if not isinstance(row, dict):
            continue
        if args.skip_dropped and row.get("drop_reason"):
            n_skipped_dropped += 1
            continue
        if args.only_missing and "src_feature_ids" in row:
            n_skipped_already += 1
            continue
        clean = row.get("clean")
        if not isinstance(clean, str) or not clean.strip():
            n_skipped_missing += 1
            continue

        try:
            tokens = model.to_tokens(clean, prepend_bos=bool(args.prepend_bos))
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

            ids = _top_k_abs_feature_ids(
                model=model,
                tokens=tokens,
                hook_name=hook_name,
                encode_fn=encode_src,
                device=device,
                k=k,
                pos_eff=pos_eff,
            )
            row["src_feature_ids"] = ids
            row["src_feature_ids_seq_pos"] = pos_eff
            row["src_feature_ids_seq_pos_source"] = pos_src
            n_written += 1
        except Exception as e:
            errors += 1
            pid = row.get("id", "?")
            print(f"[error id={pid}] {e}", file=sys.stderr)

    gm = data.get("generator_meta")
    if isinstance(gm, dict):
        gm = dict(gm)
        gm["src_feature_ids_enrichment"] = {
            "script": "benchmarks/enrich_source_sae_ids.py",
            "rule": "top_k_abs_encoded_src_sae_at_benchmark_seq_pos_on_clean",
            "position": "subject_pos first, then metric_seq_pos / loss_seq_pos / seq_pos; else seq-pos-fallback",
            "prompt_field": "clean",
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
        f"skipped_bad_row={n_skipped_missing} skipped_already_present={n_skipped_already} errors={errors}"
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
