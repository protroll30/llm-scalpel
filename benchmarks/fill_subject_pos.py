"""
Fill missing per-row ``subject_pos`` in an existing benchmark JSON.

``subject_pos`` is defined as the index of the *first token* where ``clean`` and ``corrupt`` differ
under the model tokenizer (optionally with BOS prepended).

This script is useful if a later enrichment step produced/merged a dataset that lost ``subject_pos``
for some rows; downstream scripts then fall back to ``seq_pos_fallback`` (often the last token),
which can be undesirable.

Notes
-----
- If token lengths differ between clean/corrupt, the token-diff definition is undefined. In that
  case we set ``subject_pos`` to a fallback (default: last token) and record why.

Example
-------
python benchmarks/fill_subject_pos.py --in benchmarks/processed/factual_recall_filtered_enriched.json --out benchmarks/processed/factual_recall_filtered_enriched_subjectpos.json --device cuda
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


def subject_pos_from_clean_corrupt_diff(
    model: HookedTransformer,
    clean: str,
    corrupt: str,
    *,
    prepend_bos: bool,
) -> int | None:
    tc = model.to_tokens(clean, prepend_bos=prepend_bos)
    tr = model.to_tokens(corrupt, prepend_bos=prepend_bos)
    if tc.shape != tr.shape:
        return None
    n = int(tc.shape[-1])
    for i in range(n):
        if int(tc[0, i].item()) != int(tr[0, i].item()):
            return i
    return None


def _resolve_pos(seq_pos: int, seq_len: int) -> int:
    pos = int(seq_pos)
    if pos < 0:
        pos += int(seq_len)
    if pos < 0 or pos >= int(seq_len):
        raise IndexError(f"seq_pos resolved to {pos}, invalid for seq_len={seq_len}")
    return pos


def main() -> int:
    p = argparse.ArgumentParser(description="Fill missing subject_pos fields in benchmark JSON pairs.")
    p.add_argument("--in", "-i", dest="in_path", type=Path, required=True)
    p.add_argument("--out", "-o", dest="out_path", type=Path, default=None, help="Default: overwrite --in")
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument(
        "--prepend-bos",
        action="store_true",
        help="Must match the setting used by downstream enrichment/interventions.",
    )
    p.add_argument(
        "--only-missing",
        action="store_true",
        default=True,
        help="Only fill rows where subject_pos is absent. Default: on.",
    )
    p.add_argument(
        "--no-only-missing",
        action="store_false",
        dest="only_missing",
        help="Recompute/overwrite subject_pos for all rows.",
    )
    p.add_argument(
        "--fallback",
        type=str,
        default="last",
        choices=("last", "cli"),
        help="Fallback when token lengths differ or no diff is found. Default: last.",
    )
    p.add_argument("--fallback-pos", type=int, default=4, help="Used only when --fallback cli (default: 4).")
    p.add_argument("--dry-run", action="store_true", help="Do not write output file.")
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

    n_written = 0
    n_skipped = 0
    n_missing_fields = 0
    n_fallback = 0
    n_same_tokens = 0
    n_len_mismatch = 0
    errors = 0

    for row in tqdm(pairs, desc="fill_subject_pos"):
        if not isinstance(row, dict):
            continue
        if args.only_missing and "subject_pos" in row:
            n_skipped += 1
            continue

        clean = row.get("clean")
        corrupt = row.get("corrupt")
        if not isinstance(clean, str) or not isinstance(corrupt, str):
            n_missing_fields += 1
            continue

        try:
            sp = subject_pos_from_clean_corrupt_diff(model, clean, corrupt, prepend_bos=bool(args.prepend_bos))
            if sp is None:
                tc = model.to_tokens(clean, prepend_bos=bool(args.prepend_bos))
                tr = model.to_tokens(corrupt, prepend_bos=bool(args.prepend_bos))
                if tc.shape != tr.shape:
                    n_len_mismatch += 1
                    seq_len = int(tc.shape[-1])
                else:
                    n_same_tokens += 1
                    seq_len = int(tc.shape[-1])

                if str(args.fallback).lower() == "last":
                    sp_eff = _resolve_pos(-1, seq_len)
                    src = "fallback_last"
                else:
                    sp_eff = _resolve_pos(int(args.fallback_pos), seq_len)
                    src = "fallback_cli"

                row["subject_pos"] = int(sp_eff)
                row["subject_pos_source"] = src
                row["subject_pos_reason"] = "len_mismatch" if tc.shape != tr.shape else "no_token_diff"
                n_fallback += 1
            else:
                row["subject_pos"] = int(sp)
                row["subject_pos_source"] = "token_diff"
                row.pop("subject_pos_reason", None)
            row["subject_pos_prepend_bos"] = bool(args.prepend_bos)
            n_written += 1
        except Exception as e:
            errors += 1
            pid = row.get("id", "?")
            print(f"[error id={pid}] {type(e).__name__}: {e}", file=sys.stderr)

    gm = data.get("generator_meta")
    if isinstance(gm, dict):
        gm = dict(gm)
        gm["subject_pos_fill"] = {
            "script": "benchmarks/fill_subject_pos.py",
            "model": args.model,
            "prepend_bos": bool(args.prepend_bos),
            "only_missing": bool(args.only_missing),
            "fallback": str(args.fallback),
            "fallback_pos": int(args.fallback_pos),
            "stats": {
                "written": n_written,
                "skipped_existing": n_skipped,
                "missing_clean_or_corrupt": n_missing_fields,
                "fallback_used": n_fallback,
                "fallback_len_mismatch": n_len_mismatch,
                "fallback_no_token_diff": n_same_tokens,
                "errors": errors,
            },
        }
        data["generator_meta"] = gm

    print(
        "done: "
        f"written={n_written} skipped_existing={n_skipped} missing_fields={n_missing_fields} "
        f"fallback_used={n_fallback} errors={errors}"
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

