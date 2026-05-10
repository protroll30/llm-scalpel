"""
Fill ``correct_answer_id`` on each benchmark pair with argmax |dst SAE features| at the last prompt position.

Uses the **clean** prompt so the id reflects residual structure where the model should favor the stored
``correct_answer`` continuation (same semantics as discovery readout). Matches ``intervene_layer8_to_layer9``
defaults: ``gpt2-small``, ``blocks.10.hook_resid_pre``, SAELens release ``gpt2-small-res-jb``.

Example::

  python benchmarks/enrich_correct_answer_sae_ids.py \\
    --in benchmarks/processed/factual_recall_filtered_enriched.json \\
    --out benchmarks/processed/factual_recall_filtered_enriched.json \\
    --device cuda \\
    --skip-dropped
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


def _top_abs_latent_id(
    *,
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_name: str,
    encode_fn,
    device: torch.device,
) -> tuple[int, float, float]:
    """Return (feature_id, signed_activation_at_that_id, max_abs_in_vector)."""
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
    pos = seq_len - 1
    vec = f[0, pos, :].float()
    ix = int(torch.argmax(torch.abs(vec)).item())
    return ix, float(vec[ix].item()), float(torch.abs(vec).max().item())


def main() -> int:
    p = argparse.ArgumentParser(
        description="Write per-row correct_answer_id = argmax |dst SAE latent| at last prompt position (clean)."
    )
    p.add_argument("--in", "-i", dest="in_path", type=Path, required=True)
    p.add_argument("--out", "-o", dest="out_path", type=Path, default=None, help="Default: overwrite --in")
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--prepend-bos", action="store_true", help="Match tokenizer behavior used in downstream scripts.")
    p.add_argument("--sae-release", type=str, default="gpt2-small-res-jb")
    p.add_argument("--dst-sae-id", type=str, default="blocks.10.hook_resid_pre")
    p.add_argument("--sae-dtype", type=str, default="float32")
    p.add_argument("--sae-force-download", action="store_true")
    p.add_argument(
        "--skip-dropped",
        action="store_true",
        default=True,
        help="Skip rows that have drop_reason (probability filter rejects). Default: on.",
    )
    p.add_argument(
        "--no-skip-dropped",
        action="store_false",
        dest="skip_dropped",
        help="Also score dropped pairs (slower; useful if you merged dropped into pairs).",
    )
    p.add_argument(
        "--only-missing",
        action="store_true",
        help="Only fill rows where correct_answer_id is absent.",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not write output file.")
    p.add_argument(
        "--per-row-meta",
        action="store_true",
        help="Also write correct_answer_id_meta on each row (activation stats).",
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

    dst_sae = load_pretrained_sae(
        release=str(args.sae_release),
        sae_id=str(args.dst_sae_id),
        device=device,
        dtype=str(args.sae_dtype),
        force_download=bool(args.sae_force_download),
    )
    assert_d_in_matches_model(dst_sae, d_model=int(model.cfg.d_model))
    encode_dst, _decode_dst = discovery_encode_decode(dst_sae)
    hook_name = str(args.dst_sae_id)

    n_skipped_dropped = 0
    n_skipped_missing = 0
    n_skipped_already = 0
    n_written = 0
    errors = 0

    for row in tqdm(pairs, desc="correct_answer_id"):
        if not isinstance(row, dict):
            continue
        if args.skip_dropped and row.get("drop_reason"):
            n_skipped_dropped += 1
            continue
        if args.only_missing and "correct_answer_id" in row:
            n_skipped_already += 1
            continue
        clean = row.get("clean")
        if not isinstance(clean, str) or not clean.strip():
            n_skipped_missing += 1
            continue

        try:
            tokens = model.to_tokens(clean, prepend_bos=bool(args.prepend_bos))
            fid, signed_v, max_abs = _top_abs_latent_id(
                model=model,
                tokens=tokens,
                hook_name=hook_name,
                encode_fn=encode_dst,
                device=device,
            )
            row["correct_answer_id"] = fid
            if args.per_row_meta:
                row["correct_answer_id_meta"] = {
                    "signed_activation": signed_v,
                    "max_abs_activation": max_abs,
                    "hook": hook_name,
                    "seq_pos": -1,
                    "prompt_field": "clean",
                }
            elif "correct_answer_id_meta" in row:
                del row["correct_answer_id_meta"]
            n_written += 1
        except Exception as e:
            errors += 1
            pid = row.get("id", "?")
            print(f"[error id={pid}] {e}", file=sys.stderr)

    gm = data.get("generator_meta")
    if isinstance(gm, dict):
        gm = dict(gm)
        gm["correct_answer_id_enrichment"] = {
            "script": "benchmarks/enrich_correct_answer_sae_ids.py",
            "rule": "argmax_abs_encoded_dst_sae_at_last_prompt_position",
            "model": args.model,
            "sae_release": args.sae_release,
            "dst_sae_id": args.dst_sae_id,
            "prepend_bos": bool(args.prepend_bos),
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
