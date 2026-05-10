"""
Filter raw factual-recall JSON by a next-token probability ratio (Constraint 2).

Keeps pairs where, at the last prompt position, the softmax probability of the first
token of ``correct_answer`` on the *clean* prompt exceeds ``ratio`` times the same
probability on the *corrupt* prompt:

    P(answer | clean) > ratio * P(answer | corrupt)

Typical use after ``generate_benchmark_deepseek.py``:

  python benchmarks/build_benchmark.py \\
    --in benchmarks/raw/factual_recall_250.json \\
    --out benchmarks/processed/factual_recall_filtered.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformer_lens import HookedTransformer

_DEFAULT_IN = Path(__file__).resolve().parent / "raw" / "factual_recall_250.json"
_DEFAULT_OUT = Path(__file__).resolve().parent / "processed" / "factual_recall_filtered.json"


def load_pairs(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("Input JSON must contain a 'pairs' array")
    return data, pairs


def answer_token_id(model: HookedTransformer, answer: str) -> int:
    """First token id of ``answer`` (no BOS), matching next-token prediction at end of prompt."""
    t = model.to_tokens(answer, prepend_bos=False)
    if t.numel() < 1:
        raise ValueError(f"Empty answer after tokenization: {answer!r}")
    return int(t[0, 0].item())


def prob_next_token(
    model: HookedTransformer,
    prompt: str,
    tok_id: int,
    *,
    device: torch.device,
) -> float:
    tokens = model.to_tokens(prompt).to(device)
    with torch.inference_mode():
        logits = model(tokens)
    if logits.dim() != 3:
        raise ValueError(f"Expected logits [batch, pos, vocab], got {tuple(logits.shape)}")
    last = logits[0, -1].float()
    probs = F.softmax(last, dim=-1)
    return float(probs[tok_id].item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Probability-filter factual recall benchmark pairs.")
    parser.add_argument(
        "--input",
        "--in",
        dest="in_path",
        type=Path,
        default=_DEFAULT_IN,
        help="Raw JSON from generator",
    )
    parser.add_argument("--out", dest="out_path", type=Path, default=_DEFAULT_OUT, help="Filtered JSON output")
    parser.add_argument("--model", default="gpt2-small", help="TransformerLens model name")
    parser.add_argument(
        "--ratio",
        type=float,
        default=2.0,
        help="Keep pair if P(answer|clean) > ratio * P(answer|corrupt) (default: 2)",
    )
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = parser.parse_args()

    meta_in, pairs_in = load_pairs(args.in_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading model {args.model!r} on {device}...")
    model = HookedTransformer.from_pretrained(args.model, device=device)

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for row in tqdm(pairs_in, desc="filter"):
        if not isinstance(row, dict):
            continue
        clean = row.get("clean")
        corrupt = row.get("corrupt")
        ans = row.get("correct_answer")
        if not isinstance(clean, str) or not isinstance(corrupt, str) or not isinstance(ans, str):
            dropped.append({**row, "drop_reason": "missing_field"})
            continue

        try:
            tid = answer_token_id(model, ans)
            p_clean = prob_next_token(model, clean, tid, device=device)
            p_corrupt = prob_next_token(model, corrupt, tid, device=device)
        except Exception as e:
            dropped.append({**row, "drop_reason": f"error:{e}"})
            continue

        ratio_ok = p_clean > float(args.ratio) * p_corrupt
        row_out = {
            **row,
            "p_clean": p_clean,
            "p_corrupt": p_corrupt,
            "prob_ratio": (p_clean / p_corrupt) if p_corrupt > 1e-12 else float("inf"),
        }
        if ratio_ok:
            kept.append(row_out)
        else:
            dropped.append({**row_out, "drop_reason": "prob_ratio"})

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "source_file": str(args.in_path),
        "filter": {"ratio": args.ratio, "metric": "P(first_token(correct_answer)) at last prompt position"},
        "model": args.model,
        "generator_meta": {k: v for k, v in meta_in.items() if k != "pairs"},
        "counts": {"input": len(pairs_in), "kept": len(kept), "dropped": len(dropped)},
        "pairs": kept,
        "dropped_pairs": dropped,
    }
    args.out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Kept {len(kept)}/{len(pairs_in)} pairs -> {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
