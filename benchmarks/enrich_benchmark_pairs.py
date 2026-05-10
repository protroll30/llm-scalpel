"""
Repair an existing raw benchmark JSON without re-running full DeepSeek generation.

1. **Leading-space alignment (instant, no API):** GPT-2 often expects the next token after
   ``"... is"`` (no trailing space) to be a **word token with a leading space**
   (e.g. ``" Paris"``). If the prompt already ends with whitespace, the completion is
   usually **without** an extra leading space on the stored answer string.

2. **corrupt_answer (batched API):** Fills the contrastive target for
   ``Logit(clean_answer) - Logit(corrupt_answer)``. Uses **short batched** DeepSeek
   calls (default ``deepseek-chat``) instead of regenerating all pairs.

Examples::

  # Spacing only — seconds, no key required for corrupt fill
  python benchmarks/enrich_benchmark_pairs.py \\
    --in benchmarks/raw/factual_recall_250.json \\
    --out benchmarks/raw/factual_recall_250_enriched.json \\
    --spacing-only

  # Add corrupt_answer via API (~ ceil(N/batch_size) requests vs hundreds for generation)
  python benchmarks/enrich_benchmark_pairs.py \\
    --in benchmarks/raw/factual_recall_250.json \\
    --out benchmarks/raw/factual_recall_250_enriched.json \\
    --batch-size 30

Requires ``DEEPSEEK_API_KEY`` in env or repo-root ``.env`` when not using ``--spacing-only``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_BENCHMARK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _BENCHMARK_ROOT.parent


def normalize_answer_after_prompt(prompt: str, answer: str) -> str:
    """Align stored answer string with GPT-2 continuation after ``prompt`` (BPE / leading space).

    If ``prompt`` already ends with whitespace, the next subword is usually written **without**
    an extra leading space in the answer field. If ``prompt`` does **not** end with whitespace,
    standalone-word continuations are typically stored **with** a leading space (e.g. ``\" Paris\"``).
    """
    if not isinstance(answer, str):
        return answer
    core = answer.strip()
    if not core:
        return answer
    # Single punctuation / non-alphanumeric single-char continuation
    if len(core) == 1 and not core.isalnum():
        return core
    if answer.startswith((" ", "\n", "\t")):
        return answer
    if prompt and prompt[-1].isspace():
        return core
    return " " + core


def build_corrupt_batch_prompt(corrupt_prompts: list[str]) -> str:
    lines = [
        f"Return only a JSON array of exactly {len(corrupt_prompts)} strings, in order.",
        "Each string is the short factual completion that fits immediately after the corresponding incomplete prompt ",
        "(same facts as in English encyclopedic knowledge; one word or a short phrase).",
        "Do not include numbering or keys — only the JSON array.",
        "",
    ]
    for i, p in enumerate(corrupt_prompts, start=1):
        lines.append(f"{i}. {p}")
    return "\n".join(lines)


def main() -> int:
    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv()

    parser = argparse.ArgumentParser(description="Enrich benchmark JSON: spacing + corrupt_answer.")
    parser.add_argument("--input", "--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", dest="out_path", type=Path, required=True)
    parser.add_argument(
        "--spacing-only",
        action="store_true",
        help="Only normalize correct_answer spacing (no corrupt_answer API calls)",
    )
    parser.add_argument("--batch-size", type=int, default=30, help="Corrupt prompts per API call")
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="DeepSeek model for corrupt_answer fill (default: deepseek-chat; faster/cheaper than reasoner)",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    data = json.loads(args.in_path.read_text(encoding="utf-8"))
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        print("Input JSON must contain a 'pairs' array.", file=sys.stderr)
        return 1

    # --- 1) Normalized correct_answer (always)
    for row in pairs:
        if not isinstance(row, dict):
            continue
        clean = row.get("clean")
        ca = row.get("correct_answer")
        if isinstance(clean, str) and isinstance(ca, str):
            row["correct_answer"] = normalize_answer_after_prompt(clean, ca)

    if args.spacing_only:
        data["enrichment"] = {"spacing": "correct_answer normalized", "corrupt_answer": "skipped"}
        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        args.out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote spacing-only enrich -> {args.out_path}")
        return 0

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("Missing DEEPSEEK_API_KEY (or use --spacing-only).", file=sys.stderr)
        return 1

    # Late import so spacing-only works without pulling urllib stack for users who care
    from benchmarks.generate_benchmark_deepseek import chat_completion, extract_json_array

    # --- 2) corrupt_answer in batches (few calls vs full regeneration)
    n = len(pairs)
    corrupt_out: list[str | None] = [None] * n

    batch_size = max(1, args.batch_size)
    i = 0
    while i < n:
        chunk_end = min(i + batch_size, n)
        chunk_pairs = pairs[i:chunk_end]
        corrupt_prompts = [
            str(p["corrupt"]) if isinstance(p.get("corrupt"), str) else "" for p in chunk_pairs
        ]
        user_msg = build_corrupt_batch_prompt(corrupt_prompts)
        messages = [
            {
                "role": "system",
                "content": (
                    "You complete factual prompts with minimal strings for benchmark construction. "
                    "Output only valid JSON: one array of strings, same length as requested."
                ),
            },
            {"role": "user", "content": user_msg},
        ]
        try:
            content = chat_completion(messages, api_key, args.model, args.max_tokens, args.timeout)
            arr = extract_json_array(content)
        except Exception as e:
            print(f"API/parse error batch [{i}:{chunk_end}]: {e}", file=sys.stderr)
            if chunk_end - i <= 1:
                print(f"Failed on single row {i}; abort.", file=sys.stderr)
                return 1
            batch_size = max(1, batch_size // 2)
            print(f"Retrying with batch_size={batch_size}...", file=sys.stderr)
            time.sleep(2.0)
            continue

        if not isinstance(arr, list) or len(arr) != len(corrupt_prompts):
            got = len(arr) if isinstance(arr, list) else None
            print(
                f"Expected {len(corrupt_prompts)} strings, got {got}; retrying batch [{i}:{chunk_end}]...",
                file=sys.stderr,
            )
            if chunk_end - i <= 1:
                print(f"Length mismatch on single row {i}; abort.", file=sys.stderr)
                return 1
            batch_size = max(1, batch_size // 2)
            time.sleep(2.0)
            continue

        for j, ans in enumerate(arr):
            corrupt_out[i + j] = ans if isinstance(ans, str) else str(ans)

        i = chunk_end
        batch_size = args.batch_size  # restore throughput after a good batch
        time.sleep(0.5)

    for idx, row in enumerate(pairs):
        if not isinstance(row, dict):
            continue
        raw_ca = corrupt_out[idx]
        if raw_ca is None:
            row["corrupt_answer"] = None
            continue
        corrupt_prompt = row.get("corrupt")
        row["corrupt_answer"] = (
            normalize_answer_after_prompt(str(corrupt_prompt), raw_ca)
            if isinstance(corrupt_prompt, str)
            else raw_ca
        )

    data["enrichment"] = {
        "correct_answer": "leading-space heuristic applied",
        "corrupt_answer": f"filled with model={args.model} (batched)",
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote enriched JSON ({n} pairs) -> {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
