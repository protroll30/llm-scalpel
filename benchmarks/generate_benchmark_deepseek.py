"""
Generate factual-recall benchmark pairs via DeepSeek R1 (deepseek-reasoner).

Constraint 1 — token symmetry: each clean/corrupt pair must have identical GPT-2 BPE
lengths. ``deepseek-reasoner`` often emits a reasoning trace before the JSON; we strip
known blocks then parse the first JSON array (see ``strip_reasoning_blocks`` +
``extract_json_array``).

After generation, run ``build_benchmark.py`` for Constraint 2 (probability filter).

Example (from repo root):
  set DEEPSEEK_API_KEY=sk-...
  python benchmarks/generate_benchmark_deepseek.py --total-pairs 250

Then filter:
  python benchmarks/build_benchmark.py --in benchmarks/raw/factual_recall_250.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from transformers import GPT2TokenizerFast

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-reasoner"

RETRY_HTTP_CODES = frozenset({429, 502, 503, 504})
CHAT_MAX_RETRIES = 8

# Thinking / reasoning blocks that may appear before JSON in ``message.content``.
_REASONING_PATTERNS = tuple(
    re.compile(p, re.DOTALL | re.IGNORECASE)
    for p in (
        r"<think>.*?</think>",
        r"<reasoning>.*?</reasoning>",
        r"<thought>.*?</thought>",
    )
)

_BENCHMARK_ROOT = Path(__file__).resolve().parent
_DEFAULT_OUT = _BENCHMARK_ROOT / "raw" / "factual_recall_200.json"

BASE_CATEGORY_WEIGHTS: dict[str, int] = {
    "Geography": 60,
    "Biography": 50,
    "Science": 40,
    "Grammar": 30,
    "Identity": 20,
}


def categories_scaled(total_pairs: int) -> dict[str, int]:
    """Largest-remainder allocation preserving category proportions (base sum = 200)."""
    if total_pairs < 1:
        raise ValueError("--total-pairs must be at least 1")
    base = sum(BASE_CATEGORY_WEIGHTS.values())
    exact = {k: BASE_CATEGORY_WEIGHTS[k] * total_pairs / base for k in BASE_CATEGORY_WEIGHTS}
    floors = {k: int(v) for k, v in exact.items()}
    remainder = total_pairs - sum(floors.values())
    fracs = sorted(
        ((exact[k] - floors[k], k) for k in BASE_CATEGORY_WEIGHTS),
        key=lambda x: (-x[0], x[1]),
    )
    for i in range(remainder):
        floors[fracs[i][1]] += 1
    return floors


def system_instruction(*, strict_parity: bool) -> str:
    core = (
        "You are a dataset generator for mechanistic interpretability. "
        "Generate JSON pairs for GPT-2 Small factual recall. "
        "Each pair must have keys: clean, corrupt, correct_answer (strings). "
        "CRITICAL: After GPT-2 BPE tokenization, len(tokens(clean)) MUST equal "
        "len(tokens(corrupt)). The factual contrast must differ only in content "
        "that preserves total token count. "
        "WHITESPACE: Match spacing exactly between clean and corrupt wherever the template "
        "aligns—GPT-2 tokenizes ' Paris' and 'Paris' differently (leading-space trap). "
        "Use the same leading/trailing spaces around substituted spans in both strings. "
        "Avoid punctuation-only differences that change length. "
        "correct_answer is the continuation token or short span the model should "
        "prefer under the clean prompt (often one token or a short literal). "
        "Output ONLY valid JSON: a single array of objects, no markdown fences, no commentary."
    )
    if strict_parity:
        core += (
            " Prefer simple single-token subject swaps when possible (e.g. replace one GPT-2 "
            "token with another single-token country or name like 'France' vs 'Greece') so "
            "total BPE lengths stay aligned."
        )
    return core


def user_prompt(category: str, count: int, retry_hint: str | None = None) -> str:
    base = (
        f'Generate exactly {count} distinct pairs for category "{category}". '
        "Each object must be {\"clean\": str, \"corrupt\": str, \"correct_answer\": str}. "
        "Return a JSON array only."
    )
    if retry_hint:
        base += " " + retry_hint
    return base


def strip_reasoning_blocks(text: str) -> str:
    """Remove common reasoning / thinking wrappers before JSON (R1 may emit these in ``content``)."""
    t = text
    for pat in _REASONING_PATTERNS:
        t = pat.sub("", t)
    return t.strip()


def chat_completion(
    messages: list[dict[str, str]],
    api_key: str,
    model: str,
    max_tokens: int,
    timeout_s: float,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(CHAT_MAX_RETRIES):
        req = urllib.request.Request(
            DEEPSEEK_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code in RETRY_HTTP_CODES and attempt < CHAT_MAX_RETRIES - 1:
                print(f"HTTP {e.code}, retrying ({attempt + 1}/{CHAT_MAX_RETRIES})...", file=sys.stderr)
                sleep_backoff(attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            if attempt < CHAT_MAX_RETRIES - 1:
                print(f"Network error {e!r}, retrying ({attempt + 1}/{CHAT_MAX_RETRIES})...", file=sys.stderr)
                sleep_backoff(attempt)
                continue
            raise
        obj = json.loads(raw)
        choices = obj.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices in API response: {raw[:2000]}")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not content or not str(content).strip():
            raise RuntimeError(f"Empty message.content in response: {raw[:2000]}")
        return str(content).strip()


def extract_json_array(text: str) -> list[Any]:
    """Decode first JSON array in text (handles leading prose / reasoning before ``[``)."""
    cleaned = strip_reasoning_blocks(text)
    start = cleaned.find("[")
    if start == -1:
        raise ValueError("No '[' found in model output")
    decoder = json.JSONDecoder()
    arr, _ = decoder.raw_decode(cleaned[start:])
    if not isinstance(arr, list):
        raise ValueError("Top-level JSON value is not an array")
    return arr


def normalize_pair(obj: Any, category: str) -> dict[str, str] | None:
    if not isinstance(obj, dict):
        return None
    c = obj.get("clean")
    r = obj.get("corrupt")
    a = obj.get("correct_answer")
    if not isinstance(c, str) or not isinstance(r, str) or not isinstance(a, str):
        return None
    if not c.strip() or not r.strip() or not a.strip():
        return None
    # Preserve interior spacing for GPT-2 parity (do not strip clean/corrupt).
    return {"category": category, "clean": c, "corrupt": r, "correct_answer": a.strip()}


def lengths_match(tok: GPT2TokenizerFast, clean: str, corrupt: str) -> bool:
    return len(tok.encode(clean)) == len(tok.encode(corrupt))


def generate_batch(
    category: str,
    want: int,
    api_key: str,
    model: str,
    max_tokens: int,
    timeout_s: float,
    *,
    strict_parity: bool,
    retry_hint: str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_instruction(strict_parity=strict_parity)},
        {"role": "user", "content": user_prompt(category, want, retry_hint)},
    ]
    content = chat_completion(messages, api_key, model, max_tokens, timeout_s)
    raw_list = extract_json_array(content)
    out: list[dict[str, str]] = []
    for item in raw_list:
        p = normalize_pair(item, category)
        if p is not None:
            out.append(p)
    return out


def sleep_backoff(attempt: int) -> None:
    time.sleep(min(60.0, 2.0**attempt + random.uniform(0, 1)))


def main() -> int:
    load_dotenv(_BENCHMARK_ROOT.parent / ".env")
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate factual recall benchmark JSON via DeepSeek R1.")
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output JSON path (default: {_DEFAULT_OUT})",
    )
    parser.add_argument("--total-pairs", type=int, default=200, help="Total pairs to generate (default: 200; try 250 before probability filtering)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model id (default: deepseek-reasoner)")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max completion tokens per request")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout seconds")
    parser.add_argument("--batch-size", type=int, default=15, help="Pairs requested per API call")
    parser.add_argument("--dry-run", action="store_true", help="Write mock data without calling the API")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    categories = categories_scaled(args.total_pairs)

    if args.dry_run:
        mock_pairs = [
            {
                "category": "Geography",
                "clean": "The capital of France is Paris",
                "corrupt": "The capital of France is Lyon",
                "correct_answer": ".",
            },
            {
                "category": "Science",
                "clean": "Water boils at 100 degrees Celsius",
                "corrupt": "Water boils at 101 degrees Celsius",
                "correct_answer": ".",
            },
        ]
        payload = {
            "schema_version": 1,
            "generator_model": "dry-run",
            "tokenizer": "gpt2",
            "categories": categories,
            "pairs": mock_pairs,
        }
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote dry-run sample to {args.out}")
        return 0

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print(
            "Missing DEEPSEEK_API_KEY. Set it in the environment or .env, then re-run.",
            file=sys.stderr,
        )
        return 1

    all_pairs: list[dict[str, str]] = []
    pair_id = 1
    parity_miss_streak = 0

    for category, total in categories.items():
        have = 0
        attempts = 0
        retry_hint: str | None = None
        while have < total:
            need = min(args.batch_size, total - have)
            attempts += 1
            strict = parity_miss_streak >= 2
            try:
                batch = generate_batch(
                    category,
                    need,
                    api_key,
                    args.model,
                    args.max_tokens,
                    args.timeout,
                    strict_parity=strict,
                    retry_hint=retry_hint,
                )
            except Exception as e:
                print(f"[{category}] API error (attempt {attempts}): {e}", file=sys.stderr)
                if attempts >= 8:
                    raise
                sleep_backoff(attempts)
                continue

            valid: list[dict[str, str]] = []
            invalid = 0
            for p in batch:
                if lengths_match(tok, p["clean"], p["corrupt"]):
                    valid.append(p)
                else:
                    invalid += 1

            n_batch = len(batch)
            if n_batch > 0:
                invalid_ratio = invalid / n_batch
                if invalid_ratio >= 0.5 and invalid > 0:
                    parity_miss_streak += 1
                else:
                    parity_miss_streak = 0

            if not valid:
                retry_hint = (
                    "Previous outputs failed GPT-2 token-length parity: len(encode(clean)) must equal "
                    "len(encode(corrupt)). Match whitespace around substitutions exactly (leading-space trap). "
                    "Prefer single-token-for-single-token swaps."
                )
                print(
                    f"[{category}] No valid pairs in batch (invalid={invalid}); retrying with hint...",
                    file=sys.stderr,
                )
                sleep_backoff(min(attempts, 4))
                continue

            retry_hint = None
            for p in valid:
                if have >= total:
                    break
                row = {"id": pair_id, **p}
                all_pairs.append(row)
                pair_id += 1
                have += 1

            print(f"[{category}] progress {have}/{total} (last batch valid={len(valid)}/{len(batch)})")

            if attempts % 5 == 0:
                partial = {
                    "schema_version": 1,
                    "generator_model": args.model,
                    "tokenizer": "gpt2",
                    "categories": categories,
                    "pairs": all_pairs,
                    "partial": True,
                }
                args.out.write_text(json.dumps(partial, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            time.sleep(1.0)

    payload = {
        "schema_version": 1,
        "generator_model": args.model,
        "tokenizer": "gpt2",
        "categories": categories,
        "pairs": all_pairs,
        "postprocess_note": "Run benchmarks/build_benchmark.py for probability filtering (Constraint 2).",
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(all_pairs)} pairs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
