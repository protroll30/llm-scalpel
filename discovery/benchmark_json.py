"""Load factual-recall benchmark JSON for discovery / scripts (``pairs`` schema)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_benchmark_pairs(path: Path | str) -> list[dict[str, Any]]:
    """Load ``pairs`` from a benchmark JSON file (e.g. ``factual_recall_*_enriched.json``)."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError(f"{p}: expected top-level 'pairs' array")
    out: list[dict[str, Any]] = []
    for x in pairs:
        if isinstance(x, dict):
            out.append(x)
    return out


def select_benchmark_row(
    pairs: list[dict[str, Any]],
    *,
    pair_index: int = 0,
    pair_id: int | None = None,
) -> dict[str, Any]:
    """Pick one row by ``id`` or by position in ``pairs``."""
    if pair_id is not None:
        for row in pairs:
            if row.get("id") == pair_id:
                return row
        raise KeyError(f"No pair with id={pair_id!r} ({len(pairs)} pairs in file)")
    if pair_index < 0 or pair_index >= len(pairs):
        raise IndexError(f"benchmark pair_index={pair_index} out of range for len={len(pairs)}")
    return pairs[pair_index]


def row_to_dual_prompts_and_answers(
    row: dict[str, Any],
    *,
    fallback_corrupt_answer: str | None = None,
) -> tuple[str, str, str, str]:
    """Return ``clean_prompt, corrupt_prompt, clean_answer, corrupt_answer`` strings."""
    for key in ("clean", "corrupt", "correct_answer"):
        v = row.get(key)
        if not isinstance(v, str) or not v.strip():
            raise KeyError(f"benchmark row missing non-empty string field {key!r}")
    corrupt_a = row.get("corrupt_answer")
    if not isinstance(corrupt_a, str) or not corrupt_a.strip():
        if fallback_corrupt_answer is not None and str(fallback_corrupt_answer).strip():
            corrupt_a = str(fallback_corrupt_answer)
        else:
            raise ValueError(
                "benchmark row has no 'corrupt_answer'; enrich the JSON "
                "(benchmarks/enrich_benchmark_pairs.py) or pass --corrupt-answer on the CLI."
            )
    return row["clean"], row["corrupt"], row["correct_answer"], corrupt_a


def add_discovery_benchmark_cli_args(parser: Any) -> None:
    """Register optional ``--benchmark-json`` / index / id on an ``ArgumentParser``."""
    parser.add_argument(
        "--benchmark-json",
        type=str,
        default="",
        help="Path to benchmark JSON with a top-level 'pairs' array (clean, corrupt, correct_answer, corrupt_answer).",
    )
    parser.add_argument(
        "--benchmark-index",
        type=int,
        default=0,
        help="0-based index into 'pairs' when using --benchmark-json (default: 0).",
    )
    parser.add_argument(
        "--benchmark-id",
        type=int,
        default=None,
        metavar="ID",
        help="Select pair by its 'id' field (overrides --benchmark-index).",
    )


def apply_benchmark_dual_prompts(args: Any) -> None:
    """If ``args.benchmark_json`` is set, overwrite clean/corrupt prompt and answer strings on ``args``."""
    path = str(getattr(args, "benchmark_json", "") or "").strip()
    if not path:
        return
    pairs = load_benchmark_pairs(path)
    row = select_benchmark_row(
        pairs,
        pair_index=int(getattr(args, "benchmark_index", 0)),
        pair_id=getattr(args, "benchmark_id", None),
    )
    fb = getattr(args, "corrupt_answer", None)
    c, u, ca, ua = row_to_dual_prompts_and_answers(row, fallback_corrupt_answer=fb if isinstance(fb, str) else None)
    args.clean_prompt = c
    args.corrupt_prompt = u
    args.clean_answer = ca
    args.corrupt_answer = ua


def prompt_field_from_row(row: dict[str, Any], field: str) -> str:
    """Single prompt for attention viz etc.: ``field`` is 'clean' or 'corrupt'."""
    if field not in ("clean", "corrupt"):
        raise ValueError(f"prompt field must be 'clean' or 'corrupt', got {field!r}")
    v = row.get(field)
    if not isinstance(v, str) or not v.strip():
        raise KeyError(f"benchmark row missing non-empty {field!r}")
    return v


def seq_pos_from_benchmark_row(row: dict[str, Any]) -> int | None:
    """Optional per-row token index for hooks / metrics (negative indices allowed).

    Recognized keys (first match wins): ``metric_seq_pos``, ``loss_seq_pos``, ``seq_pos``, ``subject_pos``.
    """
    for key in ("metric_seq_pos", "loss_seq_pos", "seq_pos", "subject_pos"):
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


def inject_features_from_benchmark_row(row: dict[str, Any]) -> list[int] | None:
    """Optional per-row Layer-8 SAE feature ids (inject path)."""
    for key in ("inject_features", "src_feature_ids", "layer8_feature_ids"):
        v = row.get(key)
        if isinstance(v, list) and len(v) > 0:
            return [int(x) for x in v]
    return None


def dst_features_from_correct_answer_id(row: dict[str, Any]) -> list[int] | None:
    """Optional per-row destination SAE feature ids from ``correct_answer_id`` (int or list of int)."""
    if "correct_answer_id" not in row:
        return None
    v = row["correct_answer_id"]
    if v is None:
        return None
    if isinstance(v, list):
        out: list[int] = []
        for x in v:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                return None
        return out if out else None
    try:
        return [int(v)]
    except (TypeError, ValueError):
        return None


def apply_benchmark_single_prompt(args: Any, *, prompt_attr: str = "prompt", field: str = "corrupt") -> None:
    """If ``args.benchmark_json`` is set, set ``prompt_attr`` from the chosen row."""
    path = str(getattr(args, "benchmark_json", "") or "").strip()
    if not path:
        return
    pairs = load_benchmark_pairs(path)
    row = select_benchmark_row(
        pairs,
        pair_index=int(getattr(args, "benchmark_index", 0)),
        pair_id=getattr(args, "benchmark_id", None),
    )
    f = str(getattr(args, "benchmark_prompt_field", field) or field).strip().lower()
    setattr(args, prompt_attr, prompt_field_from_row(row, f))
