"""
Flatten ``results/benchmarks/factual_recall_results.json`` (from ``intervene_layer8_to_layer9.py``) to CSV.

Example (one line; run from repo root)::

  python benchmarks/export_factual_recall_results_csv.py --in results/benchmarks/factual_recall_results.json --out results/benchmarks/factual_recall_results.csv

Optional: join ``category`` / ``subject_clean`` / ``answer_clean`` from the benchmark file referenced in the JSON
(or pass ``--join-benchmark`` explicitly).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _infer_subject_clean(clean: str) -> str:
    """Best-effort subject span from factual-recall ``clean`` prompts."""
    s = (clean or "").strip()
    if not s:
        return ""

    patterns: list[re.Pattern[str]] = [
        re.compile(r"^The capital of (?:the )?(.+?) is\s*$", re.I),
        re.compile(r"^The founder of (.+?) is\s*$", re.I),
        re.compile(r"^The longest river in (.+?) is\s*$", re.I),
        re.compile(r"^The highest mountain in (.+?) is\s*$", re.I),
        re.compile(r"^The largest city in (.+?) is\s*$", re.I),
        re.compile(r"^The largest country in (.+?) is\s*$", re.I),
        re.compile(r"^The most populous country in (.+?) is\s*$", re.I),
        re.compile(r"^The prime minister of (.+?) is\s*$", re.I),
        re.compile(r"^The president of (.+?) is\s*$", re.I),
        re.compile(r"^(.+?) was born in\s+.+[.!?]?\s*$", re.I),
        re.compile(r"^(.+?) discovered\s+", re.I),
        re.compile(r"^(.+?) wrote\s+", re.I),
        re.compile(r"^The chemical symbol for .+ is\s*$", re.I),
        re.compile(r"^The atomic number of .+ is\s*$", re.I),
    ]
    for rx in patterns:
        m = rx.match(s)
        if m and m.lastindex:
            return m.group(1).strip()
    one = " ".join(s.split())
    return one[:200] if len(one) > 200 else one


def _semi(xs: list[Any]) -> str:
    return ";".join(str(int(x)) for x in xs)


def main() -> int:
    p = argparse.ArgumentParser(description="Convert factual_recall_results.json to CSV.")
    p.add_argument("--in", "-i", dest="in_path", type=Path, required=True)
    p.add_argument("--out", "-o", dest="out_path", type=Path, required=True)
    p.add_argument(
        "--join-benchmark",
        type=Path,
        default=None,
        help="Override benchmark JSON path (default: use 'benchmark_json' from results file if present and exists).",
    )
    args = p.parse_args()

    payload = json.loads(args.in_path.read_text(encoding="utf-8"))
    rows_in = payload.get("rows")
    if not isinstance(rows_in, list):
        print("Input JSON must contain a top-level 'rows' array.", file=sys.stderr)
        return 1

    bench_path = args.join_benchmark
    if bench_path is None:
        rel = payload.get("benchmark_json")
        if isinstance(rel, str) and rel.strip():
            cand = (_REPO / rel).resolve()
            if cand.is_file():
                bench_path = cand
    pairs_by_id: dict[int, dict[str, Any]] = {}
    if bench_path is not None and bench_path.is_file():
        bdata = json.loads(bench_path.read_text(encoding="utf-8"))
        pairs = bdata.get("pairs")
        if isinstance(pairs, list):
            for pr in pairs:
                if isinstance(pr, dict) and pr.get("id") is not None:
                    try:
                        pairs_by_id[int(pr["id"])] = pr
                    except (TypeError, ValueError):
                        pass

    base_cols = [
        "id",
        "benchmark_index",
        "inject_feature_ids",
        "dst_feature_ids",
        "dst_delta_json",
        "seq_pos_raw",
        "seq_pos_resolved",
        "src_sae_id",
        "dst_sae_id",
        "pair_tag",
        "error",
    ]
    join_cols = ["category", "subject_clean", "answer_clean", "clean_prompt", "corrupt_prompt"]
    fieldnames = base_cols[:2] + join_cols + base_cols[2:]

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows_in:
            if not isinstance(row, dict):
                continue
            pid = row.get("id", "")
            inj = row.get("inject_feature_ids")
            dst_ids = row.get("dst_feature_ids")
            ddel = row.get("dst_delta")
            if not isinstance(inj, list):
                inj = []
            if not isinstance(dst_ids, list):
                dst_ids = []
            if not isinstance(ddel, dict):
                ddel = {}

            rec: dict[str, Any] = {
                "id": pid,
                "benchmark_index": row.get("benchmark_index", ""),
                "inject_feature_ids": _semi([int(x) for x in inj]) if inj else "",
                "dst_feature_ids": _semi([int(x) for x in dst_ids]) if dst_ids else "",
                "dst_delta_json": json.dumps(ddel, ensure_ascii=False, separators=(",", ":")),
                "seq_pos_raw": row.get("seq_pos_raw", ""),
                "seq_pos_resolved": row.get("seq_pos_resolved", ""),
                "src_sae_id": row.get("src_sae_id", payload.get("src_sae_id", "")),
                "dst_sae_id": row.get("dst_sae_id", payload.get("dst_sae_id", "")),
                "pair_tag": row.get("pair_tag", ""),
                "error": row.get("error", ""),
            }

            # Join benchmark fields when we have a pair for this id
            try:
                pid_i = int(pid)
            except (TypeError, ValueError):
                pid_i = -1
            pr = pairs_by_id.get(pid_i) if pid_i >= 0 else None
            if pr:
                clean = pr.get("clean", "")
                corrupt = pr.get("corrupt", "")
                ca = pr.get("correct_answer", "")
                rec["category"] = pr.get("category", "") if isinstance(pr.get("category"), str) else ""
                rec["subject_clean"] = _infer_subject_clean(clean if isinstance(clean, str) else "")
                rec["answer_clean"] = ca.strip() if isinstance(ca, str) else ""
                rec["clean_prompt"] = clean if isinstance(clean, str) else ""
                rec["corrupt_prompt"] = corrupt if isinstance(corrupt, str) else ""
            else:
                rec["category"] = ""
                rec["subject_clean"] = ""
                rec["answer_clean"] = ""
                rec["clean_prompt"] = ""
                rec["corrupt_prompt"] = ""

            # Column order: id, benchmark_index, join..., rest
            ordered = {k: rec.get(k, "") for k in fieldnames}
            w.writerow(ordered)
            n += 1

    print(f"wrote {n} row(s) -> {args.out_path.resolve()}")
    if bench_path:
        print(f"joined benchmark: {bench_path}")
    elif pairs_by_id:
        print(f"joined benchmark: (from payload benchmark_json)")
    else:
        print("note: no benchmark join (missing path or file); category/subject/answer columns are empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
