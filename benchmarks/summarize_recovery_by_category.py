"""
Aggregate ``recovery`` from ``intervene_layer8_to_layer9.py`` ``--results-json`` by benchmark ``category``.

Example::

  python benchmarks/summarize_recovery_by_category.py --in results/benchmarks/factual_recall_results.json

With CSV::

  python benchmarks/summarize_recovery_by_category.py --in results/benchmarks/factual_recall_results.json --out-csv results/benchmarks/recovery_by_category.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    p = argparse.ArgumentParser(description="Mean/std of recovery metric by category from results JSON.")
    p.add_argument("--in", "-i", dest="in_path", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=None, help="Optional CSV path for the summary table.")
    args = p.parse_args()

    data = json.loads(args.in_path.read_text(encoding="utf-8"))
    rows = data.get("rows")
    if not isinstance(rows, list):
        print("Input JSON must contain 'rows' array.", file=sys.stderr)
        return 1

    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("error"):
            continue
        cat = r.get("category")
        if not isinstance(cat, str) or not cat.strip():
            cat = "unknown"
        rv = r.get("recovery")
        if rv is None:
            continue
        try:
            x = float(rv)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x):
            continue
        by_cat[cat].append(x)

    lines = []
    lines.append(f"{'category':<14} {'n':>5} {'mean':>10} {'std':>10}")
    lines.append("-" * 44)
    summary_rows: list[dict[str, Any]] = []

    for cat in sorted(by_cat.keys()):
        vals = by_cat[cat]
        n = len(vals)
        mean_v = statistics.mean(vals)
        std_v = statistics.stdev(vals) if n > 1 else 0.0
        lines.append(f"{cat:<14} {n:5d} {mean_v:10.4f} {std_v:10.4f}")
        summary_rows.append({"category": cat, "n": n, "mean_recovery": mean_v, "std_recovery": std_v})

    print("\n".join(lines))

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["category", "n", "mean_recovery", "std_recovery"])
            w.writeheader()
            w.writerows(summary_rows)
        print(f"wrote {args.out_csv.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
