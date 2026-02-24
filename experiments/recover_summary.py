"""
Recover summary.csv from individual metrics.json files.

Usage:
  python experiments/recover_summary.py <exp_dir>

Use this after a cancelled run to reconstruct the summary CSV
from the per-run metrics.json files that were already saved.
"""

import os
import sys
import csv
import json
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_experiments import SUMMARY_FIELDS, compute_and_save_optimality_ratios


def recover(exp_dir: str) -> None:
    exp_path = Path(exp_dir)
    if not exp_path.is_dir():
        print(f"ERROR: directory not found: {exp_dir}")
        sys.exit(1)

    run_dirs = sorted(
        d for d in exp_path.iterdir()
        if d.is_dir() and (d / "metrics.json").exists()
    )

    if not run_dirs:
        print(f"ERROR: no metrics.json files found under {exp_dir}")
        sys.exit(1)

    rows = []
    skipped = 0
    for run_dir in run_dirs:
        with open(run_dir / "metrics.json") as f:
            m = json.load(f)
        row = {}
        for field in SUMMARY_FIELDS:
            val = m.get(field, "")
            # Lists are excluded from summary CSV — replace with empty string
            if isinstance(val, list):
                val = ""
            row[field] = val
        rows.append(row)

    summary_path = exp_path / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Recovered {len(rows)} runs -> {summary_path}")
    if skipped:
        print(f"  ({skipped} directories skipped — no metrics.json)")

    # Also compute optimality ratios
    compute_and_save_optimality_ratios(str(exp_path))


def main():
    if len(sys.argv) < 2:
        # Try to find the most recent experiment directory
        base = Path(PROJECT_ROOT) / "results" / "experiments"
        if base.is_dir():
            candidates = sorted(
                d for d in base.iterdir()
                if d.is_dir() and not (d / "summary.csv").exists()
            )
            if candidates:
                exp_dir = str(candidates[-1])
                print(f"No path given — using most recent incomplete run: {exp_dir}")
            else:
                print("Usage: python recover_summary.py <exp_dir>")
                sys.exit(1)
        else:
            print("Usage: python recover_summary.py <exp_dir>")
            sys.exit(1)
    else:
        exp_dir = sys.argv[1]

    recover(exp_dir)


if __name__ == "__main__":
    main()
