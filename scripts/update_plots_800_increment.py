#!/usr/bin/env python3
"""
update_plots_800_increment.py
==============================

Update comparison metrics and regenerate plots for the 800K-increment study:
800, 1600, 2400, 3200, 4000, 4800
"""

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "results" / "runs"
COMPARISON_CSV = BASE_DIR / "plots" / "comparison_metrics.csv"

STUDY_RUNS = [
    ("x800_optimal_retrained", 800000),
    ("x1600_optimal_retrained", 1600000),
    ("x2400_optimal_retrained", 2400000),
    ("x3200_optimal_retrained", 3200000),
    ("x4000_optimal_retrained", 4000000),
    ("x4800_optimal_retrained", 4800000),
]


def collect_metrics(run_tag: str, total_samples: int) -> Optional[Dict]:
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    summary_path = run_dir / "summary.json"

    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    row = {
        "dataset": run_tag,
        "total_samples": total_samples,
        "val_loss": summary.get("val_loss", 0),
        "test_loss": summary.get("test_loss", 0),
        "log_mae": summary.get("test_log_mae", summary.get("log_mae", "")),
        "log_r2": summary.get("test_log_r2", summary.get("log_r2", "")),
        "linear_mae": summary.get("test_mae_linear", ""),
        "linear_mse": summary.get("test_mse_linear", ""),
    }
    return row


def main():
    print("=" * 80)
    print("UPDATING METRICS FOR 800K-INCREMENT STUDY")
    print("=" * 80)

    rows = []
    for run_tag, total_samples in STUDY_RUNS:
        metrics = collect_metrics(run_tag, total_samples)
        if metrics:
            rows.append(metrics)
            tl = metrics["test_loss"]
            lm = metrics.get("log_mae", "N/A")
            print(f"  {run_tag}: test_loss={tl:.6f}, log_mae={lm}")
        else:
            print(f"  {run_tag}: NOT FOUND (skipping)")

    if not rows:
        print("No metrics found!")
        return

    fieldnames = ["dataset", "total_samples", "val_loss", "test_loss",
                  "log_mae", "log_r2", "linear_mae", "linear_mse"]
    with open(COMPARISON_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows to {COMPARISON_CSV}")

    best = min(rows, key=lambda x: x["test_loss"])
    print(f"\nBest model: {best['dataset']} (test_loss={best['test_loss']:.6f})")

    # Regenerate plots
    print("\nRegenerating plots...")
    for script in [
        "src/plot_comprehensive_analysis.py",
        "src/plot_full_suite.py",
    ]:
        path = BASE_DIR / script
        if path.exists():
            print(f"  Running {script}...")
            result = subprocess.run(
                [sys.executable, str(path)], cwd=BASE_DIR,
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"    OK")
            else:
                print(f"    WARN: {result.stderr[:200]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
