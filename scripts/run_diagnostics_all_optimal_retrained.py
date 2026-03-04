#!/usr/bin/env python3
"""
run_diagnostics_all_optimal_retrained.py
==========================================

Run diagnostics on all optimal_retrained models to compute log_mae and log_r2.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "results" / "runs"
SRC_DIR = BASE_DIR / "src"

OPTIMAL_RETRAINED_RUNS = [
    ("x800_optimal_retrained", 800000),
    ("x1600_optimal_retrained", 1600000),
    ("x2400_optimal_retrained", 2400000),
    ("x3200_optimal_retrained", 3200000),
    ("x4000_optimal_retrained", 4000000),
    ("x4800_optimal_retrained", 4800000),
]


def run_diagnostics(run_tag: str, dataset_size: int):
    """Run diagnostics on an optimal_retrained run."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    best_model_path = run_dir / "best_model.py"
    summary_path = run_dir / "summary.json"
    diag_dir = run_dir / "diagnostics"
    
    if not best_model_path.exists():
        print(f"  ⚠️  best_model.py not found for {run_tag}")
        return False
    
    if not summary_path.exists():
        print(f"  ⚠️  summary.json not found for {run_tag}")
        return False
    
    # Get CSV path - extract size from run tag
    size = run_tag.split('_')[0].replace('x', '')
    csv_path = BASE_DIR / "data" / "datasets" / f"all_gas_fastchem_x{size}.csv"
    
    if not csv_path.exists():
        print(f"  ⚠️  Dataset not found: {csv_path}")
        return False
    
    print(f"  Running diagnostics for {run_tag}...")
    print(f"    Model: {best_model_path}")
    print(f"    Dataset: {csv_path}")
    print(f"    Output: {diag_dir}")
    
    # Set environment variables for diagnostics script
    env = os.environ.copy()
    env["CSV_PATH"] = str(csv_path)
    env["BEST_MODULE"] = str(best_model_path)
    env["OUT_DIR"] = str(diag_dir)
    
    # Run diagnostics
    result = subprocess.run(
        [sys.executable, str(SRC_DIR / "diagnostics.py")],
        env=env,
        cwd=SRC_DIR,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Check if global_metrics.txt was created
        metrics_file = diag_dir / "global_metrics.txt"
        if metrics_file.exists():
            print(f"  ✅ Diagnostics completed")
            # Print key metrics
            with open(metrics_file) as f:
                for line in f:
                    if "Log MAE" in line or "Log R²" in line or "Log R" in line:
                        print(f"    {line.strip()}")
            return True
        else:
            print(f"  ⚠️  Diagnostics ran but metrics file not found")
            return False
    else:
        print(f"  ⚠️  Diagnostics failed:")
        print(f"    {result.stderr[:500]}")
        return False


def main():
    print("="*80)
    print("RUNNING DIAGNOSTICS ON ALL OPTIMAL_RETRAINED MODELS")
    print("="*80)
    print("This will compute log_mae and log_r2 for all optimal_retrained runs")
    print()
    
    success_count = 0
    for run_tag, dataset_size in OPTIMAL_RETRAINED_RUNS:
        print(f"\n{run_tag}:")
        if run_diagnostics(run_tag, dataset_size):
            success_count += 1
    
    print("\n" + "="*80)
    print(f"✅ Diagnostics complete! ({success_count}/{len(OPTIMAL_RETRAINED_RUNS)} successful)")
    print("="*80)
    print("\nNext step: Update comparison_metrics.csv with new metrics")
    print("Run: python scripts/update_plots_for_optimal_retrained.py")


if __name__ == "__main__":
    main()
