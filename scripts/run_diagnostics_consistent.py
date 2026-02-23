#!/usr/bin/env python3
"""
run_diagnostics_consistent.py
=============================

Run diagnostics on all consistent architecture runs to generate log_mae and log_r2 metrics.
"""

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "results" / "runs"
SRC_DIR = BASE_DIR / "src"

CONSISTENT_RUNS = [
    ("x160_static_32_consistent", 160000),
    ("x240_static_32_consistent", 240000),
    ("x480_static_32_consistent", 480000),
    ("x640_static_32_consistent", 640000),
]


def run_diagnostics(run_tag: str, dataset_size: int):
    """Run diagnostics on a consistent run."""
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
    
    # Get CSV path from summary or infer from run tag
    summary = json.load(open(summary_path))
    # Extract size from run tag (e.g., "x160_static_32_consistent" -> 160)
    size = run_tag.split('_')[0].replace('x', '')
    csv_path = BASE_DIR / "data" / "datasets" / f"all_gas_fastchem_x{size}.csv"
    
    if not csv_path.exists():
        print(f"  ⚠️  Dataset not found: {csv_path}")
        return False
    
    print(f"  Running diagnostics for {run_tag}...")
    print(f"    Model: {best_model_path}")
    print(f"    Dataset: {csv_path}")
    print(f"    Output: {diag_dir}")
    
    diag_env = {
        "CSV_PATH": str(csv_path),
        "BEST_MODULE": str(best_model_path),
        "OUT_DIR": str(diag_dir),
    }
    
    result = subprocess.run(
        [sys.executable, str(SRC_DIR / "diagnostics.py")],
        env=diag_env,
        cwd=SRC_DIR,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"  ✅ Diagnostics completed")
        return True
    else:
        print(f"  ⚠️  Diagnostics failed: {result.stderr[:200]}")
        return False


def main():
    print("="*80)
    print("RUNNING DIAGNOSTICS ON CONSISTENT ARCHITECTURE RUNS")
    print("="*80)
    
    for run_tag, dataset_size in CONSISTENT_RUNS:
        print(f"\n{run_tag}:")
        run_diagnostics(run_tag, dataset_size)
    
    print("\n" + "="*80)
    print("✅ Diagnostics complete!")
    print("="*80)


if __name__ == "__main__":
    main()
