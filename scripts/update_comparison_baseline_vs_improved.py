#!/usr/bin/env python3
"""
update_comparison_baseline_vs_improved.py
=========================================

Build comparison_metrics.csv with BOTH:
  - x4800_optimal_retrained (baseline / previous best)
  - x4800_improved (AdamW, train-only norm, correct loss naming)

Run after:
  1. Train improved: python src/train_autoencoder_improved.py --config configs/x4800_improved.json --run-dir results/runs/runs_autoencoder_x4800_improved
  2. Run diagnostics on both (run_diagnostics_all_optimal_retrained.py or manually for x4800_improved)

Then run this script to update plots/comparison_metrics.csv and regenerate plots.
"""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "results" / "runs"
COMPARISON_CSV = BASE_DIR / "plots" / "comparison_metrics.csv"
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from mfae_metrics import compute_mfae_for_run  # noqa: E402

# Baseline (previous best) + improved (with 4 changes)
COMPARISON_RUNS = [
    ("x800_optimal_retrained", 800000),
    ("x1600_optimal_retrained", 1600000),
    ("x2400_optimal_retrained", 2400000),
    ("x3200_optimal_retrained", 3200000),
    ("x4000_optimal_retrained", 4000000),
    ("x4800_optimal_retrained", 4800000),
    ("x4800_improved", 4800000),
    ("x4800_mlp", 4800000),
]


def parse_global_metrics(txt_path: Path) -> Dict[str, float]:
    metrics = {}
    if not txt_path.exists():
        return metrics
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, val = [x.strip() for x in line.split(":", 1)]
        if key.startswith("Log MAE"):
            metrics["log_mae"] = float(val.replace(",", " ").split()[0])
        elif key.startswith("Log R"):
            val_num = val.replace(",", " ").split()[0]
            metrics["log_r2"] = float(val_num)
        elif key == "AAFE":
            metrics["aafe"] = float(val.replace(",", " ").split()[0])
        elif key == "MFAE":
            metrics["mfae"] = float(val.replace(",", " ").split()[0])
    return metrics


def collect_metrics(run_tag: str, total_samples: int) -> Optional[Dict]:
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    summary_path = run_dir / "summary.json"
    diag_metrics = run_dir / "diagnostics" / "global_metrics.txt"

    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        summary = json.load(f)
    gm = parse_global_metrics(diag_metrics)

    row = {
        "dataset": run_tag,
        "total_samples": total_samples,
        "val_loss": summary.get("val_loss", 0),
        "test_loss": summary.get("test_loss", 0),
        "log_mae": gm.get("log_mae", summary.get("test_log_mae", summary.get("log_mae", ""))),
        "log_r2": gm.get("log_r2", summary.get("test_log_r2", summary.get("log_r2", ""))),
        "mfae": gm.get("mfae", ""),
        "linear_mae": summary.get("test_mae_linear", ""),
        "linear_mse": summary.get("test_loss_linear", summary.get("test_mse_linear", "")),
    }
    return row


def run_diagnostics(run_tag: str) -> bool:
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    best_model_path = run_dir / "best_model.py"
    csv_path = BASE_DIR / "data" / "datasets" / f"all_gas_fastchem_x4800.csv"
    diag_dir = run_dir / "diagnostics"

    if not best_model_path.exists():
        print(f"  ⚠️  best_model.py not found for {run_tag}")
        return False
    if not csv_path.exists():
        print(f"  ⚠️  Dataset not found: {csv_path}")
        return False

    import os
    env = os.environ.copy()
    env["CSV_PATH"] = str(csv_path)
    env["BEST_MODULE"] = str(best_model_path)
    env["OUT_DIR"] = str(diag_dir)

    result = subprocess.run(
        [sys.executable, str(SRC_DIR / "diagnostics.py")],
        env=env,
        cwd=str(SRC_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and (diag_dir / "global_metrics.txt").exists():
        print(f"  ✅ Diagnostics: {run_tag}")
        return True
    print(f"  ⚠️  Diagnostics failed for {run_tag}: {result.stderr[:200]}")
    return False


def main():
    print("=" * 80)
    print("BASELINE vs IMPROVED COMPARISON")
    print("=" * 80)

    # Run diagnostics on x4800_improved (best model) and copy scatter/parity plots to plots/
    improved_dir = RUNS_DIR / "runs_autoencoder_x4800_improved"
    plots_dir = BASE_DIR / "plots"
    if improved_dir.exists() and (improved_dir / "best_model.py").exists():
        print("\nRunning diagnostics on x4800_improved (best model)...")
        run_diagnostics("x4800_improved")
        # Copy scatter/parity plots from x4800_improved diagnostics to plots/ so they represent the best model
        diag_dir = improved_dir / "diagnostics"
        for name in ["parity_overall.png", "parity_top10.png", "residual_vs_observed.png", "MAE_per_species.png", "error_distribution.png", "global_metrics.txt", "diagnostic_summary.txt"]:
            src = diag_dir / name
            if src.exists():
                dst = plots_dir / name
                shutil.copy2(src, dst)
                print(f"  Copied {name} to plots/")

    print("\nCollecting metrics...")
    rows = []
    for run_tag, total_samples in COMPARISON_RUNS:
        metrics = collect_metrics(run_tag, total_samples)
        if metrics:
            rows.append(metrics)
            lm = metrics.get("log_mae", "N/A")
            tl = metrics.get("test_loss", "N/A")
            print(f"  {run_tag}: test_loss={tl}, log_mae={lm}")
        else:
            print(f"  {run_tag}: NOT FOUND (skipping)")

    if not rows:
        print("No metrics found!")
        return

    # MFAE: winsorized mean fractional error (see src/mfae_metrics.py); fill if missing from diagnostics
    print("\nComputing MFAE (winsor mean |pred-true|/true over scatter dots)...")
    for row in rows:
        tag = row["dataset"]
        need = True
        if row.get("mfae") not in ("", None):
            try:
                float(row["mfae"])
                need = False
            except (TypeError, ValueError):
                pass
        if not need:
            continue
        m = compute_mfae_for_run(tag)
        if m is not None:
            row["mfae"] = m
            print(f"  {tag}: MFAE={m:.6f}")
        else:
            print(f"  {tag}: MFAE= (could not compute)")

    fieldnames = ["dataset", "total_samples", "val_loss", "test_loss", "log_mae", "log_r2", "mfae", "linear_mae", "linear_mse"]
    with open(COMPARISON_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows to {COMPARISON_CSV}")

    # Compare baseline vs improved if both present
    baseline = next((r for r in rows if r["dataset"] == "x4800_optimal_retrained"), None)
    improved = next((r for r in rows if r["dataset"] == "x4800_improved"), None)
    if baseline and improved:
        print("\n--- x4800_optimal_retrained vs x4800_improved ---")
        print(f"  Baseline: test_loss={baseline['test_loss']:.6f}, log_mae={baseline.get('log_mae', 'N/A')}")
        print(f"  Improved: test_loss={improved['test_loss']:.6f}, log_mae={improved.get('log_mae', 'N/A')}")
        if improved["test_loss"] < baseline["test_loss"]:
            print("  → Improved has lower test loss")
        else:
            print("  → Baseline has lower test loss")

    print("\nRegenerating plots...")
    for script in ["src/plot_comprehensive_analysis.py", "src/plot_full_suite.py"]:
        path = BASE_DIR / script
        if path.exists():
            subprocess.run([sys.executable, str(path)], cwd=BASE_DIR, capture_output=True, text=True)
    print("Done!")


if __name__ == "__main__":
    main()
