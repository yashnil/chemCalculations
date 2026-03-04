#!/usr/bin/env python3
"""
update_plots_for_optimal_retrained.py
======================================

Update comparison metrics and regenerate ALL plots for optimal_retrained runs.
This ensures all plots reflect the new 160, 320, 480, 640, 800 dataset sizes.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "results" / "runs"
COMPARISON_CSV = BASE_DIR / "plots" / "comparison_metrics.csv"

# Optimal retrained runs (the new standard - constant 320K increments)
OPTIMAL_RETRAINED_RUNS = [
    ("x800_optimal_retrained", 800000),
    ("x1600_optimal_retrained", 1600000),
    ("x2400_optimal_retrained", 2400000),
    ("x3200_optimal_retrained", 3200000),
    ("x4000_optimal_retrained", 4000000),
    ("x4800_optimal_retrained", 4800000),
]


def parse_global_metrics(txt_path: Path) -> Dict[str, float]:
    """Parse log_mae and log_r2 from diagnostics/global_metrics.txt"""
    metrics: Dict[str, float] = {}
    if not txt_path.exists():
        return metrics
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, val = [x.strip() for x in line.split(":", 1)]
        if key.startswith("Log MAE"):
            try:
                metrics["log_mae"] = float(val.replace(",", " ").split()[0])
            except:
                pass
        elif key.startswith("Log R"):
            try:
                val_num = val.replace(",", " ").split()[0]
                metrics["log_r2"] = float(val_num)
            except:
                pass
    return metrics


def compute_log_mae_from_loss_history(run_dir: Path) -> Optional[float]:
    """Compute approximate log MAE from final validation log MAE in loss history."""
    loss_history_path = run_dir / "loss_history.csv"
    if not loss_history_path.exists():
        return None
    
    df = pd.read_csv(loss_history_path)
    if "val_log_mae" in df.columns and len(df) > 0:
        final_log_mae = df.iloc[-1]["val_log_mae"]
        if pd.notna(final_log_mae):
            return float(final_log_mae)
    return None


def collect_metrics(run_tag: str, total_samples: int) -> Optional[Dict[str, object]]:
    """Collect metrics from an optimal_retrained run."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    summary_path = run_dir / "summary.json"
    diag_metrics = run_dir / "diagnostics" / "global_metrics.txt"
    
    if not summary_path.exists():
        print(f"  ⚠️  Missing summary.json for {run_tag}")
        return None
    
    s = json.loads(summary_path.read_text())
    gm = parse_global_metrics(diag_metrics)
    
    # Try multiple sources for log_mae
    log_mae = None
    if gm.get("log_mae"):
        log_mae = float(gm["log_mae"])
    elif s.get("test_log_mae"):
        log_mae = float(s["test_log_mae"])
    else:
        log_mae_from_history = compute_log_mae_from_loss_history(run_dir)
        if log_mae_from_history:
            log_mae = log_mae_from_history
    
    # Try multiple sources for log_r2 (priority: summary > diagnostics)
    log_r2 = None
    if s.get("test_log_r2") is not None:
        log_r2 = float(s["test_log_r2"])
    elif gm.get("log_r2"):
        log_r2 = float(gm["log_r2"])
    
    # Extract dataset tag
    size = run_tag.split('_')[0].replace('x', '')
    csv_tag = f"x{size}_optimal_retrained"
    
    row: Dict[str, object] = {
        "dataset": csv_tag,
        "total_samples": int(total_samples),
        "val_loss": float(s.get("val_loss", 0)),
        "test_loss": float(s.get("test_loss", 0)),
        "log_mae": float(log_mae) if log_mae is not None else float("nan"),
        "log_r2": float(log_r2) if log_r2 is not None else float("nan"),
        "linear_mae": float(s.get("test_mae_linear", float("nan"))),
        "linear_mse": float(s.get("test_mse_linear", float("nan"))),
    }
    return row


def read_existing_csv() -> List[Dict[str, object]]:
    """Read existing comparison_metrics.csv"""
    if not COMPARISON_CSV.exists():
        return []
    
    rows = []
    with COMPARISON_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def safe_float(val, default=0.0):
                if not val or val.strip() == "":
                    return default
                try:
                    return float(val)
                except:
                    return default
            
            def safe_int(val, default=0):
                if not val or val.strip() == "":
                    return default
                try:
                    return int(val)
                except:
                    return default
            
            row["total_samples"] = safe_int(row.get("total_samples", 0))
            row["val_loss"] = safe_float(row.get("val_loss", 0))
            row["test_loss"] = safe_float(row.get("test_loss", 0))
            row["log_mae"] = safe_float(row.get("log_mae", 0), float("nan"))
            row["log_r2"] = safe_float(row.get("log_r2", 0), float("nan"))
            row["linear_mae"] = safe_float(row.get("linear_mae", 0), float("nan"))
            row["linear_mse"] = safe_float(row.get("linear_mse", 0), float("nan"))
            rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, object]]):
    """Write rows to comparison_metrics.csv"""
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "dataset",
        "total_samples",
        "val_loss",
        "test_loss",
        "log_mae",
        "log_r2",
        "linear_mae",
        "linear_mse",
    ]
    
    with COMPARISON_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"✅ Updated {COMPARISON_CSV} ({len(rows)} rows)")


def update_comparison_metrics():
    """Update comparison_metrics.csv with optimal_retrained runs."""
    print("="*80)
    print("UPDATING COMPARISON METRICS")
    print("="*80)
    
    existing_rows = read_existing_csv()
    existing_tags = {row["dataset"] for row in existing_rows}
    print(f"Found {len(existing_rows)} existing entries")
    
    # Filter to keep only optimal_retrained runs (remove old inconsistent ones)
    print("\n📊 Filtering to optimal_retrained runs...")
    keep_tags = {f"x{size}_optimal_retrained" for size in [800, 1600, 2400, 3200, 4000, 4800]}
    remove_tags = set()  # Remove everything not in keep_tags
    
    filtered_rows = []
    for row in existing_rows:
        tag = row["dataset"]
        # Remove entries we want to exclude (320, 640, 960, 1280)
        if tag in remove_tags:
            print(f"  🗑️  Removing {tag} (not in constant increment set)")
            continue
        # Remove old entries that conflict with optimal_retrained runs we're keeping
        if tag in keep_tags:
            continue  # Will be replaced
        # Keep entries that don't conflict
        filtered_rows.append(row)
    
    print(f"  Kept {len(filtered_rows)} entries after filtering")
    
    # Collect new optimal_retrained runs
    print("\n📊 Collecting optimal_retrained runs...")
    new_rows = []
    for run_tag, total_samples in OPTIMAL_RETRAINED_RUNS:
        csv_tag = f"{run_tag.split('_')[0]}_optimal_retrained"
        
        metrics = collect_metrics(run_tag, total_samples)
        if metrics:
            new_rows.append(metrics)
            print(f"  ✅ Added {csv_tag}: test_loss={metrics['test_loss']:.6f}, log_mae={metrics.get('log_mae', 'N/A')}")
        else:
            print(f"  ⚠️  Failed to collect metrics for {run_tag}")
    
    # Combine and sort
    all_rows = filtered_rows + new_rows
    all_rows.sort(key=lambda x: (int(x["total_samples"]), x["dataset"]))
    
    # Write updated CSV
    write_csv(all_rows)
    
    # Find best model
    print("\n🏆 Finding best model (optimal_retrained only)...")
    optimal_rows = [r for r in all_rows if r["dataset"].endswith("_optimal_retrained")]
    if optimal_rows:
        valid = [r for r in optimal_rows if r["test_loss"] > 0 and str(r["test_loss"]) != "nan"]
        if valid:
            best = min(valid, key=lambda x: x["test_loss"])
            print(f"  🏆 Best model: {best['dataset']}")
            print(f"    Test loss: {best['test_loss']:.6f}")
            print(f"    Val loss: {best['val_loss']:.6f}")
            print(f"    Log MAE: {best.get('log_mae', 'N/A')}")
            print(f"    Log R²: {best.get('log_r2', 'N/A')}")
            print(f"    Dataset size: {best['total_samples']:,}")
    
    return all_rows


def regenerate_all_plots():
    """Regenerate all plots."""
    print("\n" + "="*80)
    print("REGENERATING ALL PLOTS")
    print("="*80)
    
    plots = [
        ("src/plot_training_analysis.py", "Training analysis plots"),
        ("src/plot_comprehensive_analysis.py", "Comprehensive analysis plots"),
    ]
    
    for script_path, description in plots:
        script = BASE_DIR / script_path
        if script.exists():
            print(f"\n📊 {description}...")
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=BASE_DIR,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"  ✅ Success")
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if '✅' in line or 'Warning' in line:
                            print(f"    {line}")
            else:
                print(f"  ⚠️  Warning: {result.stderr[:300]}")


def main():
    print("="*80)
    print("UPDATE COMPARISON METRICS AND REGENERATE ALL PLOTS")
    print("="*80)
    print("Using optimal_retrained runs:")
    for run_tag, size in OPTIMAL_RETRAINED_RUNS:
        print(f"  - {run_tag} ({size:,} samples)")
    print()
    
    # Step 1: Update comparison metrics
    all_rows = update_comparison_metrics()
    
    # Step 2: Regenerate all plots
    regenerate_all_plots()
    
    print("\n" + "="*80)
    print("✅ ALL PLOTS REGENERATED!")
    print("="*80)
    print("\nKey files updated:")
    print(f"  - {COMPARISON_CSV}")
    print(f"  - plots/performance_vs_size.png")
    print(f"  - plots/loss_curves.png")
    print(f"  - plots/asymptote_analysis.png")
    print(f"  - plots/model_comparison.png")


if __name__ == "__main__":
    main()
