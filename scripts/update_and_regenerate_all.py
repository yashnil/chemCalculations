#!/usr/bin/env python3
"""
update_and_regenerate_all.py
=============================

Update comparison metrics and regenerate ALL plots with consistent architecture runs.
This is the master script to refresh everything.
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

# Consistent architecture runs (these are the new standard)
CONSISTENT_RUNS = [
    ("x160_static_32_consistent", 160000),
    ("x240_static_32_consistent", 240000),
    ("x480_static_32_consistent", 480000),
    ("x640_static_32_consistent", 640000),
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
        # Use final epoch's validation log MAE
        final_log_mae = df.iloc[-1]["val_log_mae"]
        if pd.notna(final_log_mae):
            return float(final_log_mae)
    return None


def collect_metrics(run_tag: str, total_samples: int) -> Optional[Dict[str, object]]:
    """Collect metrics from a consistent architecture run."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    summary_path = run_dir / "summary.json"
    diag_metrics = run_dir / "diagnostics" / "global_metrics.txt"
    
    if not summary_path.exists():
        print(f"  ⚠️  Missing summary.json for {run_tag}")
        return None
    
    s = json.loads(summary_path.read_text())
    gm = parse_global_metrics(diag_metrics)
    
    # Try multiple sources for log_mae (priority: diagnostics > summary > loss_history)
    log_mae = None
    if gm.get("log_mae"):
        log_mae = float(gm["log_mae"])
    elif s.get("test_log_mae"):
        log_mae = float(s["test_log_mae"])
    else:
        log_mae_from_history = compute_log_mae_from_loss_history(run_dir)
        if log_mae_from_history:
            log_mae = log_mae_from_history
    
    # Try multiple sources for log_r2 (priority: diagnostics > summary)
    log_r2 = None
    if gm.get("log_r2"):
        log_r2 = float(gm["log_r2"])
    elif s.get("test_log_r2"):
        log_r2 = float(s["test_log_r2"])
    
    # Extract dataset tag
    size = run_tag.split('_')[0]
    csv_tag = f"{size}_consistent"
    
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
    """Update comparison_metrics.csv with consistent architecture runs."""
    print("="*80)
    print("UPDATING COMPARISON METRICS")
    print("="*80)
    
    existing_rows = read_existing_csv()
    existing_tags = {row["dataset"] for row in existing_rows}
    print(f"Found {len(existing_rows)} existing entries")
    
    # Remove old inconsistent entries at same sizes
    print("\n📊 Filtering to consistent architecture runs...")
    consistent_tags = {f"{size.split('_')[0]}_consistent" for size, _ in CONSISTENT_RUNS}
    
    filtered_rows = []
    for row in existing_rows:
        tag = row["dataset"]
        # Remove old entries that conflict with consistent runs
        if tag in consistent_tags:
            continue  # Will be replaced
        # Keep entries that don't conflict
        filtered_rows.append(row)
    
    print(f"  Kept {len(filtered_rows)} entries after filtering")
    
    # Collect new consistent runs (preserve existing log_mae/log_r2 if they exist)
    print("\n📊 Collecting consistent architecture runs...")
    new_rows = []
    for run_tag, total_samples in CONSISTENT_RUNS:
        csv_tag = f"{run_tag.split('_')[0]}_consistent"
        
        # Check if we already have this row with metrics
        existing_row = None
        for row in filtered_rows:
            if row["dataset"] == csv_tag:
                existing_row = row
                break
        
        metrics = collect_metrics(run_tag, total_samples)
        if metrics:
            # Preserve existing log_mae/log_r2 if they exist and are not NaN
            if existing_row:
                if pd.notna(existing_row.get("log_mae")) and existing_row["log_mae"] != 0:
                    metrics["log_mae"] = existing_row["log_mae"]
                if pd.notna(existing_row.get("log_r2")) and existing_row["log_r2"] != 0:
                    metrics["log_r2"] = existing_row["log_r2"]
            
            new_rows.append(metrics)
            print(f"  ✅ Added {csv_tag}: test_loss={metrics['test_loss']:.6f}, log_mae={metrics.get('log_mae', 'N/A')}")
        else:
            print(f"  ⚠️  Failed to collect metrics for {run_tag}")
    
    # Combine and sort
    all_rows = filtered_rows + new_rows
    all_rows.sort(key=lambda x: (int(x["total_samples"]), x["dataset"]))
    
    # Write updated CSV
    write_csv(all_rows)
    
    # Find best model (from consistent runs)
    print("\n🏆 Finding best model (consistent architecture only)...")
    consistent_rows = [r for r in all_rows if r["dataset"].endswith("_consistent")]
    if consistent_rows:
        valid = [r for r in consistent_rows if r["test_loss"] > 0 and str(r["test_loss"]) != "nan"]
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
        ("src/plot_consistent_runs.py", "Consistent runs plots"),
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
    print("Using consistent architecture runs:")
    for run_tag, size in CONSISTENT_RUNS:
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
    print(f"  - plots/loss_curves_consistent.png")
    print(f"  - plots/performance_vs_size_consistent.png")


if __name__ == "__main__":
    main()
