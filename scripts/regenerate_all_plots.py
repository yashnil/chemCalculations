#!/usr/bin/env python3
"""
regenerate_all_plots.py
=======================

Update comparison metrics with new consistent architecture runs and regenerate all plots.
This ensures everything uses the same architecture for fair comparison.
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

# 800K-increment study runs
CONSISTENT_RUNS = [
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
    
    # Extract dataset tag (e.g., "x160_static_32_consistent" -> "x160_consistent")
    size = run_tag.split('_')[0]
    csv_tag = f"{size}_consistent"
    
    row: Dict[str, object] = {
        "dataset": csv_tag,
        "total_samples": int(total_samples),
        "val_loss": float(s.get("val_loss", 0)),
        "test_loss": float(s.get("test_loss", 0)),
        "log_mae": float(gm.get("log_mae")) if gm.get("log_mae") else float("nan"),
        "log_r2": float(gm.get("log_r2")) if gm.get("log_r2") else float("nan"),
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
            # Convert numeric fields, handling empty strings
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
    
    # Read existing CSV
    existing_rows = read_existing_csv()
    existing_tags = {row["dataset"] for row in existing_rows}
    print(f"Found {len(existing_rows)} existing entries")
    
    # Remove old inconsistent entries (keep only consistent ones)
    print("\n📊 Filtering to consistent architecture runs...")
    consistent_tags = {f"{size.split('_')[0]}_consistent" for size, _ in CONSISTENT_RUNS}
    
    # Keep only consistent runs and other non-conflicting entries
    filtered_rows = []
    for row in existing_rows:
        tag = row["dataset"]
        # Keep consistent runs, exclude old inconsistent ones at same sizes
        if tag in consistent_tags:
            filtered_rows.append(row)
        elif not any(tag.startswith(f"x{size}_") for size in [160, 240, 480, 640]):
            # Keep entries that don't conflict
            filtered_rows.append(row)
    
    print(f"  Kept {len(filtered_rows)} entries after filtering")
    
    # Collect new consistent runs
    print("\n📊 Collecting consistent architecture runs...")
    new_rows = []
    for run_tag, total_samples in CONSISTENT_RUNS:
        csv_tag = f"{run_tag.split('_')[0]}_consistent"
        
        if csv_tag in existing_tags:
            print(f"  ⏭️  {csv_tag} already exists, replacing...")
            filtered_rows = [r for r in filtered_rows if r["dataset"] != csv_tag]
        
        metrics = collect_metrics(run_tag, total_samples)
        if metrics:
            new_rows.append(metrics)
            print(f"  ✅ Added {csv_tag}")
        else:
            print(f"  ⚠️  Failed to collect metrics for {run_tag}")
    
    # Combine and sort
    all_rows = filtered_rows + new_rows
    all_rows.sort(key=lambda x: (int(x["total_samples"]), x["dataset"]))
    
    # Write updated CSV
    write_csv(all_rows)
    
    # Find best model
    print("\n🏆 Finding best model...")
    valid_rows = [r for r in all_rows if not (r["test_loss"] == 0 or str(r["test_loss"]) == "nan")]
    if valid_rows:
        best = min(valid_rows, key=lambda x: x["test_loss"])
        print(f"  Best model: {best['dataset']}")
        print(f"    Test loss: {best['test_loss']:.6f}")
        print(f"    Log MAE: {best.get('log_mae', 'N/A')}")
        print(f"    Log R²: {best.get('log_r2', 'N/A')}")
        print(f"    Dataset size: {best['total_samples']:,}")
    
    return all_rows


def regenerate_plots():
    """Regenerate all plots using updated metrics."""
    print("\n" + "="*80)
    print("REGENERATING PLOTS")
    print("="*80)
    
    plots_to_generate = [
        ("src/plot_training_analysis.py", "Training analysis plots"),
        ("src/plot_consistent_runs.py", "Consistent runs plots"),
    ]
    
    for script_path, description in plots_to_generate:
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
            else:
                print(f"  ⚠️  Warning: {result.stderr[:200]}")
        else:
            print(f"  ⚠️  Script not found: {script}")


def main():
    print("="*80)
    print("REGENERATE ALL PLOTS WITH CONSISTENT ARCHITECTURE")
    print("="*80)
    
    # Step 1: Update comparison metrics
    all_rows = update_comparison_metrics()
    
    # Step 2: Regenerate plots
    regenerate_plots()
    
    print("\n" + "="*80)
    print("✅ ALL PLOTS REGENERATED!")
    print("="*80)
    print("\nKey files updated:")
    print(f"  - {COMPARISON_CSV}")
    print(f"  - plots/performance_vs_size.png (using consistent runs)")
    print(f"  - plots/loss_curves_consistent.png")
    print(f"  - plots/performance_vs_size_consistent.png")


if __name__ == "__main__":
    main()
