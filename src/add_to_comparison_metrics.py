#!/usr/bin/env python3
"""
add_to_comparison_metrics.py
============================

Add new model results to comparison_metrics.csv.
This script reads results from training runs and appends them to the comparison CSV.

Usage:
    python add_to_comparison_metrics.py --run-dir runs_autoencoder_optimal_x160 --tag x160_optimal
    python add_to_comparison_metrics.py --run-dir runs_autoencoder_latent192 --tag latent192
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
COMPARISON_CSV = BASE_DIR / "plots" / "comparison_metrics.csv"


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


def collect_metrics(run_dir: Path, total_samples: Optional[int] = None) -> Optional[Dict[str, object]]:
    """Collect metrics from a training run directory."""
    summary_path = run_dir / "summary.json"
    diag_metrics = run_dir / "diagnostics" / "global_metrics.txt"
    
    if not summary_path.exists():
        print(f"⚠️  Summary not found: {summary_path}")
        return None
    
    s = json.loads(summary_path.read_text())
    gm = parse_global_metrics(diag_metrics)
    
    # Calculate total samples if not provided
    if total_samples is None:
        total_samples = (
            s.get("train_samples", 0) + 
            s.get("val_samples", 0) + 
            s.get("test_samples", 0)
        )
    
    row: Dict[str, object] = {
        "dataset": "",  # Will be set by caller
        "total_samples": int(total_samples),
        "val_loss": float(s.get("val_loss", 0)),
        "test_loss": float(s.get("test_loss", 0)),
        "log_mae": float(gm.get("log_mae")) if gm.get("log_mae") else float("nan"),
        "log_r2": float(gm.get("log_r2")) if gm.get("log_r2") else float("nan"),
        "linear_mae": float(s.get("test_mae_linear", float("nan"))),
        "linear_mse": float(s.get("test_mse_linear", float("nan"))),
    }
    return row


def read_existing_csv() -> list[Dict[str, object]]:
    """Read existing comparison_metrics.csv"""
    if not COMPARISON_CSV.exists():
        return []
    
    rows = []
    with COMPARISON_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_csv(rows: list[Dict[str, object]]):
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


def main():
    parser = argparse.ArgumentParser(description="Add model results to comparison_metrics.csv")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to training run directory (e.g., runs_autoencoder_optimal_x160)"
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Dataset tag for this run (e.g., x160_optimal, latent192)"
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=None,
        help="Total samples (if not in summary.json)"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR / "models" / "archive",
        help="Base directory where run directories are located"
    )
    args = parser.parse_args()
    
    # Resolve run directory path
    run_dir = args.base_dir / args.run_dir if not args.run_dir.is_absolute() else args.run_dir
    
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        return
    
    # Collect metrics
    print(f"📊 Collecting metrics from {run_dir}")
    metrics = collect_metrics(run_dir, args.total_samples)
    
    if not metrics:
        print("❌ Failed to collect metrics")
        return
    
    metrics["dataset"] = args.tag
    
    # Read existing CSV
    existing_rows = read_existing_csv()
    
    # Check if tag already exists
    existing_tags = {row["dataset"] for row in existing_rows}
    if args.tag in existing_tags:
        print(f"⚠️  Tag '{args.tag}' already exists. Replacing...")
        existing_rows = [r for r in existing_rows if r["dataset"] != args.tag]
    
    # Add new row
    existing_rows.append(metrics)
    
    # Sort by total_samples, then by dataset name
    existing_rows.sort(key=lambda x: (int(x["total_samples"]), x["dataset"]))
    
    # Write updated CSV
    write_csv(existing_rows)
    
    # Print summary
    print(f"\n✅ Added {args.tag}:")
    print(f"   test_loss={metrics['test_loss']:.6f}")
    print(f"   log_mae={metrics.get('log_mae', 'N/A')}")
    print(f"   log_r2={metrics.get('log_r2', 'N/A')}")


if __name__ == "__main__":
    main()

