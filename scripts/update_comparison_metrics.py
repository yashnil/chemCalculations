#!/usr/bin/env python3
"""
update_comparison_metrics.py
=============================

Update comparison_metrics.csv with all available results from:
- Optimal dataset size runs (x32_optimal through x176_optimal)
- Latent dimension study runs (latent64 through latent512)
- Any other runs in models/archive/

Usage:
    python scripts/update_comparison_metrics.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = BASE_DIR / "models" / "archive"
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


def collect_metrics(run_dir: Path, tag: str, total_samples: Optional[int] = None) -> Optional[Dict[str, object]]:
    """Collect metrics from a training run directory."""
    summary_path = run_dir / "summary.json"
    diag_metrics = run_dir / "diagnostics" / "global_metrics.txt"
    
    if not summary_path.exists():
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
        "dataset": tag,
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
            # Convert numeric fields
            row["total_samples"] = int(row["total_samples"])
            row["val_loss"] = float(row["val_loss"])
            row["test_loss"] = float(row["test_loss"])
            try:
                row["log_mae"] = float(row["log_mae"]) if row["log_mae"] else float("nan")
            except:
                row["log_mae"] = float("nan")
            try:
                row["log_r2"] = float(row["log_r2"]) if row["log_r2"] else float("nan")
            except:
                row["log_r2"] = float("nan")
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


def main():
    print("="*80)
    print("UPDATE COMPARISON METRICS")
    print("="*80)
    
    # Read existing CSV
    existing_rows = read_existing_csv()
    existing_tags = {row["dataset"] for row in existing_rows}
    print(f"Found {len(existing_rows)} existing entries")
    
    # Dataset size mappings (post-filter counts)
    dataset_sizes = {
        "x32": 31997, "x48": 47997, "x64": 63985, "x80": 79992,
        "x96": 96000, "x112": 112000, "x128": 128000, "x144": 144000,
        "x160": 160000, "x176": 176000, "x192": 192000, "x208": 208000, "x224": 224000,
    }
    
    new_rows = []
    
    # Collect optimal dataset size results
    print("\n📊 Collecting optimal dataset size results...")
    for tag in ["x32", "x48", "x64", "x80", "x96", "x112", "x128", "x144", "x160", "x176", "x192", "x208", "x224"]:
        run_dir = ARCHIVE_DIR / f"runs_autoencoder_optimal_{tag}"
        csv_tag = f"{tag}_optimal"
        
        if run_dir.exists():
            metrics = collect_metrics(run_dir, csv_tag, dataset_sizes.get(tag))
            if metrics:
                if csv_tag not in existing_tags:
                    new_rows.append(metrics)
                    print(f"  ✅ Added {csv_tag}")
                else:
                    print(f"  ⏭️  Skipped {csv_tag} (already exists)")
        else:
            print(f"  ⚠️  Missing {run_dir}")
    
    # Collect latent dimension study results
    print("\n📊 Collecting latent dimension study results...")
    latent_dims = [64, 96, 128, 160, 192, 256, 320, 384, 448, 512]
    for dim in latent_dims:
        run_dir = ARCHIVE_DIR / f"runs_autoencoder_latent{dim}"
        csv_tag = f"latent{dim}"
        
        if run_dir.exists():
            metrics = collect_metrics(run_dir, csv_tag, dataset_sizes.get("x160"))  # All use x160 dataset
            if metrics:
                if csv_tag not in existing_tags:
                    new_rows.append(metrics)
                    print(f"  ✅ Added {csv_tag}")
                else:
                    print(f"  ⏭️  Skipped {csv_tag} (already exists)")
    
    # Add new rows to existing
    all_rows = existing_rows + new_rows
    
    # Sort by total_samples, then by dataset name
    all_rows.sort(key=lambda x: (int(x["total_samples"]), x["dataset"]))
    
    # Write updated CSV
    write_csv(all_rows)
    
    print(f"\n✅ Total entries: {len(all_rows)}")
    print(f"   New entries: {len(new_rows)}")
    print(f"   Existing entries: {len(existing_rows)}")


if __name__ == "__main__":
    main()

