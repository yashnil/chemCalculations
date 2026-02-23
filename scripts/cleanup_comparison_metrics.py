#!/usr/bin/env python3
"""
cleanup_comparison_metrics.py
==============================

Clean up comparison_metrics.csv to only include the main comparative dataset sizes:
- base (baseline)
- x160, x480, x800, x1120, x1440, x1760, x2080, x2400, x2720, x3040 (optimal_retrained)
"""

import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
COMPARISON_CSV = BASE_DIR / "plots" / "comparison_metrics.csv"

# Main comparative dataset sizes to keep
MAIN_SIZES = [160, 480, 800, 1120, 1440, 1760, 2080, 2400, 2720, 3040]
KEEP_TAGS = {f"x{size}_optimal_retrained" for size in MAIN_SIZES}
KEEP_TAGS.add("base")  # Keep baseline


def read_existing_csv():
    """Read existing comparison_metrics.csv"""
    if not COMPARISON_CSV.exists():
        print(f"❌ File not found: {COMPARISON_CSV}")
        return []
    
    rows = []
    with COMPARISON_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_csv(rows):
    """Write cleaned rows to comparison_metrics.csv"""
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
    print("CLEANING UP COMPARISON METRICS")
    print("="*80)
    print()
    print("Keeping only:")
    print("  - base (baseline)")
    print(f"  - {len(MAIN_SIZES)} optimal_retrained runs: {', '.join(f'x{s}K' for s in MAIN_SIZES)}")
    print()
    
    # Read existing CSV
    existing_rows = read_existing_csv()
    print(f"Found {len(existing_rows)} existing entries")
    
    # Filter to keep only main comparative sizes
    cleaned_rows = []
    removed_count = 0
    
    for row in existing_rows:
        tag = row["dataset"]
        if tag in KEEP_TAGS or tag == "base":
            cleaned_rows.append(row)
        else:
            removed_count += 1
            print(f"  🗑️  Removing: {tag}")
    
    # Sort by dataset size
    def sort_key(row):
        tag = row["dataset"]
        if tag == "base":
            return (0, tag)
        # Extract size from tag like "x160_optimal_retrained"
        try:
            size_str = tag.split("_")[0].replace("x", "")
            return (int(size_str), tag)
        except:
            return (999999, tag)
    
    cleaned_rows.sort(key=sort_key)
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Removed: {removed_count} entries")
    print(f"Kept: {len(cleaned_rows)} entries")
    print()
    print("Kept entries:")
    for row in cleaned_rows:
        tag = row["dataset"]
        size = row.get("total_samples", "N/A")
        test_loss = row.get("test_loss", "N/A")
        log_mae = row.get("log_mae", "N/A")
        print(f"  ✓ {tag:30s} | samples: {size:>10s} | test_loss: {test_loss:>12s} | log_mae: {log_mae}")
    
    # Write cleaned CSV
    print()
    write_csv(cleaned_rows)
    
    print()
    print("="*80)
    print("✅ CLEANUP COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("  - Regenerate plots: python scripts/update_plots_for_optimal_retrained.py")
    print("  - Check plots/performance_vs_size.png to see clean trend")


if __name__ == "__main__":
    main()
