#!/usr/bin/env python3
"""
generate_large_datasets.py
===========================

Generate larger dataset sizes (x240, x480, x640) by resampling from x160 with FastChem.
This script prepares FastChem job shards that need to be run.

Usage:
    python generate_large_datasets.py
    # Then run FastChem on the generated jobs
    # Then merge results using merge_fastchem_outputs.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "datasets"
SCRIPTS_DIR = BASE_DIR / "scripts" / "data_generation"

# Reference CSV (use x160 as base)
REFERENCE_CSV = DATA_DIR / "all_gas_fastchem_x160.csv"

# New dataset sizes to generate
NEW_DATASETS = {
    "x240": 240000,
    "x480": 480000,
    "x640": 640000,
}


def generate_dataset(tag: str, total_samples: int):
    """Generate a new dataset by resampling."""
    print(f"\n{'='*80}")
    print(f"Generating dataset: {tag} ({total_samples:,} samples)")
    print(f"{'='*80}")
    
    # Output paths
    jobs_root = BASE_DIR / f"fastchem_jobs_{tag}"
    output_csv = DATA_DIR / f"all_gas_fastchem_{tag}.csv"
    
    if output_csv.exists():
        print(f"⚠️  Dataset {tag} already exists at {output_csv}")
        print(f"⏭️  Skipping {tag} (delete file to regenerate)")
        return
    
    # Check if reference CSV exists
    if not REFERENCE_CSV.exists():
        print(f"❌ Reference CSV not found: {REFERENCE_CSV}")
        print("   Please ensure x160 dataset exists first.")
        return
    
    # Step 1: Prepare FastChem jobs
    print(f"\n📝 Step 1: Preparing FastChem job shards...")
    prepare_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "prepare_fastchem_jobs.py"),
        "--reference-csv", str(REFERENCE_CSV),
        "--output-root", str(jobs_root),
        "--total-samples", str(total_samples),
        "--shard-size", "2000",
        "--strategy", "empirical",
        "--temp-jitter", "50.0",  # ±50K jitter
        "--logp-jitter", "0.1",   # ±0.1 dex jitter
        "--dex-jitter", "0.05",  # ±0.05 dex jitter
    ]
    
    result = subprocess.run(prepare_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to prepare jobs for {tag}")
        print(result.stderr)
        return
    
    print(f"✅ Job shards prepared in {jobs_root}")
    print(f"\n📊 Next steps:")
    print(f"   1. Run FastChem on all shards in {jobs_root}")
    print(f"   2. Merge results:")
    print(f"      python {SCRIPTS_DIR / 'merge_fastchem_outputs.py'} \\")
    print(f"          --jobs-root {jobs_root} \\")
    print(f"          --reference-csv {REFERENCE_CSV} \\")
    print(f"          --output-csv {output_csv}")
    
    print(f"\n✅ Dataset generation setup complete for {tag}")
    print(f"   Final dataset will be at: {output_csv}")


def main():
    print("="*80)
    print("GENERATE LARGE DATASETS (x240, x480, x640)")
    print("="*80)
    print("\nThis script prepares FastChem job shards.")
    print("After FastChem completes, merge results using merge_fastchem_outputs.py")
    print("="*80)
    
    for tag, total_samples in NEW_DATASETS.items():
        generate_dataset(tag, total_samples)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Dataset generation jobs prepared. Next steps:")
    print("1. Run FastChem on the generated job shards (this may take hours)")
    print("2. Merge results using merge_fastchem_outputs.py")
    print("3. Train models using train_large_datasets.py")
    print("="*80)


if __name__ == "__main__":
    main()
