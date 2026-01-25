#!/usr/bin/env python3
"""
generate_new_datasets.py
========================

Generate new dataset sizes (x192, x208, x224) by resampling from existing data.

Usage:
    python generate_new_datasets.py
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
    "x192": 192000,
    "x208": 208000,
    "x224": 224000,
}


def generate_dataset(tag: str, total_samples: int):
    """Generate a new dataset by resampling."""
    print(f"\n{'='*80}")
    print(f"Generating dataset: {tag} ({total_samples:,} samples)")
    print(f"{'='*80}")
    
    # Output paths
    jobs_root = BASE_DIR / "results" / "fastchem_jobs" / f"fastchem_jobs_{tag}"
    output_csv = DATA_DIR / f"all_gas_fastchem_{tag}.csv"
    
    if output_csv.exists():
        print(f"⚠️  Dataset {tag} already exists at {output_csv}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print(f"⏭️  Skipping {tag}")
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
    ]
    
    result = subprocess.run(prepare_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to prepare jobs for {tag}")
        print(result.stderr)
        return
    
    print(f"✅ Job shards prepared in {jobs_root}")
    
    # Step 2: Run FastChem (if needed - this might take a while)
    print(f"\n⚙️  Step 2: Running FastChem...")
    print(f"   This step may require manual execution or a cluster.")
    print(f"   Jobs are in: {jobs_root}")
    print(f"   After FastChem completes, run merge_fastchem_outputs.py")
    
    # Step 3: Merge outputs (commented out - run after FastChem completes)
    print(f"\n📊 Step 3: To merge results after FastChem completes:")
    print(f"   python {SCRIPTS_DIR / 'merge_fastchem_outputs.py'} \\")
    print(f"       --jobs-root {jobs_root} \\")
    print(f"       --reference-csv {REFERENCE_CSV} \\")
    print(f"       --output-csv {output_csv}")
    
    print(f"\n✅ Dataset generation setup complete for {tag}")
    print(f"   Final dataset will be at: {output_csv}")


def main():
    print("="*80)
    print("GENERATE NEW DATASETS (x192, x208, x224)")
    print("="*80)
    
    for tag, total_samples in NEW_DATASETS.items():
        generate_dataset(tag, total_samples)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Dataset generation jobs prepared. Next steps:")
    print("1. Run FastChem on the generated job shards")
    print("2. Merge results using merge_fastchem_outputs.py")
    print("3. Train models using test_dataset_sizes_optimal.py")
    print("="*80)


if __name__ == "__main__":
    main()

