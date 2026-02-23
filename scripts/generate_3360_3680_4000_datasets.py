#!/usr/bin/env python3
"""
generate_3360_3680_4000_datasets.py
===================================

Generate datasets for 3360K, 3680K, and 4000K samples to continue the constant increment study.
"""

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
    "x3360": 3360000,
    "x3680": 3680000,
    "x4000": 4000000,
}


def generate_dataset(tag: str, total_samples: int):
    """Generate a new dataset by preparing FastChem job shards."""
    print(f"\n{'='*80}")
    print(f"Generating dataset: {tag} ({total_samples:,} samples)")
    print(f"{'='*80}")
    
    # Output paths
    jobs_root = BASE_DIR / "results" / "fastchem_jobs" / f"fastchem_jobs_{tag}"
    output_csv = DATA_DIR / f"all_gas_fastchem_{tag}.csv"
    
    if output_csv.exists():
        print(f"✓ Dataset {tag} already exists at {output_csv}")
        print(f"⏭️  Skipping {tag}")
        return True
    
    # Check if reference CSV exists
    if not REFERENCE_CSV.exists():
        print(f"❌ Reference CSV not found: {REFERENCE_CSV}")
        print("   Please ensure x160 dataset exists first.")
        return False
    
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
        return False
    
    print(f"✅ Job shards prepared in {jobs_root}")
    
    # Check if FastChem environment variables are set
    import os
    fastchem_logk = os.environ.get("FASTCHEM_LOGK")
    fastchem_cond = os.environ.get("FASTCHEM_COND")
    fastchem_elem = os.environ.get("FASTCHEM_ELEM")
    
    if fastchem_logk and fastchem_cond and fastchem_elem:
        print(f"\n📊 Step 2: Running FastChem on all shards...")
        print(f"   This will take 3-5 hours per dataset...")
        
        run_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "run_fastchem_all.py"),
            "--jobs-root", str(jobs_root),
            "--logk", fastchem_logk,
            "--logk-cond", fastchem_cond,
            "--element-abundances", fastchem_elem,
            "--chunksize", "128",
        ]
        
        print(f"   Running: {' '.join(run_cmd)}")
        result = subprocess.run(run_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ FastChem failed for {tag}")
            print(result.stderr[:500])
            return False
        
        print(f"✅ FastChem completed for {tag}")
        
        # Step 3: Merge results
        print(f"\n📊 Step 3: Merging FastChem outputs...")
        merge_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "merge_fastchem_outputs.py"),
            "--jobs-root", str(jobs_root),
            "--reference-csv", str(REFERENCE_CSV),
            "--output-csv", str(output_csv),
        ]
        
        result = subprocess.run(merge_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to merge results for {tag}")
            print(result.stderr)
            return False
        
        print(f"✅ Dataset {tag} generated successfully!")
        print(f"   Output: {output_csv}")
        return True
    else:
        print(f"\n⚠️  FastChem environment variables not set")
        print(f"   Set FASTCHEM_LOGK, FASTCHEM_COND, and FASTCHEM_ELEM environment variables")
        print(f"   Then run FastChem manually:")
        print(f"   python {SCRIPTS_DIR / 'run_fastchem_all.py'} \\")
        print(f"       --jobs-root {jobs_root} \\")
        print(f"       --logk $FASTCHEM_LOGK \\")
        print(f"       --logk-cond $FASTCHEM_COND \\")
        print(f"       --element-abundances $FASTCHEM_ELEM \\")
        print(f"       --chunksize 128")
        print(f"\n   Then merge results:")
        print(f"   python {SCRIPTS_DIR / 'merge_fastchem_outputs.py'} \\")
        print(f"       --jobs-root {jobs_root} \\")
        print(f"       --reference-csv {REFERENCE_CSV} \\")
        print(f"       --output-csv {output_csv}")
        return False


def main():
    print("="*80)
    print("GENERATE DATASETS FOR 3360K, 3680K, AND 4000K")
    print("="*80)
    print("\nThis script will:")
    print("1. Prepare FastChem job shards")
    print("2. Run FastChem (if environment variables are set)")
    print("3. Merge results into final CSV files")
    print("="*80)
    
    success_count = 0
    for tag, total_samples in NEW_DATASETS.items():
        if generate_dataset(tag, total_samples):
            success_count += 1
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Successfully generated: {success_count}/{len(NEW_DATASETS)} datasets")
    
    if success_count == len(NEW_DATASETS):
        print("\n✅ All datasets generated! Ready to train models.")
    else:
        print("\n⚠️  Some datasets still need FastChem to be run manually.")
        print("   Check the output above for instructions.")


if __name__ == "__main__":
    main()
