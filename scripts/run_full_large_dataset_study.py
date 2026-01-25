#!/usr/bin/env python3
"""
run_full_large_dataset_study.py
================================

Complete pipeline to run large dataset study overnight:
1. Run FastChem for x240, x480, x640 (sequential)
2. Merge FastChem outputs
3. Train models
4. Run diagnostics
5. Update comparison metrics

Usage:
    python scripts/run_full_large_dataset_study.py

Prerequisites:
    - FASTCHEM_LOGK and FASTCHEM_COND environment variables must be set
    - FastChem Python bindings installed (pyfastchem)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_DIR = BASE_DIR / "data" / "datasets"
LOG_FILE = BASE_DIR / "large_dataset_study.log"

DATASETS = ["x240", "x480", "x640"]
REFERENCE_CSV = DATA_DIR / "all_gas_fastchem_x160.csv"


def log(message: str, level: str = "INFO"):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {
        "INFO": f"[{timestamp}]",
        "SUCCESS": f"[{timestamp}] ✓",
        "ERROR": f"[{timestamp}] ✗",
        "WARNING": f"[{timestamp}] ⚠",
    }.get(level, f"[{timestamp}]")
    
    msg = f"{prefix} {message}"
    print(msg)
    
    # Also write to log file
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def check_prerequisites() -> tuple[str | None, str | None]:
    """Check prerequisites and return FastChem paths."""
    log("Checking prerequisites...")
    
    logk = os.environ.get("FASTCHEM_LOGK")
    cond = os.environ.get("FASTCHEM_COND")
    
    if not logk or not cond:
        log("FASTCHEM_LOGK or FASTCHEM_COND not set", "WARNING")
        log("Trying to find FastChem paths automatically...", "WARNING")
        
        # Try common locations
        common_paths = [
            Path.home() / "FastChem" / "input" / "logK.dat",
            Path.home() / "FastChem" / "input" / "logK_condensates.dat",
            Path("/usr/local/share/fastchem/logK.dat"),
            Path("/opt/fastchem/logK.dat"),
        ]
        
        # Try to infer from pyfastchem location
        try:
            import pyfastchem
            pyfastchem_path = Path(pyfastchem.__file__).parent
            potential_logk = pyfastchem_path.parent / "input" / "logK.dat"
            potential_cond = pyfastchem_path.parent / "input" / "logK_condensates.dat"
            
            if potential_logk.exists():
                logk = str(potential_logk)
                log(f"Found logK.dat at: {logk}", "SUCCESS")
            if potential_cond.exists():
                cond = str(potential_cond)
                log(f"Found logK_condensates.dat at: {cond}", "SUCCESS")
        except Exception:
            pass
    
    if logk and Path(logk).exists():
        log(f"Using FASTCHEM_LOGK: {logk}", "SUCCESS")
    else:
        log(f"FASTCHEM_LOGK not found: {logk}", "ERROR")
        return None, None
    
    if cond and Path(cond).exists():
        log(f"Using FASTCHEM_COND: {cond}", "SUCCESS")
    else:
        log(f"FASTCHEM_COND not found: {cond}", "ERROR")
        return None, None
    
    # Check pyfastchem
    try:
        import pyfastchem
        log("pyfastchem installed", "SUCCESS")
    except ImportError:
        log("pyfastchem not installed!", "ERROR")
        return None, None
    
    return logk, cond


def run_fastchem_for_dataset(tag: str, logk: str, cond: str) -> bool:
    """Run FastChem for a single dataset."""
    log("=" * 80)
    log(f"Processing dataset: {tag}")
    log("=" * 80)
    
    jobs_root = BASE_DIR / "results" / "fastchem_jobs" / f"fastchem_jobs_{tag}"
    output_csv = DATA_DIR / f"all_gas_fastchem_{tag}.csv"
    
    # Check if dataset already exists
    if output_csv.exists():
        log(f"Dataset {tag} already exists: {output_csv}", "WARNING")
        log(f"Skipping FastChem generation for {tag}")
        return True
    
    # Check if job shards exist
    if not jobs_root.exists():
        log(f"Job shards not found: {jobs_root}", "ERROR")
        log("Run: python scripts/generate_large_datasets.py first", "ERROR")
        return False
    
    num_shards = len(list(jobs_root.glob("job_*")))
    log(f"Found {num_shards} shards for {tag}")
    
    # Run FastChem
    log(f"Running FastChem for {tag} (this may take 2-4 hours)...")
    fastchem_log = BASE_DIR / f"fastchem_{tag}.log"
    
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "data_generation" / "run_fastchem_all.py"),
        "--jobs-root", str(jobs_root),
        "--logk", logk,
        "--logk-cond", cond,
        "--chunksize", "128",
    ]
    
    start_time = time.time()
    with open(fastchem_log, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        log(f"FastChem failed for {tag} (check {fastchem_log})", "ERROR")
        return False
    
    log(f"FastChem completed for {tag} in {elapsed/3600:.2f} hours", "SUCCESS")
    
    # Merge results
    log(f"Merging FastChem outputs for {tag}...")
    merge_log = BASE_DIR / f"merge_{tag}.log"
    
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "data_generation" / "merge_fastchem_outputs.py"),
        "--jobs-root", str(jobs_root),
        "--reference-csv", str(REFERENCE_CSV),
        "--output-csv", str(output_csv),
    ]
    
    with open(merge_log, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    if result.returncode != 0:
        log(f"Merge failed for {tag} (check {merge_log})", "ERROR")
        return False
    
    log(f"Merge completed for {tag}", "SUCCESS")
    
    # Check output file
    if output_csv.exists():
        num_rows = sum(1 for _ in open(output_csv)) - 1  # Subtract header
        log(f"Merged dataset has {num_rows:,} rows", "SUCCESS")
    
    log(f"Dataset {tag} complete!", "SUCCESS")
    return True


def train_models() -> bool:
    """Train models and run diagnostics."""
    log("=" * 80)
    log("Training models")
    log("=" * 80)
    
    cmd = [sys.executable, str(SCRIPTS_DIR / "train_large_datasets.py")]
    result = subprocess.run(cmd, cwd=BASE_DIR)
    
    if result.returncode != 0:
        log("Model training failed", "ERROR")
        return False
    
    log("Model training completed", "SUCCESS")
    return True


def main():
    """Main execution."""
    # Initialize log file
    LOG_FILE.write_text(f"=== Large Dataset Study Log ===\n")
    LOG_FILE.write_text(f"Started: {datetime.now()}\n\n")
    
    print()
    print("=" * 80)
    print("LARGE DATASET STUDY - FULL PIPELINE")
    print("=" * 80)
    print()
    print("This script will:")
    print("  1. Run FastChem for x240, x480, x640 (sequential)")
    print("  2. Merge FastChem outputs")
    print("  3. Train models")
    print("  4. Run diagnostics")
    print("  5. Update comparison metrics")
    print()
    print(f"Estimated time: 6-12 hours (depending on hardware)")
    print(f"Log file: {LOG_FILE}")
    print()
    
    # Check prerequisites
    logk, cond = check_prerequisites()
    if not logk or not cond:
        log("Cannot proceed without FastChem paths", "ERROR")
        print()
        print("Please set:")
        print("  export FASTCHEM_LOGK=/path/to/logK.dat")
        print("  export FASTCHEM_COND=/path/to/logK_condensates.dat")
        print()
        print("Then run:")
        print("  python scripts/run_full_large_dataset_study.py")
        return 1
    
    # Process each dataset
    all_success = True
    for tag in DATASETS:
        if not run_fastchem_for_dataset(tag, logk, cond):
            log(f"Failed to process {tag}, continuing with next...", "WARNING")
            all_success = False
    
    # Train models
    if not train_models():
        log("Model training failed", "ERROR")
        all_success = False
    
    # Final summary
    print()
    print("=" * 80)
    if all_success:
        print("STUDY COMPLETE!")
    else:
        print("STUDY COMPLETE (with some errors)")
    print("=" * 80)
    print()
    log("All datasets processed and models trained", "SUCCESS")
    print()
    print("Results:")
    print("  - Comparison metrics: plots/comparison_metrics.csv")
    print("  - Comparison plot: plots/large_dataset_comparison.png")
    print("  - Individual diagnostics: results/runs/runs_autoencoder_*_static_32/diagnostics/")
    print()
    print(f"Full log: {LOG_FILE}")
    
    LOG_FILE.write_text(f"\nCompleted: {datetime.now()}\n", mode="a")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
