#!/usr/bin/env python3
"""
resume_3360_3680_4000_pipeline.py
==================================

Resume the 3360K, 3680K, 4000K pipeline from where it left off.
Checks current state and continues from the appropriate step.
"""

import subprocess
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "datasets"
SCRIPTS_DIR = BASE_DIR / "scripts" / "data_generation"
RUNS_DIR = BASE_DIR / "results" / "runs"
JOBS_DIR = BASE_DIR / "results" / "fastchem_jobs"
REFERENCE_CSV = DATA_DIR / "all_gas_fastchem_x160.csv"

DATASET_SIZES = {
    "x3360": 3360000,
    "x3680": 3680000,
    "x4000": 4000000,
}


def check_dataset_exists(tag: str) -> bool:
    """Check if final dataset CSV exists."""
    csv_path = DATA_DIR / f"all_gas_fastchem_{tag}.csv"
    if csv_path.exists():
        # Verify it's not empty
        try:
            size = csv_path.stat().st_size
            if size > 1024:  # At least 1KB
                return True
        except:
            pass
    return False


def check_fastchem_jobs_prepared(tag: str) -> bool:
    """Check if FastChem job shards are prepared."""
    jobs_root = JOBS_DIR / f"fastchem_jobs_{tag}"
    if not jobs_root.exists():
        return False
    # Check if there are any job directories
    job_dirs = [d for d in jobs_root.iterdir() if d.is_dir() and d.name.startswith("job_")]
    return len(job_dirs) > 0


def check_fastchem_complete(tag: str) -> bool:
    """Check if FastChem has been run for all shards."""
    jobs_root = JOBS_DIR / f"fastchem_jobs_{tag}"
    if not jobs_root.exists():
        return False
    
    job_dirs = sorted([d for d in jobs_root.iterdir() if d.is_dir() and d.name.startswith("job_")])
    if not job_dirs:
        return False
    
    # Check if all jobs have results
    complete_count = 0
    for job_dir in job_dirs:
        result_file = job_dir / "results" / "gas_species.csv"
        if result_file.exists() and result_file.stat().st_size > 0:
            complete_count += 1
    
    # Consider complete if at least 95% of jobs are done (allowing for some failures)
    return complete_count >= len(job_dirs) * 0.95


def check_model_trained(tag: str) -> bool:
    """Check if model has been trained."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{tag}_optimal_retrained"
    if not run_dir.exists():
        return False
    summary_path = run_dir / "summary.json"
    return summary_path.exists()


def prepare_fastchem_jobs(tag: str, total_samples: int):
    """Prepare FastChem job shards."""
    print(f"\n{'='*80}")
    print(f"PREPARING FASTCHEM JOBS: {tag}")
    print(f"{'='*80}")
    
    jobs_root = JOBS_DIR / f"fastchem_jobs_{tag}"
    
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "prepare_fastchem_jobs.py"),
        "--reference-csv", str(REFERENCE_CSV),
        "--output-root", str(jobs_root),
        "--total-samples", str(total_samples),
        "--shard-size", "2000",
        "--strategy", "empirical",
        "--temp-jitter", "50.0",
        "--logp-jitter", "0.1",
        "--dex-jitter", "0.05",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to prepare jobs for {tag}")
        print(result.stderr)
        return False
    
    print(f"✅ Job shards prepared: {jobs_root}")
    return True


def run_fastchem(tag: str):
    """Run FastChem on all shards."""
    print(f"\n{'='*80}")
    print(f"RUNNING FASTCHEM: {tag}")
    print(f"{'='*80}")
    
    jobs_root = JOBS_DIR / f"fastchem_jobs_{tag}"
    
    # Check environment variables
    fastchem_logk = os.environ.get("FASTCHEM_LOGK")
    fastchem_cond = os.environ.get("FASTCHEM_COND")
    fastchem_elem = os.environ.get("FASTCHEM_ELEM")
    
    if not (fastchem_logk and fastchem_cond):
        print(f"⚠️  FastChem environment variables not set")
        print(f"   Set FASTCHEM_LOGK, FASTCHEM_COND, and FASTCHEM_ELEM")
        print(f"   Then run manually:")
        print(f"   python {SCRIPTS_DIR / 'run_fastchem_all.py'} \\")
        print(f"       --jobs-root {jobs_root} \\")
        print(f"       --logk $FASTCHEM_LOGK \\")
        print(f"       --logk-cond $FASTCHEM_COND \\")
        if fastchem_elem:
            print(f"       --element-abundances $FASTCHEM_ELEM \\")
        print(f"       --chunksize 128")
        return False
    
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_fastchem_all.py"),
        "--jobs-root", str(jobs_root),
        "--logk", fastchem_logk,
        "--logk-cond", fastchem_cond,
        "--chunksize", "128",
    ]
    
    if fastchem_elem:
        cmd.extend(["--element-abundances", fastchem_elem])
    
    print(f"Running FastChem (this will take 3-5 hours)...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ FastChem failed for {tag}")
        print(result.stderr[:1000])
        return False
    
    print(f"✅ FastChem completed for {tag}")
    return True


def merge_fastchem_results(tag: str):
    """Merge FastChem outputs into final CSV."""
    print(f"\n{'='*80}")
    print(f"MERGING FASTCHEM RESULTS: {tag}")
    print(f"{'='*80}")
    
    jobs_root = JOBS_DIR / f"fastchem_jobs_{tag}"
    output_csv = DATA_DIR / f"all_gas_fastchem_{tag}.csv"
    
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "merge_fastchem_outputs.py"),
        "--jobs-root", str(jobs_root),
        "--reference-csv", str(REFERENCE_CSV),
        "--output-csv", str(output_csv),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to merge results for {tag}")
        print(result.stderr)
        return False
    
    print(f"✅ Dataset {tag} merged successfully!")
    print(f"   Output: {output_csv}")
    return True


def create_config(size: int):
    """Create config file for given dataset size."""
    import json
    
    config = {
        "data": {
            "train_frac": 0.85,
            "val_frac": 0.10,
            "test_frac": 0.05,
            "target_topk_species": 20,
            "include_fz_as_feature": True,
            "use_static_species_list": True,
            "static_species_list_path": "static_species_list_32.json",
            "input_cols_manual": None,
            "target_cols_manual": None,
            "csv_path": f"data/datasets/all_gas_fastchem_x{size}.csv"
        },
        "optimization": {
            "epochs": 200,
            "batch_size": 512,
            "learning_rate": 5e-4,
            "weight_decay": 1e-5,
            "grad_clip": 5.0,
            "seed": 42
        },
        "architecture": {
            "latent_dim": 192,
            "encoder_hidden": [512, 512, 512],
            "dynamics_hidden": [512, 512, 512],
            "decoder_hidden": [512, 512, 512],
            "activation": "silu",
            "dropout": 0.0
        },
        "loss": {
            "type": "log_ratio",
            "use_weighted": True
        },
        "normalization": {
            "temp_divisor": 4000.0,
            "input_log_scale": 10.0,
            "abund_epsilon_offset": 12.0,
            "abund_dex_scale": 10.0,
            "target_zero_floor": 1e-30,
            "target_log_scale": 30.0,
            "log_eps": 1e-30
        },
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6
        }
    }
    
    config_path = BASE_DIR / "configs" / f"x{size}_optimal_retrained.json"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return config_path


def train_model(tag: str, size: int):
    """Train model for given dataset."""
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL: {tag}")
    print(f"{'='*80}")
    
    # Check if config exists, create if needed
    config_path = BASE_DIR / "configs" / f"x{size}_optimal_retrained.json"
    if not config_path.exists():
        print(f"Creating config file...")
        config_path = create_config(size)
        if not config_path.exists():
            print(f"❌ Failed to create config")
            return False
        print(f"✅ Created config: {config_path}")
    
    run_dir = f"results/runs/runs_autoencoder_x{size}_optimal_retrained"
    
    cmd = [
        sys.executable,
        "src/train_autoencoder.py",
        "--config", str(config_path),
        "--loss-type", "log_ratio",
        "--run-dir", run_dir
    ]
    
    print(f"Training model (this will take ~40-50 minutes)...")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    return result.returncode == 0


def main():
    print("="*80)
    print("RESUMING 3360K, 3680K, 4000K PIPELINE")
    print("="*80)
    print()
    
    # Check current state
    print("Checking current state...")
    print()
    
    status = {}
    for tag, total_samples in DATASET_SIZES.items():
        dataset_exists = check_dataset_exists(tag)
        jobs_prepared = check_fastchem_jobs_prepared(tag)
        fastchem_complete = check_fastchem_complete(tag) if jobs_prepared else False
        model_trained = check_model_trained(tag)
        
        status[tag] = {
            "dataset_exists": dataset_exists,
            "jobs_prepared": jobs_prepared,
            "fastchem_complete": fastchem_complete,
            "model_trained": model_trained,
        }
        
        print(f"{tag}:")
        print(f"  Dataset CSV: {'✓ EXISTS' if dataset_exists else '✗ MISSING'}")
        print(f"  Jobs prepared: {'✓ YES' if jobs_prepared else '✗ NO'}")
        print(f"  FastChem complete: {'✓ YES' if fastchem_complete else '✗ NO'}")
        print(f"  Model trained: {'✓ YES' if model_trained else '✗ NO'}")
        print()
    
    # Determine what needs to be done
    print("="*80)
    print("RESUMING FROM CHECKPOINT")
    print("="*80)
    print()
    
    # Process each dataset
    for tag, total_samples in DATASET_SIZES.items():
        s = status[tag]
        
        if s["model_trained"]:
            print(f"⏭️  {tag}: Model already trained, skipping")
            continue
        
        if s["dataset_exists"]:
            print(f"✅ {tag}: Dataset exists, proceeding to training")
            if not train_model(tag, int(tag.replace("x", ""))):
                print(f"❌ Training failed for {tag}")
                return
            continue
        
        if s["fastchem_complete"]:
            print(f"✅ {tag}: FastChem complete, merging results...")
            if not merge_fastchem_results(tag):
                print(f"❌ Merge failed for {tag}")
                return
            print(f"✅ {tag}: Dataset ready, proceeding to training")
            if not train_model(tag, int(tag.replace("x", ""))):
                print(f"❌ Training failed for {tag}")
                return
            continue
        
        if s["jobs_prepared"]:
            print(f"⚠️  {tag}: Jobs prepared but FastChem not complete")
            print(f"   Checking if FastChem can be run...")
            
            # Check environment variables
            if os.environ.get("FASTCHEM_LOGK") and os.environ.get("FASTCHEM_COND"):
                print(f"   Running FastChem...")
                if not run_fastchem(tag):
                    print(f"   ⚠️  FastChem failed or needs manual intervention")
                    print(f"   Continuing with next dataset...")
                    continue
                
                # Merge after FastChem
                if not merge_fastchem_results(tag):
                    print(f"❌ Merge failed for {tag}")
                    return
                
                # Train model
                if not train_model(tag, int(tag.replace("x", ""))):
                    print(f"❌ Training failed for {tag}")
                    return
            else:
                print(f"   ⚠️  FastChem environment variables not set")
                print(f"   Please set FASTCHEM_LOGK and FASTCHEM_COND, then run:")
                print(f"   python {SCRIPTS_DIR / 'run_fastchem_all.py'} \\")
                print(f"       --jobs-root {JOBS_DIR / f'fastchem_jobs_{tag}'} \\")
                print(f"       --logk $FASTCHEM_LOGK \\")
                print(f"       --logk-cond $FASTCHEM_COND \\")
                print(f"       --chunksize 128")
                continue
        
        # Need to prepare jobs
        print(f"📝 {tag}: Preparing FastChem jobs...")
        if not prepare_fastchem_jobs(tag, total_samples):
            print(f"❌ Failed to prepare jobs for {tag}")
            return
        
        # Check if we can run FastChem
        if os.environ.get("FASTCHEM_LOGK") and os.environ.get("FASTCHEM_COND"):
            print(f"   Running FastChem...")
            if not run_fastchem(tag):
                print(f"   ⚠️  FastChem failed or needs manual intervention")
                continue
            
            # Merge
            if not merge_fastchem_results(tag):
                print(f"❌ Merge failed for {tag}")
                return
            
            # Train
            if not train_model(tag, int(tag.replace("x", ""))):
                print(f"❌ Training failed for {tag}")
                return
        else:
            print(f"   ⚠️  FastChem environment variables not set")
            print(f"   Jobs prepared. Please set FASTCHEM_LOGK and FASTCHEM_COND, then run FastChem manually.")
            continue
    
    # Update plots
    print("\n" + "="*80)
    print("UPDATING METRICS AND PLOTS")
    print("="*80)
    
    result = subprocess.run(
        [sys.executable, "scripts/update_plots_for_optimal_retrained.py"],
        cwd=BASE_DIR
    )
    
    if result.returncode == 0:
        print("\n✅ Pipeline complete!")
    else:
        print("\n⚠️  Pipeline completed but plot update had issues")


if __name__ == "__main__":
    main()
