#!/usr/bin/env python3
"""
generate_and_retrain_complete.py
=================================

Complete pipeline to:
1. Generate missing datasets (320K, 800K)
2. Create configs for 160, 320, 480, 640, 800 using optimal architecture
3. Train all models
4. Update comparison metrics
5. Regenerate all plots

This ensures we have complete side-by-side comparison for all requested sizes.
"""

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Optimal architecture from x160_static_32_consistent (best 160K model)
OPTIMAL_CONFIG = {
    "data": {
        "train_frac": 0.85,
        "val_frac": 0.10,
        "test_frac": 0.05,
        "target_topk_species": 20,
        "include_fz_as_feature": True,
        "use_static_species_list": True,
        "static_species_list_path": "static_species_list_32.json",
        "input_cols_manual": None,
        "target_cols_manual": None
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

# Dataset sizes to train
DATASET_SIZES = [160, 320, 480, 640, 800]
TARGET_RUN_TAG = "optimal_retrained"  # Consistent tag for all runs


def check_dataset_exists(size: int) -> bool:
    """Check if dataset file exists."""
    csv_path = BASE_DIR / "data" / "datasets" / f"all_gas_fastchem_x{size}.csv"
    return csv_path.exists()


def generate_dataset(size: int):
    """Generate a dataset using the data generation pipeline."""
    print(f"\n{'='*80}")
    print(f"Generating dataset: x{size} ({size:,} samples)")
    print(f"{'='*80}")
    
    tag = f"x{size}"
    csv_path = BASE_DIR / "data" / "datasets" / f"all_gas_fastchem_{tag}.csv"
    
    if csv_path.exists():
        print(f"✓ Dataset {tag} already exists")
        return True
    
    # Use generate_large_datasets.py approach
    reference_csv = BASE_DIR / "data" / "datasets" / "all_gas_fastchem_x160.csv"
    if not reference_csv.exists():
        print(f"❌ Reference CSV not found: {reference_csv}")
        return False
    
    jobs_root = BASE_DIR / "results" / "fastchem_jobs" / f"fastchem_jobs_{tag}"
    scripts_dir = BASE_DIR / "scripts" / "data_generation"
    
    # Step 1: Prepare FastChem jobs
    print(f"\n📝 Step 1: Preparing FastChem job shards...")
    prepare_cmd = [
        sys.executable,
        str(scripts_dir / "prepare_fastchem_jobs.py"),
        "--reference-csv", str(reference_csv),
        "--output-root", str(jobs_root),
        "--total-samples", str(size * 1000),
        "--shard-size", "2000",
        "--strategy", "empirical",
        "--temp-jitter", "50.0",
        "--logp-jitter", "0.1",
        "--dex-jitter", "0.05",
    ]
    
    result = subprocess.run(prepare_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to prepare jobs for {tag}")
        print(result.stderr)
        return False
    
    print(f"✅ Job shards prepared in {jobs_root}")
    print(f"\n⚠️  NOTE: FastChem jobs need to be run manually, then merge results:")
    print(f"   python {scripts_dir / 'merge_fastchem_outputs.py'} \\")
    print(f"       --jobs-root {jobs_root} \\")
    print(f"       --reference-csv {reference_csv} \\")
    print(f"       --output-csv {csv_path}")
    
    return False  # Not complete until FastChem runs


def create_config(size: int) -> Path:
    """Create config file for given dataset size."""
    config = OPTIMAL_CONFIG.copy()
    config["data"]["csv_path"] = f"data/datasets/all_gas_fastchem_x{size}.csv"
    
    config_path = BASE_DIR / "configs" / f"x{size}_{TARGET_RUN_TAG}.json"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return config_path


def train_model(size: int, config_path: Path):
    """Train model for given dataset size."""
    run_dir = BASE_DIR / "results" / "runs" / f"runs_autoencoder_x{size}_{TARGET_RUN_TAG}"
    
    cmd = [
        sys.executable,
        str(BASE_DIR / "src" / "train_autoencoder.py"),
        "--config", str(config_path),
        "--loss-type", "log_ratio",
        "--run-dir", str(run_dir)
    ]
    
    print(f"\n{'='*80}")
    print(f"Training x{size}K model")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"Run dir: {run_dir}")
    print()
    
    result = subprocess.run(cmd, cwd=BASE_DIR)
    return result.returncode == 0


def main():
    print("="*80)
    print("COMPLETE PIPELINE: GENERATE DATASETS & RETRAIN MODELS")
    print("="*80)
    print(f"Architecture: latent_dim=192, width=512, layers=3, log_ratio loss, static_32")
    print(f"Dataset sizes: {DATASET_SIZES}K")
    print()
    
    # Step 1: Check datasets
    print("="*80)
    print("STEP 1: CHECKING DATASETS")
    print("="*80)
    
    missing = []
    for size in DATASET_SIZES:
        exists = check_dataset_exists(size)
        status = "✓" if exists else "✗"
        print(f"  {status} x{size}K: {'EXISTS' if exists else 'MISSING'}")
        if not exists:
            missing.append(size)
    
    if missing:
        print(f"\n⚠️  Missing datasets: {missing}")
        print("   These need to be generated before training.")
        print("   Generating missing datasets...")
        
        for size in missing:
            if not generate_dataset(size):
                print(f"\n❌ Could not generate x{size}K dataset")
                print("   Please generate manually using:")
                print(f"   python scripts/generate_large_datasets.py")
                print("   Then run FastChem and merge results.")
                return
    
    # Step 2: Create configs
    print("\n" + "="*80)
    print("STEP 2: CREATING CONFIG FILES")
    print("="*80)
    
    configs_created = []
    for size in DATASET_SIZES:
        if not check_dataset_exists(size):
            print(f"⚠️  Skipping x{size}K - dataset not found")
            continue
        
        config_path = create_config(size)
        configs_created.append((size, config_path))
        print(f"✓ Created config: {config_path}")
    
    print(f"\n✓ Created {len(configs_created)} config files")
    
    # Step 3: Train models
    print("\n" + "="*80)
    print("STEP 3: TRAINING MODELS")
    print("="*80)
    print(f"Will train {len(configs_created)} models:")
    for size, config_path in configs_created:
        print(f"  - x{size}K using {config_path.name}")
    print("\n⚠️  Each model takes ~30 minutes (200 epochs)")
    print(f"   Total estimated time: ~{len(configs_created) * 30} minutes")
    
    response = input("\nStart training? (y/n): ")
    if response.lower() != 'y':
        print("Configs created. Run training manually when ready.")
        return
    
    for size, config_path in configs_created:
        success = train_model(size, config_path)
        if success:
            print(f"✓ Completed training for x{size}K")
        else:
            print(f"✗ Training failed for x{size}K")
    
    print("\n" + "="*80)
    print("✅ All training jobs completed!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run diagnostics on all models to get log_mae and log_r2")
    print("2. Update comparison_metrics.csv")
    print("3. Regenerate all plots")


if __name__ == "__main__":
    main()
