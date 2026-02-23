#!/usr/bin/env python3
"""
train_960_1120_models.py
=========================

Train models for 960K and 1120K dataset sizes using the optimal architecture.
"""

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Optimal architecture from x160_optimal_retrained (best architecture)
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
DATASET_SIZES = [960, 1120]
TARGET_RUN_TAG = "optimal_retrained"


def check_dataset_exists(size: int) -> bool:
    """Check if dataset file exists."""
    csv_path = BASE_DIR / "data" / "datasets" / f"all_gas_fastchem_x{size}.csv"
    return csv_path.exists()


def create_config(size: int) -> Path:
    """Create config file for given dataset size."""
    config = OPTIMAL_CONFIG.copy()
    config["data"]["csv_path"] = f"data/datasets/all_gas_fastchem_x{size}.csv"
    
    config_path = BASE_DIR / "configs" / f"x{size}_optimal_retrained.json"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return config_path


def train_model(size: int, config_path: Path):
    """Train model for given dataset size."""
    run_dir = f"results/runs/runs_autoencoder_x{size}_optimal_retrained"
    
    cmd = [
        sys.executable,
        "src/train_autoencoder.py",
        "--config", str(config_path),
        "--loss-type", "log_ratio",
        "--run-dir", run_dir
    ]
    
    print(f"\n{'='*80}")
    print(f"Training x{size}K model")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"Run dir: {run_dir}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=BASE_DIR)
    return result.returncode == 0


def main():
    print("="*80)
    print("TRAINING MODELS FOR 960K AND 1120K")
    print("="*80)
    print(f"Architecture: latent_dim=192, width=512, layers=3, log_ratio loss, static_32")
    print(f"Dataset sizes: {DATASET_SIZES}K")
    print()
    
    # Check datasets
    print("Checking datasets...")
    missing = []
    for size in DATASET_SIZES:
        exists = check_dataset_exists(size)
        status = "✓" if exists else "✗"
        print(f"  {status} x{size}K: {'EXISTS' if exists else 'MISSING'}")
        if not exists:
            missing.append(size)
    
    if missing:
        print(f"\n❌ Error: Missing datasets for sizes: {missing}")
        print("   Please generate datasets first:")
        print("   python scripts/generate_960_1120_datasets.py")
        return
    
    # Create configs
    print("\n" + "="*80)
    print("CREATING CONFIG FILES")
    print("="*80)
    
    configs_created = []
    for size in DATASET_SIZES:
        config_path = create_config(size)
        configs_created.append((size, config_path))
        print(f"✓ Created config: {config_path}")
    
    print(f"\n✓ Created {len(configs_created)} config files")
    
    # Ask before training (since this takes time)
    print("\n" + "="*80)
    print("READY TO TRAIN")
    print("="*80)
    print(f"Will train {len(configs_created)} models:")
    for size, config_path in configs_created:
        print(f"  - x{size}K using {config_path.name}")
    print("\n⚠️  Each model takes ~30-45 minutes (200 epochs)")
    print(f"   Total estimated time: ~{len(configs_created) * 40} minutes")
    
    response = input("\nStart training? (y/n): ")
    if response.lower() != 'y':
        print("Configs created. Run training manually when ready.")
        return
    
    # Train models
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
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
    print("1. Update comparison metrics: python scripts/update_and_regenerate_all.py")
    print("2. Regenerate plots: python scripts/regenerate_all_plots_consistent.py")
    print("3. Analyze if performance has asymptoted")


if __name__ == "__main__":
    main()
