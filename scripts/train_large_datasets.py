#!/usr/bin/env python3
"""
train_large_datasets.py
========================

Train models on larger dataset sizes (x240, x480, x640) using the best configuration
(x160_static_32 settings: latent_dim=192, log_ratio loss, static 32 species).

This script:
1. Trains models for x240, x480, x640
2. Runs diagnostics for each
3. Updates comparison metrics
4. Creates comparison plots vs x160_static_32

Usage:
    python train_large_datasets.py
    python train_large_datasets.py --datasets x240 x480  # Train specific datasets
    python train_large_datasets.py --skip-training  # Only run diagnostics/comparison
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data" / "datasets"
CONFIGS_DIR = BASE_DIR / "configs"
PLOTS_DIR = BASE_DIR / "plots"

# Datasets to train
DEFAULT_DATASETS = ["x240", "x480", "x640"]
BASELINE_MODEL = "x160_static_32"
BASELINE_RUN_DIR = BASE_DIR / "results" / "runs" / f"runs_autoencoder_{BASELINE_MODEL}"


def check_dataset_exists(tag: str) -> bool:
    """Check if dataset CSV exists."""
    csv_path = DATA_DIR / f"all_gas_fastchem_{tag}.csv"
    return csv_path.exists()


def train_model(tag: str) -> Optional[Path]:
    """Train a model for a given dataset tag."""
    config_path = CONFIGS_DIR / f"{tag}_static_32_config.json"
    run_dir = BASE_DIR / "results" / "runs" / f"runs_autoencoder_{tag}_static_32"
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return None
    
    if not check_dataset_exists(tag):
        print(f"❌ Dataset not found: {DATA_DIR / f'all_gas_fastchem_{tag}.csv'}")
        print(f"   Please generate it first using generate_large_datasets.py")
        return None
    
    print(f"\n{'='*80}")
    print(f"Training model: {tag}_static_32")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"Dataset: {DATA_DIR / f'all_gas_fastchem_{tag}.csv'}")
    print(f"Run directory: {run_dir}")
    print(f"{'='*80}\n")
    
    # Run training
    cmd = [
        sys.executable,
        str(SRC_DIR / "train_autoencoder.py"),
        "--config", str(config_path),
        "--loss-type", "log_ratio",
        "--run-dir", str(run_dir),
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ Training failed for {tag}")
        return None
    
    print(f"✅ Training completed in {elapsed/60:.1f} minutes")
    return run_dir


def run_diagnostics(run_dir: Path) -> bool:
    """Run diagnostics for a trained model."""
    print(f"\n{'='*80}")
    print(f"Running diagnostics: {run_dir.name}")
    print(f"{'='*80}\n")
    
    best_model_py = run_dir / "best_model.py"
    if not best_model_py.exists():
        print(f"❌ best_model.py not found in {run_dir}")
        return False
    
    # Set environment variables for diagnostics
    env = os.environ.copy()
    env["BEST_MODULE"] = str(best_model_py)
    env["CSV_PATH"] = str(DATA_DIR / f"all_gas_fastchem_{run_dir.name.split('_')[-2]}.csv")
    env["OUT_DIR"] = str(run_dir / "diagnostics")
    
    cmd = [sys.executable, str(SRC_DIR / "diagnostics.py")]
    result = subprocess.run(cmd, cwd=BASE_DIR, env=env)
    
    if result.returncode != 0:
        print(f"❌ Diagnostics failed for {run_dir.name}")
        return False
    
    print(f"✅ Diagnostics completed")
    return True


def update_comparison_metrics() -> None:
    """Update comparison_metrics.csv with new results."""
    print(f"\n{'='*80}")
    print("Updating comparison metrics")
    print(f"{'='*80}\n")
    
    cmd = [sys.executable, str(SRC_DIR / "make_comparison_metrics.py")]
    result = subprocess.run(cmd, cwd=BASE_DIR)
    
    if result.returncode != 0:
        print(f"⚠️  Failed to update comparison metrics")
    else:
        print(f"✅ Comparison metrics updated")


def create_comparison_plot() -> None:
    """Create comparison plot showing x160 vs larger datasets."""
    print(f"\n{'='*80}")
    print("Creating comparison plot")
    print(f"{'='*80}\n")
    
    # Read comparison metrics
    metrics_csv = PLOTS_DIR / "comparison_metrics.csv"
    if not metrics_csv.exists():
        print(f"❌ Comparison metrics not found: {metrics_csv}")
        return
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(metrics_csv)
    
    # Filter to relevant models
    relevant = df[df['dataset'].str.contains('x160_static_32|x240_static_32|x480_static_32|x640_static_32', regex=True, na=False)]
    
    if len(relevant) < 2:
        print(f"⚠️  Not enough models to compare (found {len(relevant)})")
        return
    
    # Extract dataset sizes
    relevant['size'] = relevant['dataset'].str.extract(r'x(\d+)').astype(float) * 1000
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Test Loss
    ax = axes[0]
    ax.plot(relevant['size'], relevant['test_loss'], 'o-', linewidth=2, markersize=8, label='Test Loss')
    ax.axhline(y=relevant[relevant['dataset'] == BASELINE_MODEL]['test_loss'].values[0], 
               color='r', linestyle='--', alpha=0.5, label=f'{BASELINE_MODEL} baseline')
    ax.set_xlabel('Dataset Size (K samples)', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title('Test Loss vs Dataset Size', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Log MAE
    ax = axes[1]
    ax.plot(relevant['size'], relevant['log_mae'], 'o-', linewidth=2, markersize=8, label='Log MAE', color='green')
    ax.axhline(y=relevant[relevant['dataset'] == BASELINE_MODEL]['log_mae'].values[0], 
               color='r', linestyle='--', alpha=0.5, label=f'{BASELINE_MODEL} baseline')
    ax.set_xlabel('Dataset Size (K samples)', fontsize=12)
    ax.set_ylabel('Log MAE', fontsize=12)
    ax.set_title('Log MAE vs Dataset Size', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Log R²
    ax = axes[2]
    ax.plot(relevant['size'], relevant['log_r2'], 'o-', linewidth=2, markersize=8, label='Log R²', color='purple')
    ax.axhline(y=relevant[relevant['dataset'] == BASELINE_MODEL]['log_r2'].values[0], 
               color='r', linestyle='--', alpha=0.5, label=f'{BASELINE_MODEL} baseline')
    ax.set_xlabel('Dataset Size (K samples)', fontsize=12)
    ax.set_ylabel('Log R²', fontsize=12)
    ax.set_title('Log R² vs Dataset Size', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "large_dataset_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plot saved to {output_path}")


def print_summary(datasets: List[str]) -> None:
    """Print summary of results."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    import pandas as pd
    metrics_csv = PLOTS_DIR / "comparison_metrics.csv"
    
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        relevant = df[df['dataset'].str.contains('|'.join([BASELINE_MODEL] + [f'{d}_static_32' for d in datasets]), regex=True, na=False)]
        
        print("Model Performance Comparison:")
        print("-" * 80)
        print(f"{'Model':<25} {'Test Loss':<15} {'Log MAE':<15} {'Log R²':<15}")
        print("-" * 80)
        
        baseline = relevant[relevant['dataset'] == BASELINE_MODEL]
        if len(baseline) > 0:
            bl = baseline.iloc[0]
            print(f"{BASELINE_MODEL:<25} {bl['test_loss']:<15.6f} {bl['log_mae']:<15.6f} {bl['log_r2']:<15.6f} [BASELINE]")
        
        for tag in datasets:
            model_tag = f"{tag}_static_32"
            model_data = relevant[relevant['dataset'] == model_tag]
            if len(model_data) > 0:
                md = model_data.iloc[0]
                improvement = ""
                if len(baseline) > 0:
                    loss_improve = (bl['test_loss'] - md['test_loss']) / bl['test_loss'] * 100
                    mae_improve = (bl['log_mae'] - md['log_mae']) / bl['log_mae'] * 100
                    if loss_improve > 0 or mae_improve > 0:
                        improvement = f" [↑ {loss_improve:.1f}% loss, {mae_improve:.1f}% MAE]"
                    else:
                        improvement = f" [↓ {abs(loss_improve):.1f}% loss, {abs(mae_improve):.1f}% MAE]"
                print(f"{model_tag:<25} {md['test_loss']:<15.6f} {md['log_mae']:<15.6f} {md['log_r2']:<15.6f}{improvement}")
        
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Dataset tags to train (default: {DEFAULT_DATASETS})"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only run diagnostics and comparison"
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip diagnostics, only train models"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("LARGE DATASET TRAINING PIPELINE")
    print("="*80)
    print(f"Datasets to process: {args.datasets}")
    print(f"Baseline model: {BASELINE_MODEL}")
    print(f"Skip training: {args.skip_training}")
    print(f"Skip diagnostics: {args.skip_diagnostics}")
    print("="*80)
    
    # Check baseline exists
    if not BASELINE_RUN_DIR.exists():
        print(f"⚠️  Baseline model directory not found: {BASELINE_RUN_DIR}")
        print("   Comparison may not be accurate")
    
    # Train models
    trained_runs = []
    if not args.skip_training:
        for tag in args.datasets:
            run_dir = train_model(tag)
            if run_dir:
                trained_runs.append(run_dir)
    else:
        # Find existing run directories
        for tag in args.datasets:
            run_dir = BASE_DIR / "results" / "runs" / f"runs_autoencoder_{tag}_static_32"
            if run_dir.exists():
                trained_runs.append(run_dir)
    
    # Run diagnostics
    if not args.skip_diagnostics:
        for run_dir in trained_runs:
            run_diagnostics(run_dir)
    
    # Update comparison metrics
    update_comparison_metrics()
    
    # Create comparison plot
    create_comparison_plot()
    
    # Print summary
    print_summary(args.datasets)
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import os
    main()
