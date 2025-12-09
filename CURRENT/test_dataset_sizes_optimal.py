#!/usr/bin/env python3
"""
test_dataset_sizes_optimal.py
==============================

Test different dataset sizes with the optimal hyperparameters from previous studies:
- latent_dim: 192 (best from latent dim study)
- layer_width: 512 (best from layer width study, assuming it stays best)
- num_layers: 3 (best from layer width study, assuming it stays best)

Usage:
    python test_dataset_sizes_optimal.py
    python test_dataset_sizes_optimal.py --datasets x64 x96 x128 x160
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Optimal hyperparameters from previous studies
OPTIMAL_LATENT_DIM = 192
OPTIMAL_LAYER_WIDTH = 512
OPTIMAL_NUM_LAYERS = 3

# Default dataset sizes to test (matching comparison_metrics.csv)
DEFAULT_DATASETS = ["x32", "x48", "x64", "x80", "x96", "x112", "x128", "x144", "x160", "x176"]
DEFAULT_EPOCHS = 200  # Full training for final study


def train_model(dataset_tag: str, latent_dim: int, layer_width: int, num_layers: int, epochs: int = DEFAULT_EPOCHS) -> dict:
    """Train a model with optimal hyperparameters on a specific dataset size."""
    run_dir = BASE_DIR / f"runs_autoencoder_optimal_{dataset_tag}"
    
    print(f"\n{'='*80}")
    print(f"Training optimal model on {dataset_tag}")
    print(f"  latent_dim={latent_dim}, width={layer_width}, layers={num_layers}")
    print(f"{'='*80}")
    
    # Read original file
    train_script = BASE_DIR / "train_autoencoder.py"
    original_content = train_script.read_text()
    
    # Create layer configuration
    hidden_layers = [layer_width] * num_layers
    layers_str = str(hidden_layers)
    
    # Modify configuration
    import re
    
    # Update LATENT_DIM
    modified_content = original_content.replace(
        f"LATENT_DIM = 96",
        f"LATENT_DIM = {latent_dim}"
    )
    
    # Update ENCODER_HIDDEN, DYNAMICS_HIDDEN, DECODER_HIDDEN
    modified_content = re.sub(
        r'ENCODER_HIDDEN = \[.*?\]',
        f'ENCODER_HIDDEN = {layers_str}',
        modified_content
    )
    
    modified_content = re.sub(
        r'DYNAMICS_HIDDEN = \[.*?\]',
        f'DYNAMICS_HIDDEN = {layers_str}',
        modified_content
    )
    
    modified_content = re.sub(
        r'DECODER_HIDDEN = \[.*?\]',
        f'DECODER_HIDDEN = {layers_str}',
        modified_content
    )
    
    # Update EPOCHS if needed
    if epochs != 200:
        modified_content = modified_content.replace(
            f"EPOCHS = 200",
            f"EPOCHS = {epochs}"
        )
    
    # Write modified version
    train_script.write_text(modified_content)
    
    try:
        # Run training
        env = {
            "CSV_PATH": str(BASE_DIR / "datasets" / f"all_gas_fastchem_{dataset_tag}.csv"),
        }
        
        cmd = [
            sys.executable,
            str(train_script),
            "--loss-type", "huber",
            "--run-dir", str(run_dir)
        ]
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            env=env,
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"❌ Training failed for {dataset_tag}")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return None
        
        # Read results from summary.json
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            print(f"⚠️ Summary not found for {dataset_tag}")
            return None
        
        summary = json.loads(summary_path.read_text())
        
        # Run diagnostics to get log_mae and log_r2
        log_mae = None
        log_r2 = None
        
        best_model_path = run_dir / "best_model.py"
        csv_path = BASE_DIR / "datasets" / f"all_gas_fastchem_{dataset_tag}.csv"
        if best_model_path.exists() and csv_path.exists():
            diag_env = {
                "CSV_PATH": str(csv_path),
                "BEST_MODULE": str(best_model_path),
                "OUT_DIR": str(run_dir / "diagnostics"),
            }
            diag_result = subprocess.run(
                [sys.executable, str(BASE_DIR / "diagnostics.py")],
                env=diag_env,
                cwd=BASE_DIR,
                capture_output=True,
                text=True
            )
            
            # Read diagnostics results
            diag_path = run_dir / "diagnostics" / "global_metrics.txt"
            if diag_path.exists():
                for line in diag_path.read_text().splitlines():
                    if "Log MAE" in line and ":" in line:
                        try:
                            log_mae = float(line.split(":")[1].strip().split()[0].replace(",", ""))
                        except:
                            pass
                    if "Log R" in line and ":" in line:
                        try:
                            log_r2 = float(line.split(":")[1].strip().split()[0].replace(",", ""))
                        except:
                            pass
        
        results = {
            "dataset": dataset_tag,
            "total_samples": summary.get("train_samples", 0) + summary.get("val_samples", 0) + summary.get("test_samples", 0),
            "test_loss": summary.get("test_loss"),
            "val_loss": summary.get("val_loss"),
            "test_mae_linear": summary.get("test_mae_linear"),
            "log_mae": log_mae,
            "log_r2": log_r2,
            "training_time": elapsed,
        }
        
        print(f"✅ Completed: test_loss={results['test_loss']:.6f}, log_mae={log_mae}, time={elapsed/60:.1f}min")
        return results
        
    finally:
        # Restore original file
        train_script.write_text(original_content)


def plot_results(results: List[dict], output_path: Path):
    """Plot dataset size vs performance metrics."""
    if not results:
        print("❌ No results to plot!")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values("total_samples")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Test Loss vs Dataset Size
    axes[0, 0].plot(df["total_samples"] / 1000, df["test_loss"], 
                   marker="o", linewidth=2, markersize=8, color="steelblue")
    axes[0, 0].set_xlabel("Dataset Size (×1000 samples)", fontsize=12)
    axes[0, 0].set_ylabel("Test Loss (Normalized)", fontsize=12)
    axes[0, 0].set_title("Test Loss vs Dataset Size (Optimal Hyperparameters)", fontsize=14, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Log MAE vs Dataset Size
    if "log_mae" in df.columns and df["log_mae"].notna().any():
        valid = df["log_mae"].notna()
        axes[0, 1].plot(df[valid]["total_samples"] / 1000, df[valid]["log_mae"], 
                        marker="s", linewidth=2, markersize=8, color="coral")
        axes[0, 1].set_xlabel("Dataset Size (×1000 samples)", fontsize=12)
        axes[0, 1].set_ylabel("Log MAE", fontsize=12)
        axes[0, 1].set_title("Log MAE vs Dataset Size", fontsize=14, fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Log R² vs Dataset Size
    if "log_r2" in df.columns and df["log_r2"].notna().any():
        valid = df["log_r2"].notna()
        axes[1, 0].plot(df[valid]["total_samples"] / 1000, df[valid]["log_r2"], 
                       marker="^", linewidth=2, markersize=8, color="green")
        axes[1, 0].set_xlabel("Dataset Size (×1000 samples)", fontsize=12)
        axes[1, 0].set_ylabel("Log R²", fontsize=12)
        axes[1, 0].set_title("Log R² vs Dataset Size", fontsize=14, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0.95, 1.0])
    
    # Plot 4: Validation Loss vs Dataset Size
    axes[1, 1].plot(df["total_samples"] / 1000, df["val_loss"], 
                   marker="d", linewidth=2, markersize=8, color="purple")
    axes[1, 1].set_xlabel("Dataset Size (×1000 samples)", fontsize=12)
    axes[1, 1].set_ylabel("Validation Loss (Normalized)", fontsize=12)
    axes[1, 1].set_title("Validation Loss vs Dataset Size", fontsize=14, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test different dataset sizes with optimal hyperparameters")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Dataset tags to test (default: {DEFAULT_DATASETS})"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=OPTIMAL_LATENT_DIM,
        help=f"Latent dimension (default: {OPTIMAL_LATENT_DIM})"
    )
    parser.add_argument(
        "--layer-width",
        type=int,
        default=OPTIMAL_LAYER_WIDTH,
        help=f"Layer width (default: {OPTIMAL_LAYER_WIDTH})"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=OPTIMAL_NUM_LAYERS,
        help=f"Number of layers (default: {OPTIMAL_NUM_LAYERS})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of epochs per model (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "dataset_size_study_optimal.png",
        help="Output path for plot"
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=BASE_DIR / "dataset_size_results_optimal.csv",
        help="Output path for CSV results"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("DATASET SIZE STUDY - OPTIMAL HYPERPARAMETERS")
    print("="*80)
    print(f"Optimal hyperparameters:")
    print(f"  latent_dim: {args.latent_dim}")
    print(f"  layer_width: {args.layer_width}")
    print(f"  num_layers: {args.num_layers}")
    print(f"Testing datasets: {args.datasets}")
    print(f"Epochs per model: {args.epochs}")
    print(f"Total models to train: {len(args.datasets)}")
    print(f"Estimated total time: ~{len(args.datasets) * args.epochs * 7 / 60:.1f} minutes")
    print("="*80)
    
    results = []
    for dataset_tag in args.datasets:
        result = train_model(dataset_tag, args.latent_dim, args.layer_width, args.num_layers, args.epochs)
        if result:
            results.append(result)
        time.sleep(2)  # Brief pause between runs
    
    if not results:
        print("❌ No successful training runs!")
        return
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.csv_output, index=False)
    print(f"\n✅ Results saved to {args.csv_output}")
    
    # Generate plot
    plot_results(results, args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    if len(df) > 0:
        best = df.loc[df["test_loss"].idxmin()]
        print(f"\n🏆 Best model: dataset={best['dataset']}, test_loss={best['test_loss']:.6f}")


if __name__ == "__main__":
    main()

