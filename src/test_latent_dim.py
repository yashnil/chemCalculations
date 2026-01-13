#!/usr/bin/env python3
"""
test_latent_dim.py
==================

Train models with different latent dimensions and plot the relationship
between latent dimension and test loss.

Usage:
    python test_latent_dim.py --latent-dims 32 64 96 128 160 192 256
    python test_latent_dim.py  # Uses default range
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
DEFAULT_LATENT_DIMS = [64, 96, 128, 160, 192, 256, 320, 384, 448, 512]
DEFAULT_EPOCHS = 50  # Fewer epochs for faster experimentation


def train_model(latent_dim: int, epochs: int = DEFAULT_EPOCHS) -> dict:
    """Train a model with a specific latent dimension and return results."""
    run_dir = BASE_DIR / f"runs_autoencoder_latent{latent_dim}"
    
    print(f"\n{'='*80}")
    print(f"Training model with latent_dim={latent_dim}")
    print(f"{'='*80}")
    
    # Set up environment
    env = {
        "CSV_PATH": str(BASE_DIR.parent / "data" / "datasets" / "all_gas_fastchem_x160.csv"),
    }
    
    # Modify train_autoencoder.py temporarily or use command-line override
    # Since we can't easily override LATENT_DIM, we'll need to modify the file temporarily
    # Actually, let's create a wrapper script or modify the approach
    
    # For now, let's create a modified version that accepts latent_dim as an argument
    # Or we can directly modify the train script temporarily
    
    # Better approach: create a training function we can call directly
    # But train_autoencoder.py doesn't expose latent_dim as CLI arg, so we need to:
    # 1. Temporarily modify LATENT_DIM in train_autoencoder.py
    # 2. Run training
    # 3. Restore original value
    
    # Read original file
    train_script = BASE_DIR / "train_autoencoder.py"
    original_content = train_script.read_text()
    
    # Modify LATENT_DIM - try multiple patterns
    import re
    modified_content = re.sub(
        r'LATENT_DIM = \d+',
        f'LATENT_DIM = {latent_dim}',
        original_content
    )
    
    # Also modify EPOCHS if needed
    if epochs != 200:
        modified_content = modified_content.replace(
            f"EPOCHS = 200",
            f"EPOCHS = {epochs}"
        )
    
    # Write modified version
    train_script.write_text(modified_content)
    
    try:
        # Run training
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
            print(f"❌ Training failed for latent_dim={latent_dim}")
            print(result.stderr)
            return None
        
        # Read results from summary.json
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            print(f"⚠️ Summary not found for latent_dim={latent_dim}")
            return None
        
        summary = json.loads(summary_path.read_text())
        
        results = {
            "latent_dim": latent_dim,
            "test_loss": summary.get("test_loss"),
            "val_loss": summary.get("val_loss"),
            "test_mae_linear": summary.get("test_mae_linear"),
            "train_samples": summary.get("train_samples"),
            "training_time": elapsed,
        }
        
        print(f"✅ Completed: test_loss={results['test_loss']:.6f}, time={elapsed/60:.1f}min")
        return results
        
    finally:
        # Restore original file
        train_script.write_text(original_content)


def plot_results(results: List[dict], output_path: Path):
    """Plot latent dimension vs loss."""
    df = pd.DataFrame(results)
    df = df.sort_values("latent_dim")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Test Loss vs Latent Dim
    axes[0].plot(df["latent_dim"], df["test_loss"], 
                marker="o", linewidth=2, markersize=8, color="steelblue")
    axes[0].set_xlabel("Latent Dimension", fontsize=12)
    axes[0].set_ylabel("Test Loss (Normalized)", fontsize=12)
    axes[0].set_title("Test Loss vs Latent Dimension", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(left=0)
    
    # Add value labels
    for _, row in df.iterrows():
        axes[0].annotate(
            f"{row['test_loss']:.4f}",
            (row["latent_dim"], row["test_loss"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9
        )
    
    # Plot 2: Validation Loss vs Latent Dim
    axes[1].plot(df["latent_dim"], df["val_loss"], 
                marker="s", linewidth=2, markersize=8, color="coral")
    axes[1].set_xlabel("Latent Dimension", fontsize=12)
    axes[1].set_ylabel("Validation Loss (Normalized)", fontsize=12)
    axes[1].set_title("Validation Loss vs Latent Dimension", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(left=0)
    
    # Add value labels
    for _, row in df.iterrows():
        axes[1].annotate(
            f"{row['val_loss']:.4f}",
            (row["latent_dim"], row["val_loss"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test different latent dimensions")
    parser.add_argument(
        "--latent-dims",
        type=int,
        nargs="+",
        default=DEFAULT_LATENT_DIMS,
        help=f"Latent dimensions to test (default: {DEFAULT_LATENT_DIMS})"
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
        default=BASE_DIR.parent / "plots" / "latent_dim_study.png",
        help="Output path for plot"
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=BASE_DIR.parent / "plots" / "latent_dim_results.csv",
        help="Output path for CSV results"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("LATENT DIMENSION STUDY")
    print("="*80)
    print(f"Testing latent dimensions: {args.latent_dims}")
    print(f"Epochs per model: {args.epochs}")
    print(f"Total models to train: {len(args.latent_dims)}")
    print(f"Estimated total time: ~{len(args.latent_dims) * args.epochs * 6.5 / 60:.1f} minutes")
    print("="*80)
    
    results = []
    for latent_dim in args.latent_dims:
        result = train_model(latent_dim, epochs=args.epochs)
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
    
    best = df.loc[df["test_loss"].idxmin()]
    print(f"\n🏆 Best model: latent_dim={best['latent_dim']}, test_loss={best['test_loss']:.6f}")


if __name__ == "__main__":
    main()



