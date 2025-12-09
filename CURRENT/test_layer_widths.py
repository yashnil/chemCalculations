#!/usr/bin/env python3
"""
test_layer_widths.py
====================

Test different layer widths (256, 512, 768, 1024) with 3 and 4 layers.
Uses the best latent dimension from the previous study (192).

Usage:
    python test_layer_widths.py
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
BEST_LATENT_DIM = 192  # From latent dim study
DEFAULT_EPOCHS = 50  # Fewer epochs for faster experimentation
DEFAULT_WIDTHS = [256, 512, 768, 1024]
DEFAULT_LAYERS = [3, 4]


def train_model(layer_width: int, num_layers: int, latent_dim: int = BEST_LATENT_DIM, epochs: int = DEFAULT_EPOCHS) -> dict:
    """Train a model with specific layer width and number of layers."""
    run_dir = BASE_DIR / f"runs_autoencoder_width{layer_width}_layers{num_layers}"
    
    print(f"\n{'='*80}")
    print(f"Training: width={layer_width}, layers={num_layers}, latent_dim={latent_dim}")
    print(f"{'='*80}")
    
    # Read original file
    train_script = BASE_DIR / "train_autoencoder.py"
    original_content = train_script.read_text()
    
    # Create layer configuration
    hidden_layers = [layer_width] * num_layers
    layers_str = str(hidden_layers)
    
    # Modify configuration
    modified_content = original_content
    
    # Update LATENT_DIM
    modified_content = modified_content.replace(
        f"LATENT_DIM = 96",
        f"LATENT_DIM = {latent_dim}"
    )
    
    # Update ENCODER_HIDDEN, DYNAMICS_HIDDEN, DECODER_HIDDEN
    # Find and replace each one
    import re
    
    # Replace ENCODER_HIDDEN
    modified_content = re.sub(
        r'ENCODER_HIDDEN = \[.*?\]',
        f'ENCODER_HIDDEN = {layers_str}',
        modified_content
    )
    
    # Replace DYNAMICS_HIDDEN
    modified_content = re.sub(
        r'DYNAMICS_HIDDEN = \[.*?\]',
        f'DYNAMICS_HIDDEN = {layers_str}',
        modified_content
    )
    
    # Replace DECODER_HIDDEN
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
            "CSV_PATH": str(BASE_DIR / "datasets" / "all_gas_fastchem_x160.csv"),
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
            print(f"❌ Training failed for width={layer_width}, layers={num_layers}")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return None
        
        # Read results from summary.json
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            print(f"⚠️ Summary not found for width={layer_width}, layers={num_layers}")
            return None
        
        summary = json.loads(summary_path.read_text())
        
        results = {
            "layer_width": layer_width,
            "num_layers": num_layers,
            "latent_dim": latent_dim,
            "test_loss": summary.get("test_loss"),
            "val_loss": summary.get("val_loss"),
            "test_mae_linear": summary.get("test_mae_linear"),
            "training_time": elapsed,
        }
        
        print(f"✅ Completed: test_loss={results['test_loss']:.6f}, time={elapsed/60:.1f}min")
        return results
        
    finally:
        # Restore original file
        train_script.write_text(original_content)


def plot_results(results: List[dict], output_path: Path):
    """Plot layer width vs loss for different numbers of layers."""
    if not results:
        print("❌ No results to plot!")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values(["num_layers", "layer_width"])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Test Loss vs Layer Width (separate lines for 3 vs 4 layers)
    for num_layers in sorted(df["num_layers"].unique()):
        subset = df[df["num_layers"] == num_layers].sort_values("layer_width")
        axes[0].plot(
            subset["layer_width"], 
            subset["test_loss"],
            marker="o", 
            linewidth=2, 
            markersize=8, 
            label=f"{num_layers} layers"
        )
    
    axes[0].set_xlabel("Layer Width", fontsize=12)
    axes[0].set_ylabel("Test Loss (Normalized)", fontsize=12)
    axes[0].set_title("Test Loss vs Layer Width", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(left=0)
    
    # Plot 2: Validation Loss vs Layer Width
    for num_layers in sorted(df["num_layers"].unique()):
        subset = df[df["num_layers"] == num_layers].sort_values("layer_width")
        axes[1].plot(
            subset["layer_width"], 
            subset["val_loss"],
            marker="s", 
            linewidth=2, 
            markersize=8, 
            label=f"{num_layers} layers"
        )
    
    axes[1].set_xlabel("Layer Width", fontsize=12)
    axes[1].set_ylabel("Validation Loss (Normalized)", fontsize=12)
    axes[1].set_title("Validation Loss vs Layer Width", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test different layer widths")
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=DEFAULT_WIDTHS,
        help=f"Layer widths to test (default: {DEFAULT_WIDTHS})"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help=f"Number of layers to test (default: {DEFAULT_LAYERS})"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=BEST_LATENT_DIM,
        help=f"Latent dimension to use (default: {BEST_LATENT_DIM} from previous study)"
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
        default=BASE_DIR / "layer_width_study.png",
        help="Output path for plot"
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=BASE_DIR / "layer_width_results.csv",
        help="Output path for CSV results"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("LAYER WIDTH STUDY")
    print("="*80)
    print(f"Testing layer widths: {args.widths}")
    print(f"Testing number of layers: {args.layers}")
    print(f"Using latent_dim: {args.latent_dim} (best from previous study)")
    print(f"Epochs per model: {args.epochs}")
    total_models = len(args.widths) * len(args.layers)
    print(f"Total models to train: {total_models}")
    print(f"Estimated total time: ~{total_models * args.epochs * 6.5 / 60:.1f} minutes")
    print("="*80)
    
    results = []
    for num_layers in args.layers:
        for layer_width in args.widths:
            result = train_model(layer_width, num_layers, args.latent_dim, args.epochs)
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
    print(f"\n🏆 Best model: width={best['layer_width']}, layers={best['num_layers']}, test_loss={best['test_loss']:.6f}")
    
    # Print summary by number of layers
    print("\n" + "="*80)
    print("SUMMARY BY NUMBER OF LAYERS")
    print("="*80)
    for num_layers in sorted(df["num_layers"].unique()):
        subset = df[df["num_layers"] == num_layers]
        best_subset = subset.loc[subset["test_loss"].idxmin()]
        print(f"\n{num_layers} layers:")
        print(subset[["layer_width", "test_loss", "val_loss"]].to_string(index=False))
        print(f"  Best: width={best_subset['layer_width']}, test_loss={best_subset['test_loss']:.6f}")


if __name__ == "__main__":
    main()

