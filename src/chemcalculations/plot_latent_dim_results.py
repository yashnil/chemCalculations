#!/usr/bin/env python3
"""
plot_latent_dim_results.py
===========================

Generate plot from completed latent dimension test results.
Reads summary.json files from run directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Legacy study outputs: run dirs under ``src/`` (see training scripts)
BASE_DIR = Path(__file__).resolve().parent.parent


def collect_results(latent_dims: list[int]) -> list[dict]:
    """Collect results from completed training runs."""
    results = []
    
    for dim in latent_dims:
        run_dir = BASE_DIR / f"runs_autoencoder_latent{dim}"
        summary_path = run_dir / "summary.json"
        
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            
            results.append({
                "latent_dim": dim,
                "test_loss": summary.get("test_loss"),
                "val_loss": summary.get("val_loss"),
                "test_mae_linear": summary.get("test_mae_linear"),
            })
            print(f"✅ Found results for latent_dim={dim}: test_loss={summary.get('test_loss'):.6f}")
        else:
            print(f"⏳ latent_dim={dim}: Still training or not started")
    
    return results


def plot_results(results: list[dict], output_path: Path):
    """Plot latent dimension vs loss."""
    if not results:
        print("❌ No results to plot!")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values("latent_dim")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Test Loss vs Latent Dim
    axes[0].plot(df["latent_dim"], df["test_loss"], 
                marker="o", linewidth=2, markersize=10, color="steelblue", label="Test Loss")
    axes[0].set_xlabel("Latent Dimension", fontsize=12)
    axes[0].set_ylabel("Test Loss (Normalized)", fontsize=12)
    axes[0].set_title("Test Loss vs Latent Dimension", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(left=0)
    axes[0].legend()
    
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
                marker="s", linewidth=2, markersize=10, color="coral", label="Validation Loss")
    axes[1].set_xlabel("Latent Dimension", fontsize=12)
    axes[1].set_ylabel("Validation Loss (Normalized)", fontsize=12)
    axes[1].set_title("Validation Loss vs Latent Dimension", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(left=0)
    axes[1].legend()
    
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
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    if len(df) > 0:
        best = df.loc[df["test_loss"].idxmin()]
        print(f"\n🏆 Best model: latent_dim={best['latent_dim']}, test_loss={best['test_loss']:.6f}")


def main():
    latent_dims = [64, 96, 128, 160, 192]
    output_path = BASE_DIR / "latent_dim_study.png"
    
    print("="*80)
    print("LATENT DIMENSION STUDY - PLOT GENERATION")
    print("="*80)
    print(f"Checking for results: {latent_dims}")
    print("="*80)
    
    results = collect_results(latent_dims)
    
    if results:
        plot_results(results, output_path)
    else:
        print("\n❌ No completed results found. Please wait for training to complete.")


if __name__ == "__main__":
    main()



