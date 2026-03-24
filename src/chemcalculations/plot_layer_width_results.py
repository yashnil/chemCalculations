#!/usr/bin/env python3
"""
plot_layer_width_results.py
============================

Generate plot from completed layer width test results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def collect_results(widths: list[int], num_layers_list: list[int]) -> list[dict]:
    """Collect results from completed training runs."""
    results = []
    
    for num_layers in num_layers_list:
        for width in widths:
            run_dir = BASE_DIR / f"runs_autoencoder_width{width}_layers{num_layers}"
            summary_path = run_dir / "summary.json"
            
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                
                results.append({
                    "layer_width": width,
                    "num_layers": num_layers,
                    "test_loss": summary.get("test_loss"),
                    "val_loss": summary.get("val_loss"),
                    "test_mae_linear": summary.get("test_mae_linear"),
                })
                print(f"✅ width={width}, layers={num_layers}: test_loss={summary.get('test_loss'):.6f}")
            else:
                print(f"⏳ width={width}, layers={num_layers}: Still training or not started")
    
    return results


def plot_results(results: list[dict], output_path: Path):
    """Plot layer width vs loss for different numbers of layers."""
    if not results:
        print("❌ No results to plot!")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values(["num_layers", "layer_width"])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Test Loss vs Layer Width
    for num_layers in sorted(df["num_layers"].unique()):
        subset = df[df["num_layers"] == num_layers].sort_values("layer_width")
        axes[0].plot(
            subset["layer_width"], 
            subset["test_loss"],
            marker="o", 
            linewidth=2, 
            markersize=10, 
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
            markersize=10, 
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
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    if len(df) > 0:
        best = df.loc[df["test_loss"].idxmin()]
        print(f"\n🏆 Best model: width={best['layer_width']}, layers={best['num_layers']}, test_loss={best['test_loss']:.6f}")
        
        # Summary by layers
        print("\n" + "="*80)
        print("SUMMARY BY NUMBER OF LAYERS")
        print("="*80)
        for num_layers in sorted(df["num_layers"].unique()):
            subset = df[df["num_layers"] == num_layers]
            best_subset = subset.loc[subset["test_loss"].idxmin()]
            print(f"\n{num_layers} layers - Best: width={best_subset['layer_width']}, test_loss={best_subset['test_loss']:.6f}")


def main():
    widths = [256, 512, 768, 1024]
    num_layers_list = [3, 4]
    output_path = BASE_DIR / "layer_width_study.png"
    
    print("="*80)
    print("LAYER WIDTH STUDY - PLOT GENERATION")
    print("="*80)
    print(f"Checking for results: widths={widths}, layers={num_layers_list}")
    print("="*80)
    
    results = collect_results(widths, num_layers_list)
    
    if results:
        plot_results(results, output_path)
    else:
        print("\n❌ No completed results found. Please wait for training to complete.")


if __name__ == "__main__":
    main()



