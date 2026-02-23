#!/usr/bin/env python3
"""
plot_consistent_runs.py
=======================

Generate comprehensive plots for models trained with consistent architecture.
This ensures fair comparison across dataset sizes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Consistent run tags
# Use optimal_retrained runs (the new standard - constant 320K increments)
OPTIMAL_RETRAINED_RUNS = [
    "x160_optimal_retrained",
    "x480_optimal_retrained",
    "x800_optimal_retrained",
    "x1120_optimal_retrained",
    "x1440_optimal_retrained",
    "x1760_optimal_retrained",
    "x2080_optimal_retrained",
    "x2400_optimal_retrained",
    "x2720_optimal_retrained",
    "x3040_optimal_retrained",
    "x3360_optimal_retrained",
    "x3680_optimal_retrained",
    "x4000_optimal_retrained",
]

# Fallback to consistent runs
CONSISTENT_RUNS = [
    "x160_static_32_consistent",
    "x240_static_32_consistent", 
    "x480_static_32_consistent",
    "x640_static_32_consistent",
]

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "results" / "runs"


def load_loss_history(run_tag: str) -> pd.DataFrame | None:
    """Load loss_history.csv from a run directory."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    history_path = run_dir / "loss_history.csv"
    if not history_path.exists():
        return None
    return pd.read_csv(history_path)


def load_summary(run_tag: str) -> dict | None:
    """Load summary.json from a run directory."""
    import json
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def plot_loss_curves(output_path: Path):
    """Plot training and validation loss curves for all optimal_retrained runs."""
    # Check which runs exist
    runs_to_plot = []
    for run_tag in OPTIMAL_RETRAINED_RUNS:
        if (RUNS_DIR / f"runs_autoencoder_{run_tag}").exists():
            runs_to_plot.append(run_tag)
    
    if not runs_to_plot:
        # Fallback to consistent runs
        runs_to_plot = [r for r in CONSISTENT_RUNS if (RUNS_DIR / f"runs_autoencoder_{r}").exists()]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs_to_plot)))
    
    for idx, run_tag in enumerate(runs_to_plot):
        history = load_loss_history(run_tag)
        if history is None:
            print(f"Warning: No loss history for {run_tag}")
            continue
        
        size = run_tag.split('_')[0].replace('x', '')
        color = colors[idx]
        
        # Plot 1: Training & Validation Loss
        axes[0, 0].plot(history["epoch"], history["train_loss"], 
                       color=color, linestyle="-", linewidth=2, 
                       label=f"x{size}K train")
        axes[0, 0].plot(history["epoch"], history["val_loss"], 
                       color=color, linestyle="--", linewidth=2, 
                       label=f"x{size}K val")
        
        # Plot 2: Validation Log MAE
        if "val_log_mae" in history.columns:
            axes[0, 1].plot(history["epoch"], history["val_log_mae"], 
                           color=color, linewidth=2, label=f"x{size}K")
        
        # Plot 3: Training Loss Only (for comparison)
        axes[1, 0].plot(history["epoch"], history["train_loss"], 
                       color=color, linewidth=2, label=f"x{size}K")
        
        # Plot 4: Validation Loss Only (for comparison)
        axes[1, 1].plot(history["epoch"], history["val_loss"], 
                       color=color, linewidth=2, label=f"x{size}K")
    
    # Configure axes
    axes[0, 0].set_xlabel("Epoch", fontsize=12)
    axes[0, 0].set_ylabel("Loss (log_ratio)", fontsize=12)
    axes[0, 0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=9, loc="best")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel("Epoch", fontsize=12)
    axes[0, 1].set_ylabel("Validation Log MAE", fontsize=12)
    axes[0, 1].set_title("Validation Log MAE", fontsize=14, fontweight="bold")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(fontsize=9, loc="best")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel("Epoch", fontsize=12)
    axes[1, 0].set_ylabel("Training Loss (log_ratio)", fontsize=12)
    axes[1, 0].set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend(fontsize=9, loc="best")
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel("Epoch", fontsize=12)
    axes[1, 1].set_ylabel("Validation Loss (log_ratio)", fontsize=12)
    axes[1, 1].set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(fontsize=9, loc="best")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved loss curves to {output_path}")
    plt.close()


def plot_performance_vs_size(output_path: Path):
    """Plot performance metrics vs dataset size."""
    # Use optimal_retrained runs if available
    runs_to_use = OPTIMAL_RETRAINED_RUNS
    if not any((RUNS_DIR / f"runs_autoencoder_{tag}").exists() for tag in OPTIMAL_RETRAINED_RUNS):
        runs_to_use = CONSISTENT_RUNS
    
    # Collect metrics from summaries
    metrics = []
    for run_tag in runs_to_use:
        summary = load_summary(run_tag)
        if summary is None:
            continue
        
        size = int(run_tag.split('_')[0].replace('x', ''))
        metrics.append({
            'size': size,
            'test_loss': summary.get('test_loss'),
            'val_loss': summary.get('val_loss'),
            'log_mae': summary.get('val_log_mae') or summary.get('log_mae'),
            'log_r2': summary.get('log_r2'),
        })
    
    df = pd.DataFrame(metrics).sort_values('size')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Test Loss
    axes[0, 0].semilogy(df['size'] / 1000, df['test_loss'], 
                       marker='o', linewidth=2, markersize=10, color='steelblue')
    axes[0, 0].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[0, 0].set_ylabel("Test Loss (log_ratio)", fontsize=12)
    axes[0, 0].set_title("Test Loss vs Dataset Size", fontsize=14, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log MAE - log scale, units: dex
    if 'log_mae' in df.columns and df['log_mae'].notna().any():
        axes[0, 1].semilogy(df['size'] / 1000, df['log_mae'], 
                           marker='o', linewidth=2, markersize=10, color='coral')
    axes[0, 1].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[0, 1].set_ylabel("Log MAE (dex)", fontsize=12)
    axes[0, 1].set_title("Log MAE vs Dataset Size", fontsize=14, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Log R² - linear scale, units: unitless (0-1)
    if 'log_r2' in df.columns and df['log_r2'].notna().any():
        axes[1, 0].plot(df['size'] / 1000, df['log_r2'], 
                       marker='o', linewidth=2, markersize=10, color='green')
    axes[1, 0].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[1, 0].set_ylabel("Log R²", fontsize=12)
    axes[1, 0].set_title("Log R² vs Dataset Size", fontsize=14, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation Loss - log scale, units: log_ratio
    axes[1, 1].semilogy(df['size'] / 1000, df['val_loss'], 
                       marker='o', linewidth=2, markersize=10, color='purple')
    axes[1, 1].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[1, 1].set_ylabel("Validation Loss (log_ratio)", fontsize=12)
    axes[1, 1].set_title("Validation Loss vs Dataset Size", fontsize=14, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved performance vs size plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot consistent architecture runs")
    parser.add_argument("--output-dir", type=Path, 
                       default=Path(__file__).resolve().parent.parent / "plots",
                       help="Output directory for plots")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GENERATING PLOTS FOR CONSISTENT ARCHITECTURE RUNS")
    print("="*80)
    
    # Plot loss curves
    print("\n1. Generating loss curves...")
    plot_loss_curves(args.output_dir / "loss_curves_consistent.png")
    
    # Plot performance vs size
    print("\n2. Generating performance vs size plot...")
    plot_performance_vs_size(args.output_dir / "performance_vs_size_consistent.png")
    
    print("\n" + "="*80)
    print("✅ All plots generated!")
    print("="*80)


if __name__ == "__main__":
    main()
