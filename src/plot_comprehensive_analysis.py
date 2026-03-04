#!/usr/bin/env python3
"""
plot_comprehensive_analysis.py
===============================

Generate comprehensive plots for consistent architecture runs:
1. Loss curves for all sizes
2. Performance vs dataset size (with asymptote analysis)
3. Convergence analysis
4. Model comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "results" / "runs"
COMPARISON_CSV = BASE_DIR / "plots" / "comparison_metrics.csv"

# Use optimal_retrained runs (the new standard - constant 320K increments)
OPTIMAL_RETRAINED_RUNS = [
    ("x800_optimal_retrained", 800000),
    ("x1600_optimal_retrained", 1600000),
    ("x2400_optimal_retrained", 2400000),
    ("x3200_optimal_retrained", 3200000),
    ("x4000_optimal_retrained", 4000000),
    ("x4800_optimal_retrained", 4800000),
]

# Fallback to consistent runs if optimal_retrained don't exist
CONSISTENT_RUNS = [
    ("x160_static_32_consistent", 160000),
    ("x240_static_32_consistent", 240000),
    ("x480_static_32_consistent", 480000),
    ("x640_static_32_consistent", 640000),
]

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


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


def plot_loss_curves_all_sizes(output_path: Path):
    """Plot training and validation loss curves for all optimal_retrained runs."""
    # Use optimal_retrained if available, otherwise fallback to consistent
    runs_to_plot = OPTIMAL_RETRAINED_RUNS
    # Check if optimal_retrained runs exist
    if not any((RUNS_DIR / f"runs_autoencoder_{tag}").exists() for tag, _ in OPTIMAL_RETRAINED_RUNS):
        runs_to_plot = CONSISTENT_RUNS
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs_to_plot)))
    
    for idx, (run_tag, size_k) in enumerate(runs_to_plot):
        history = load_loss_history(run_tag)
        if history is None:
            print(f"Warning: No loss history for {run_tag}")
            continue
        
        size = run_tag.split('_')[0].replace('x', '')
        color = colors[idx]
        label = f"x{size}K"
        
        # Plot 1: Training & Validation Loss
        axes[0, 0].plot(history["epoch"], history["train_loss"], 
                       color=color, linestyle="-", linewidth=2, 
                       label=f"{label} train")
        axes[0, 0].plot(history["epoch"], history["val_loss"], 
                       color=color, linestyle="--", linewidth=2, 
                       label=f"{label} val")
        
        # Plot 2: Validation Log MAE
        if "val_log_mae" in history.columns:
            axes[0, 1].plot(history["epoch"], history["val_log_mae"], 
                           color=color, linewidth=2, label=label)
        
        # Plot 3: Training Loss Only
        axes[1, 0].plot(history["epoch"], history["train_loss"], 
                       color=color, linewidth=2, label=label)
        
        # Plot 4: Validation Loss Only
        axes[1, 1].plot(history["epoch"], history["val_loss"], 
                       color=color, linewidth=2, label=label)
    
    # Configure axes
    for ax in axes.flat:
        ax.set_xlabel("Epoch", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    
    axes[0, 0].set_ylabel("Loss (log_ratio)", fontsize=12)
    axes[0, 0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0, 0].set_yscale("log")
    
    axes[0, 1].set_ylabel("Validation Log MAE", fontsize=12)
    axes[0, 1].set_title("Validation Log MAE", fontsize=14, fontweight="bold")
    axes[0, 1].set_yscale("log")
    
    axes[1, 0].set_ylabel("Training Loss (log_ratio)", fontsize=12)
    axes[1, 0].set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
    axes[1, 0].set_yscale("log")
    
    axes[1, 1].set_ylabel("Validation Loss (log_ratio)", fontsize=12)
    axes[1, 1].set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    axes[1, 1].set_yscale("log")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved loss curves to {output_path}")
    plt.close()


def plot_performance_vs_size_with_asymptote(output_path: Path):
    """Plot performance vs dataset size with asymptote analysis."""
    df = pd.read_csv(COMPARISON_CSV)
    
    # Prioritize optimal_retrained runs, fallback to consistent
    df_optimal = df[df["dataset"].str.contains("_optimal_retrained", na=False)].copy()
    if df_optimal.empty:
        df_optimal = df[df["dataset"].str.contains("_consistent", na=False)].copy()
    df_optimal = df_optimal.sort_values("total_samples")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Test Loss - log scale, units: log_ratio
    valid_test = ~df_optimal["test_loss"].isna() & (df_optimal["test_loss"] > 0)
    if valid_test.sum() > 0:
        axes[0, 0].semilogy(df_optimal[valid_test]["total_samples"] / 1000, 
                           df_optimal[valid_test]["test_loss"], 
                           marker='o', linewidth=2, markersize=10, color='steelblue', label='Optimal architecture')
        # Add trend line
        x = df_optimal[valid_test]["total_samples"] / 1000
        y = df_optimal[valid_test]["test_loss"]
        z = np.polyfit(x, np.log10(y), 1)
        p = np.poly1d(z)
        axes[0, 0].plot(x, 10**p(x), '--', alpha=0.5, color='gray', label='Trend')
    axes[0, 0].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[0, 0].set_ylabel("Test Loss (log_ratio)", fontsize=12)
    axes[0, 0].set_title("Test Loss vs Dataset Size", fontsize=14, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Log MAE - log scale, units: dex
    valid_mae = ~df_optimal["log_mae"].isna() & (df_optimal["log_mae"] > 0)
    if valid_mae.sum() > 0:
        axes[0, 1].semilogy(df_optimal[valid_mae]["total_samples"] / 1000, 
                           df_optimal[valid_mae]["log_mae"], 
                           marker='o', linewidth=2, markersize=10, color='coral')
    axes[0, 1].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[0, 1].set_ylabel("Log MAE (dex)", fontsize=12)
    axes[0, 1].set_title("Log MAE vs Dataset Size", fontsize=14, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Log R² - linear scale, units: unitless (0-1)
    valid_r2 = ~df_optimal["log_r2"].isna()
    if valid_r2.sum() > 0:
        axes[1, 0].plot(df_optimal[valid_r2]["total_samples"] / 1000, 
                       df_optimal[valid_r2]["log_r2"], 
                       marker='o', linewidth=2, markersize=10, color='green')
    axes[1, 0].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[1, 0].set_ylabel("Log R²", fontsize=12)
    axes[1, 0].set_title("Log R² vs Dataset Size", fontsize=14, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation Loss - log scale, units: log_ratio
    valid_val = ~df_optimal["val_loss"].isna() & (df_optimal["val_loss"] > 0)
    if valid_val.sum() > 0:
        axes[1, 1].semilogy(df_optimal[valid_val]["total_samples"] / 1000, 
                           df_optimal[valid_val]["val_loss"], 
                           marker='o', linewidth=2, markersize=10, color='purple')
    axes[1, 1].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[1, 1].set_ylabel("Validation Loss (log_ratio)", fontsize=12)
    axes[1, 1].set_title("Validation Loss vs Dataset Size", fontsize=14, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved performance vs size plot to {output_path}")
    plt.close()


def plot_asymptote_analysis(output_path: Path):
    """Zoomed view of large dataset sizes to check for asymptoting."""
    df = pd.read_csv(COMPARISON_CSV)
    # Prioritize optimal_retrained runs
    df_optimal = df[df["dataset"].str.contains("_optimal_retrained", na=False)].copy()
    if df_optimal.empty:
        df_optimal = df[df["dataset"].str.contains("_consistent", na=False)].copy()
    df_optimal = df_optimal.sort_values("total_samples")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Test Loss - zoomed, log scale, units: log_ratio
    valid = ~df_optimal["test_loss"].isna() & (df_optimal["test_loss"] > 0)
    if valid.sum() > 0:
        x = df_optimal[valid]["total_samples"] / 1000
        y = df_optimal[valid]["test_loss"]
        
        axes[0].semilogy(x, y, marker='o', linewidth=2, markersize=12, color='steelblue')
        axes[0].set_xlabel("Training Samples (×1000)", fontsize=12)
        axes[0].set_ylabel("Test Loss (log_ratio)", fontsize=12)
        axes[0].set_title("Test Loss: Asymptote Analysis", fontsize=14, fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        
        # Add annotations
        for _, row in df_optimal[valid].iterrows():
            size = row["total_samples"] / 1000
            loss = row["test_loss"]
            axes[0].annotate(f"{size:.0f}K", (size, loss), 
                           textcoords="offset points", xytext=(0, 10), fontsize=9)
    
    # Validation Loss - zoomed, log scale, units: log_ratio
    valid_val = ~df_optimal["val_loss"].isna() & (df_optimal["val_loss"] > 0)
    if valid_val.sum() > 0:
        x = df_optimal[valid_val]["total_samples"] / 1000
        y = df_optimal[valid_val]["val_loss"]
        
        axes[1].semilogy(x, y, marker='o', linewidth=2, markersize=12, color='purple')
        axes[1].set_xlabel("Training Samples (×1000)", fontsize=12)
        axes[1].set_ylabel("Validation Loss (log_ratio)", fontsize=12)
        axes[1].set_title("Validation Loss: Asymptote Analysis", fontsize=14, fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        
        # Add annotations
        for _, row in df_optimal[valid_val].iterrows():
            size = row["total_samples"] / 1000
            loss = row["val_loss"]
            axes[1].annotate(f"{size:.0f}K", (size, loss), 
                           textcoords="offset points", xytext=(0, 10), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved asymptote analysis to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive analysis plots")
    parser.add_argument("--output-dir", type=Path, 
                       default=BASE_DIR / "plots",
                       help="Output directory for plots")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GENERATING COMPREHENSIVE ANALYSIS PLOTS")
    print("="*80)
    
    # Plot 1: Loss curves for all sizes
    print("\n1. Generating loss curves for all sizes...")
    plot_loss_curves_all_sizes(args.output_dir / "loss_curves_all_sizes.png")
    
    # Plot 2: Performance vs size with asymptote analysis
    print("\n2. Generating performance vs size plot...")
    plot_performance_vs_size_with_asymptote(args.output_dir / "performance_vs_size_comprehensive.png")
    
    # Plot 3: Asymptote analysis
    print("\n3. Generating asymptote analysis...")
    plot_asymptote_analysis(args.output_dir / "asymptote_analysis.png")
    
    print("\n" + "="*80)
    print("✅ All comprehensive plots generated!")
    print("="*80)


if __name__ == "__main__":
    main()
