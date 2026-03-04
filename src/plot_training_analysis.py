#!/usr/bin/env python3
"""
plot_training_analysis.py
=========================

Generate comprehensive training analysis plots:
1. Loss curves (train/val) vs epochs for specific runs
2. Performance metrics vs dataset size
3. Model size comparison

Usage:
    python plot_training_analysis.py
    python plot_training_analysis.py --runs x160_new x160_mse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style settings
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def load_run_summary(run_dir: Path) -> Dict[str, Any]:
    """Load summary.json from a run directory."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def load_loss_history(run_dir: Path) -> pd.DataFrame:
    """Load loss_history.csv from a run directory."""
    history_path = run_dir / "loss_history.csv"
    if not history_path.exists():
        return None
    return pd.read_csv(history_path)


def plot_loss_curves(run_tags: List[str], base_dir: Path, output_path: Path, use_consistent: bool = True):
    """Plot training and validation loss curves for specified runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Try multiple base directories as fallback
    # Assume project root is 2 levels up from src/ or results/
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    fallback_dirs = [
        base_dir,
        project_root / "models" / "archive",
        project_root / "results" / "runs",
    ]
    
    for idx, tag in enumerate(run_tags):
        run_dir = None
        # Try multiple possible directory patterns in each base directory
        for search_dir in fallback_dirs:
            patterns = [
                f"runs_autoencoder_{tag}",
                f"runs_autoencoder_optimal_{tag}",
                f"runs_autoencoder_latent{tag}",
            ]
            for pattern in patterns:
                candidate = search_dir / pattern
                if candidate.exists():
                    run_dir = candidate
                    break
            if run_dir:
                break
        
        if run_dir is None:
            print(f"Warning: Run '{tag}' not found in any base directory, skipping...")
            continue
        
        history = load_loss_history(run_dir)
        if history is None:
            print(f"Warning: No loss history for {tag}, skipping...")
            continue
        
        summary = load_run_summary(run_dir)
        loss_type = summary.get("loss_type", "unknown") if summary else "unknown"
        n_samples = summary.get("train_samples", "?") if summary else "?"
        
        color = COLORS[idx % len(COLORS)]
        label = f"{tag} ({loss_type}, n={n_samples})"
        
        # Detect incomparable validation losses (MSE vs log_ratio issue)
        val_losses = history["val_loss"].values
        train_losses = history["train_loss"].values
        
        # Find where validation loss jumps significantly (>10x increase)
        val_diff = np.diff(val_losses)
        jump_mask = np.abs(val_diff) > val_losses[:-1] * 10
        jump_indices = np.where(jump_mask)[0]
        
        # Plot training loss (always correct - uses log_ratio throughout)
        axes[0].plot(history["epoch"], history["train_loss"], 
                    color=color, linestyle="-", linewidth=2, label=f"{label} train")
        
        if len(jump_indices) > 0:
            # Found a jump - validation loss function changed
            switch_epoch = jump_indices[-1] + 1
            
            # Plot incomparable epochs (MSE) with different style
            epochs_incomparable = history["epoch"].iloc[:switch_epoch].values
            val_loss_incomparable = val_losses[:switch_epoch]
            
            # Plot comparable epochs (log_ratio) with normal style
            epochs_comparable = history["epoch"].iloc[switch_epoch:].values
            val_loss_comparable = val_losses[switch_epoch:]
            
            # Plot incomparable portion (MSE - different scale, show but mark as incomparable)
            if len(epochs_incomparable) > 0:
                axes[0].plot(epochs_incomparable, val_loss_incomparable, 
                            color=color, linestyle=":", linewidth=1, alpha=0.3,
                            label=f"{label} val (MSE, incomparable)" if idx == 0 else "")
            
            # Plot comparable portion (log_ratio)
            if len(epochs_comparable) > 0:
                axes[0].plot(epochs_comparable, val_loss_comparable, 
                            color=color, linestyle="--", linewidth=2, 
                            label=f"{label} val (log_ratio)")
            
            print(f"Note: {tag} - Epochs 1-{switch_epoch} validation loss uses MSE (incomparable)")
            print(f"      Epochs {switch_epoch+1}-{len(history)} validation loss uses {loss_type} (comparable)")
        else:
            # No jump detected - assume all epochs are comparable
            axes[0].plot(history["epoch"], history["val_loss"], 
                        color=color, linestyle="--", linewidth=2, label=f"{label} val")
        
        # Plot Log MAE curves (preferred) or fallback to linear MAE
        if "val_log_mae" in history.columns:
            axes[1].plot(history["epoch"], history["val_log_mae"], 
                        color=color, linewidth=2, label=label)
        else:
            # Fallback to linear MAE if log MAE not available (for older runs)
            axes[1].semilogy(history["epoch"], history["val_mae"], 
                            color=color, linewidth=2, label=label, linestyle=":")
    
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss (Normalized Space)", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9, loc="best")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Validation Log MAE", fontsize=12)
    axes[1].set_title("Validation Log MAE", fontsize=14, fontweight="bold")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=9, loc="best")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved loss curves to {output_path}")
    plt.close()


def plot_performance_vs_size(metrics_csv: Path, output_path: Path):
    """Plot performance metrics vs dataset size."""
    if not metrics_csv.exists():
        print(f"Warning: {metrics_csv} not found, skipping performance vs size plot")
        return
    
    df = pd.read_csv(metrics_csv)
    # Use 'dataset' or 'tag' column depending on what's available
    tag_col = "dataset" if "dataset" in df.columns else "tag"
    sample_col = "total_samples" if "total_samples" in df.columns else "train_samples"
    
    # Exclude base model and hyperparameter studies (latent dimension studies)
    df = df[df[tag_col] != "base"]
    # Filter out hyperparameter studies - only keep actual dataset size studies
    # These should start with 'x' followed by numbers (e.g., x32, x160, x240)
    df = df[df[tag_col].str.match(r'^x\d+', na=False)]
    
    # Prioritize optimal_retrained runs (the new standard)
    df['config_type'] = df[tag_col].str.replace(r'^x\d+', '', regex=True)
    df['dataset_size'] = df[tag_col].str.extract(r'x(\d+)').astype(int)
    
    # Check if we have optimal_retrained runs (new standard)
    df_optimal_retrained = df[df['config_type'] == '_optimal_retrained'].copy()
    
    if len(df_optimal_retrained) > 0:
        keep_sizes = [800, 1600, 2400, 3200, 4000, 4800]
        df_optimal_retrained = df_optimal_retrained[df_optimal_retrained['dataset_size'].isin(keep_sizes)].copy()
        df = df_optimal_retrained.sort_values(sample_col)
        print(f"Using optimal_retrained runs: {df[tag_col].tolist()}")
    elif len(df[df['config_type'] == '_consistent']) > 0:
        # Fallback to consistent runs
        df_consistent = df[df['config_type'] == '_consistent'].copy()
        df = df_consistent.sort_values(sample_col)
        print(f"Using consistent architecture runs: {df[tag_col].tolist()}")
    else:
        # Fallback to _optimal if no optimal_retrained runs
        df_optimal = df[df['config_type'] == '_optimal'].copy()
        df = df_optimal.sort_values(sample_col)
        print(f"Using '_optimal' configuration (no optimal_retrained runs found): {df[tag_col].tolist()}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Group by loss type if available
    if "loss_type" in df.columns:
        groups = df.groupby("loss_type")
    else:
        groups = [("default", df)]
    
    for loss_type, group in groups:
        color = COLORS[0] if loss_type == "huber" or loss_type == "default" else COLORS[1]
        marker = "o" if loss_type == "huber" or loss_type == "default" else "s"
        label = f"{loss_type}" if loss_type != "default" else "huber"
        
        # Test Loss - use log scale for consistency
        valid_test = ~group["test_loss"].isna()
        if valid_test.sum() > 0:
            axes[0, 0].semilogy(group[valid_test][sample_col] / 1000, group[valid_test]["test_loss"], 
                               marker=marker, color=color, linewidth=2, markersize=8, label=label)
        axes[0, 0].set_xlabel("Training Samples (×1000)", fontsize=12)
        axes[0, 0].set_ylabel("Test Loss (log_ratio)", fontsize=12)
        axes[0, 1].set_ylabel("Log MAE (dex)", fontsize=12)
        axes[0, 0].set_title("Test Loss vs Dataset Size", fontsize=14, fontweight="bold")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Log MAE - use log scale for consistency
        if "log_mae" in group.columns:
            valid = ~group["log_mae"].isna()
            if valid.sum() > 0:
                axes[0, 1].semilogy(group[valid][sample_col] / 1000, group[valid]["log_mae"], 
                                   marker=marker, color=color, linewidth=2, markersize=8, label=label)
        axes[0, 1].set_xlabel("Training Samples (×1000)", fontsize=12)
        axes[0, 1].set_title("Log MAE vs Dataset Size", fontsize=14, fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Log R² - linear scale (already normalized 0-1)
        if "log_r2" in group.columns:
            valid = ~group["log_r2"].isna()
            if valid.sum() > 0:
                axes[1, 0].plot(group[valid][sample_col] / 1000, group[valid]["log_r2"], 
                               marker=marker, color=color, linewidth=2, markersize=8, label=label)
        axes[1, 0].set_xlabel("Training Samples (×1000)", fontsize=12)
        axes[1, 0].set_ylabel("Log R²", fontsize=12)
        axes[1, 0].set_title("Log R² vs Dataset Size", fontsize=14, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Val Loss - use log scale for consistency
        valid_val = ~group["val_loss"].isna()
        if valid_val.sum() > 0:
            axes[1, 1].semilogy(group[valid_val][sample_col] / 1000, group[valid_val]["val_loss"], 
                               marker=marker, color=color, linewidth=2, markersize=8, label=label)
        axes[1, 1].set_xlabel("Training Samples (×1000)", fontsize=12)
        axes[1, 1].set_ylabel("Validation Loss (log_ratio)", fontsize=12)
        axes[1, 1].set_title("Validation Loss vs Dataset Size", fontsize=14, fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved performance vs size plot to {output_path}")
    plt.close()


def plot_model_size_comparison(metrics_csv: Path, output_path: Path):
    """Plot bar chart comparing different model configurations."""
    if not metrics_csv.exists():
        print(f"Warning: {metrics_csv} not found, skipping model size comparison")
        return
    
    df = pd.read_csv(metrics_csv)
    tag_col = "dataset" if "dataset" in df.columns else "tag"
    sample_col = "total_samples" if "total_samples" in df.columns else "train_samples"
    
    # Focus on recent models with similar sample counts
    recent = df[df[sample_col] >= 150000].copy()
    if len(recent) == 0:
        recent = df.tail(5).copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(recent))
    width = 0.6
    
    df_optimal = recent[recent[tag_col].str.contains("_optimal_retrained", na=False)].copy()
    if len(df_optimal) > 0:
        df_optimal['dataset_size'] = df_optimal[tag_col].str.extract(r'x(\d+)').astype(float)
        keep_sizes = [800, 1600, 2400, 3200, 4000, 4800]
        df_optimal = df_optimal[df_optimal['dataset_size'].isin(keep_sizes)].copy()
    if not df_optimal.empty:
        recent = df_optimal.sort_values(sample_col)
    
    x = np.arange(len(recent))
    
    # Test Loss - units: log_ratio
    axes[0].bar(x, recent["test_loss"], width, color=COLORS[0], alpha=0.8)
    axes[0].set_ylabel("Test Loss (log_ratio)", fontsize=12)
    axes[0].set_title("Test Loss Comparison", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(recent[tag_col], rotation=45, ha="right")
    axes[0].grid(axis="y", alpha=0.3)
    
    # Val Loss - units: log_ratio
    axes[1].bar(x, recent["val_loss"], width, color=COLORS[1], alpha=0.8)
    axes[1].set_ylabel("Validation Loss (log_ratio)", fontsize=12)
    axes[1].set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(recent[tag_col], rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.3)
    
    # Log MAE - units: dex
    if "log_mae" in recent.columns:
        valid = ~recent["log_mae"].isna()
        if valid.sum() > 0:
            axes[2].bar(x[valid], recent[valid]["log_mae"], width, color=COLORS[2], alpha=0.8)
    axes[2].set_ylabel("Log MAE (dex)", fontsize=12)
    axes[2].set_title("Log MAE Comparison", fontsize=14, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(recent[tag_col], rotation=45, ha="right")
    axes[2].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved model comparison to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate training analysis plots")
    # Default to optimal_retrained runs
    parser.add_argument("--runs", nargs="+", 
                        default=["x800_optimal_retrained", "x1600_optimal_retrained",
                                "x2400_optimal_retrained", "x3200_optimal_retrained",
                                "x4000_optimal_retrained", "x4800_optimal_retrained"],
                        help="Run tags to plot loss curves for")
    parser.add_argument("--base-dir", type=Path, 
                        default=Path(__file__).resolve().parent.parent / "results" / "runs",
                        help="Base directory containing run folders")
    parser.add_argument("--metrics-csv", type=Path, 
                        default=Path(__file__).resolve().parent.parent / "plots" / "comparison_metrics.csv",
                        help="Path to comparison metrics CSV")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "plots",
                        help="Output directory for plots")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TRAINING ANALYSIS PLOTS")
    print("="*80)
    print(f"Using runs: {', '.join(args.runs)}")
    
    # 1. Loss curves for specified runs
    print(f"\n1. Generating loss curves for: {', '.join(args.runs)}")
    plot_loss_curves(args.runs, args.base_dir, args.output_dir / "loss_curves.png")
    
    # 2. Performance vs dataset size
    print("\n2. Generating performance vs dataset size plot")
    plot_performance_vs_size(args.metrics_csv, args.output_dir / "performance_vs_size.png")
    
    # 3. Model comparison
    print("\n3. Generating model size comparison plot")
    plot_model_size_comparison(args.metrics_csv, args.output_dir / "model_comparison.png")
    
    print("\n" + "="*80)
    print("✅ All plots generated successfully!")
    print("="*80)
    print(f"\nPlots saved to: {args.output_dir}")
    print("  - loss_curves.png")
    print("  - performance_vs_size.png")
    print("  - model_comparison.png")


if __name__ == "__main__":
    main()

