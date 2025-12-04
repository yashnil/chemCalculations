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


def plot_loss_curves(run_tags: List[str], base_dir: Path, output_path: Path):
    """Plot training and validation loss curves for specified runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, tag in enumerate(run_tags):
        run_dir = base_dir / f"runs_autoencoder_{tag}"
        if not run_dir.exists():
            print(f"Warning: {run_dir} not found, skipping...")
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
        
        # Plot loss curves
        axes[0].plot(history["epoch"], history["train_loss"], 
                    color=color, linestyle="-", linewidth=2, label=f"{label} train")
        axes[0].plot(history["epoch"], history["val_loss"], 
                    color=color, linestyle="--", linewidth=2, label=f"{label} val")
        
        # Plot MAE curves
        axes[1].semilogy(history["epoch"], history["val_mae"], 
                        color=color, linewidth=2, label=label)
    
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss (Normalized Space)", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=9, loc="best")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Validation MAE (Linear Space)", fontsize=12)
    axes[1].set_title("Validation MAE", fontsize=14, fontweight="bold")
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
    
    df = df[df[tag_col] != "base"]  # Exclude base model
    df = df.sort_values(sample_col)
    
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
        
        # Test Loss
        axes[0, 0].plot(group[sample_col] / 1000, group["test_loss"], 
                       marker=marker, color=color, linewidth=2, markersize=8, label=label)
        axes[0, 0].set_xlabel("Training Samples (×1000)", fontsize=12)
        axes[0, 0].set_ylabel("Test Loss (Normalized)", fontsize=12)
        axes[0, 0].set_title("Test Loss vs Dataset Size", fontsize=14, fontweight="bold")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Log MAE (if exists)
        if "log_mae" in group.columns:
            # Filter out nan values
            valid = ~group["log_mae"].isna()
            if valid.sum() > 0:
                axes[0, 1].plot(group[valid][sample_col] / 1000, group[valid]["log_mae"], 
                               marker=marker, color=color, linewidth=2, markersize=8, label=label)
        axes[0, 1].set_xlabel("Training Samples (×1000)", fontsize=12)
        axes[0, 1].set_ylabel("Log MAE", fontsize=12)
        axes[0, 1].set_title("Log MAE vs Dataset Size", fontsize=14, fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Log R²
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
        
        # Val Loss
        axes[1, 1].plot(group[sample_col] / 1000, group["val_loss"], 
                       marker=marker, color=color, linewidth=2, markersize=8, label=label)
        axes[1, 1].set_xlabel("Training Samples (×1000)", fontsize=12)
        axes[1, 1].set_ylabel("Validation Loss", fontsize=12)
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
    
    # Test Loss
    axes[0].bar(x, recent["test_loss"], width, color=COLORS[0], alpha=0.8)
    axes[0].set_ylabel("Test Loss (Normalized)", fontsize=12)
    axes[0].set_title("Test Loss Comparison", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(recent[tag_col], rotation=45, ha="right")
    axes[0].grid(axis="y", alpha=0.3)
    
    # Val Loss
    axes[1].bar(x, recent["val_loss"], width, color=COLORS[1], alpha=0.8)
    axes[1].set_ylabel("Validation Loss (Normalized)", fontsize=12)
    axes[1].set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(recent[tag_col], rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.3)
    
    # Log MAE (if exists)
    if "log_mae" in recent.columns:
        valid = ~recent["log_mae"].isna()
        if valid.sum() > 0:
            axes[2].bar(x[valid], recent[valid]["log_mae"], width, color=COLORS[2], alpha=0.8)
    axes[2].set_ylabel("Log MAE", fontsize=12)
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
    parser.add_argument("--runs", nargs="+", default=["x160_new", "x160_mse"],
                        help="Run tags to plot loss curves for (default: x160_new x160_mse)")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent,
                        help="Base directory containing run folders")
    parser.add_argument("--metrics-csv", type=Path, 
                        default=Path(__file__).parent / "comparison_metrics.csv",
                        help="Path to comparison metrics CSV")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent,
                        help="Output directory for plots")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TRAINING ANALYSIS PLOTS")
    print("="*80)
    
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


if __name__ == "__main__":
    main()

