#!/usr/bin/env python3
"""
plot_baseline_vs_improved.py
============================

Generate dedicated comparison plots between:
- x4800_optimal_retrained (baseline: Adam, eval normalization)
- x4800_improved (AdamW, train-only normalization, same FlowMap architecture)

Reads from plots/comparison_metrics.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chemcalculations._paths import project_root as repo_root

BASE_DIR = repo_root()
PLOTS_DIR = BASE_DIR / "plots"
COMPARISON_CSV = PLOTS_DIR / "comparison_metrics.csv"

plt.style.use("seaborn-v0_8-darkgrid")


def plot_baseline_vs_improved_bar(output_path: Path):
    """Side-by-side bar chart comparing x4800 baseline vs improved."""
    if not COMPARISON_CSV.exists():
        print(f"⚠️  {COMPARISON_CSV} not found")
        return

    df = pd.read_csv(COMPARISON_CSV)
    baseline = df[df["dataset"] == "x4800_optimal_retrained"].iloc[0]
    improved = df[df["dataset"] == "x4800_improved"].iloc[0]

    metrics = ["test_loss", "log_mae", "log_r2"]
    labels = ["Test Loss\n(log_ratio)", "Log MAE\n(dex)", "Log R²"]
    colors_baseline = "steelblue"
    colors_improved = "forestgreen"

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for ax, metric, label in zip(axes, metrics, labels):
        b_val = baseline[metric]
        i_val = improved[metric]
        x = np.arange(2)
        vals = [b_val, i_val]
        cols = [colors_baseline, colors_improved]
        bars = ax.bar(x, vals, color=cols, alpha=0.85, edgecolor="black", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(["x4800\n(baseline)", "x4800\n(improved)"])
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label.replace("\n", " "), fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, v in zip(bars, vals):
            h = bar.get_height()
            ax.annotate(f"{v:.4f}" if metric != "log_r2" else f"{v:.5f}",
                       xy=(bar.get_x() + bar.get_width() / 2, h),
                       xytext=(0, 5), textcoords="offset points", ha="center",
                       fontsize=10, fontweight="bold")

        if metric in ("test_loss", "log_mae"):
            ax.set_yscale("log")

    plt.suptitle("x4800 Baseline vs Improved (AdamW, train-only norm)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved baseline vs improved bar chart to {output_path}")
    plt.close()


def plot_baseline_vs_improved_performance_curve(output_path: Path):
    """Performance vs size: baseline series + x4800_improved overlaid at 4800K."""
    if not COMPARISON_CSV.exists():
        print(f"⚠️  {COMPARISON_CSV} not found")
        return

    df = pd.read_csv(COMPARISON_CSV)
    df_baseline = df[df["dataset"].str.contains("_optimal_retrained", na=False)].copy()
    df_baseline = df_baseline.sort_values("total_samples")

    improved = df[df["dataset"] == "x4800_improved"]
    if improved.empty:
        print("⚠️  x4800_improved not found in comparison_metrics.csv")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test Loss
    valid = ~df_baseline["test_loss"].isna() & (df_baseline["test_loss"] > 0)
    if valid.sum() > 0:
        x_b = df_baseline[valid]["total_samples"] / 1000
        y_b = df_baseline[valid]["test_loss"]
        axes[0, 0].semilogy(x_b, y_b, marker="o", linewidth=2, markersize=10,
                           color="steelblue", label="Baseline (optimal_retrained)")
    axes[0, 0].semilogy(4800, improved["test_loss"].iloc[0], marker="s", markersize=14,
                       color="forestgreen", label="x4800_improved (AdamW)", linestyle="")
    axes[0, 0].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[0, 0].set_ylabel("Test Loss (log_ratio)", fontsize=12)
    axes[0, 0].set_title("Test Loss vs Dataset Size", fontsize=14, fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Log MAE
    valid = ~df_baseline["log_mae"].isna() & (df_baseline["log_mae"] > 0)
    if valid.sum() > 0:
        x_b = df_baseline[valid]["total_samples"] / 1000
        y_b = df_baseline[valid]["log_mae"]
        axes[0, 1].semilogy(x_b, y_b, marker="o", linewidth=2, markersize=10,
                           color="coral", label="Baseline (optimal_retrained)")
    axes[0, 1].semilogy(4800, improved["log_mae"].iloc[0], marker="s", markersize=14,
                       color="forestgreen", label="x4800_improved (AdamW)", linestyle="")
    axes[0, 1].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[0, 1].set_ylabel("Log MAE (dex)", fontsize=12)
    axes[0, 1].set_title("Log MAE vs Dataset Size", fontsize=14, fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Log R²
    valid = ~df_baseline["log_r2"].isna()
    if valid.sum() > 0:
        x_b = df_baseline[valid]["total_samples"] / 1000
        y_b = df_baseline[valid]["log_r2"]
        axes[1, 0].plot(x_b, y_b, marker="o", linewidth=2, markersize=10,
                       color="green", label="Baseline (optimal_retrained)")
    axes[1, 0].plot(4800, improved["log_r2"].iloc[0], marker="s", markersize=14,
                   color="forestgreen", label="x4800_improved (AdamW)", linestyle="")
    axes[1, 0].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[1, 0].set_ylabel("Log R²", fontsize=12)
    axes[1, 0].set_title("Log R² vs Dataset Size", fontsize=14, fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Validation Loss
    valid = ~df_baseline["val_loss"].isna() & (df_baseline["val_loss"] > 0)
    if valid.sum() > 0:
        x_b = df_baseline[valid]["total_samples"] / 1000
        y_b = df_baseline[valid]["val_loss"]
        axes[1, 1].semilogy(x_b, y_b, marker="o", linewidth=2, markersize=10,
                           color="purple", label="Baseline (optimal_retrained)")
    axes[1, 1].semilogy(4800, improved["val_loss"].iloc[0], marker="s", markersize=14,
                       color="forestgreen", label="x4800_improved (AdamW)", linestyle="")
    axes[1, 1].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[1, 1].set_ylabel("Validation Loss (log_ratio)", fontsize=12)
    axes[1, 1].set_title("Validation Loss vs Dataset Size", fontsize=14, fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Baseline vs Improved: x4800_improved overlaid at 4800K", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved baseline vs improved performance curve to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Baseline vs improved comparison plots")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR, help="Output directory")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating baseline vs improved comparison plots...")
    plot_baseline_vs_improved_bar(args.output_dir / "baseline_vs_improved_bar.png")
    plot_baseline_vs_improved_performance_curve(args.output_dir / "baseline_vs_improved_performance.png")
    print("Done.")


if __name__ == "__main__":
    main()
