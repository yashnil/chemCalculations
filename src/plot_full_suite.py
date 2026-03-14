#!/usr/bin/env python3
"""
plot_full_suite.py
==================

Generate a comprehensive suite of plots for the optimal_retrained models:
1. Loss curves for all dataset sizes
2. Performance metrics vs dataset size
3. Scatter plots (predicted vs true) for optimal model
4. Parity plots for top species
5. Error distribution plots
6. Model comparison plots
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "results" / "runs"
PLOTS_DIR = BASE_DIR / "plots"
COMPARISON_CSV = PLOTS_DIR / "comparison_metrics.csv"

# Optimal retrained runs
OPTIMAL_RETRAINED_RUNS = [
    "x800_optimal_retrained",
    "x1600_optimal_retrained",
    "x2400_optimal_retrained",
    "x3200_optimal_retrained",
    "x4000_optimal_retrained",
    "x4800_optimal_retrained",
]
# Best model: prefer improved, else largest optimal_retrained
BEST_MODEL_RUNS = ["x4800_improved"] + list(reversed(OPTIMAL_RETRAINED_RUNS))

# Add src/ to path
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def load_best_model_module(run_tag: str):
    """Load the best_model.py module for a run."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    best_model_path = run_dir / "best_model.py"
    
    if not best_model_path.exists():
        return None
    
    spec = importlib.util.spec_from_file_location("best_model", best_model_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_loss_history(run_tag: str) -> pd.DataFrame | None:
    """Load loss_history.csv from a run directory."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    history_path = run_dir / "loss_history.csv"
    if not history_path.exists():
        return None
    return pd.read_csv(history_path)


def load_summary(run_tag: str) -> dict | None:
    """Load summary.json from a run directory."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    import json
    with open(summary_path) as f:
        return json.load(f)


def plot_loss_curves_all_sizes(output_path_loss: Path, output_path_log_mae: Path):
    """Plot training and validation loss curves for all optimal_retrained runs.
    
    Creates two separate plots:
    1. Training and validation loss vs epochs
    2. Validation log MAE vs epochs
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(OPTIMAL_RETRAINED_RUNS)))
    
    # Plot 1: Training & Validation Loss
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    for idx, run_tag in enumerate(OPTIMAL_RETRAINED_RUNS):
        history = load_loss_history(run_tag)
        if history is None:
            continue
        
        size = run_tag.split('_')[0].replace('x', '')
        color = colors[idx]
        
        ax1.plot(history["epoch"], history["train_loss"], 
                color=color, linestyle="-", linewidth=2, 
                label=f"x{size}K train", alpha=0.8)
        ax1.plot(history["epoch"], history["val_loss"], 
                color=color, linestyle="--", linewidth=2, 
                label=f"x{size}K val", alpha=0.8)
    
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Loss (log_ratio)", fontsize=14)
    ax1.set_title("Training & Validation Loss", fontsize=16, fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, ncol=2, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path_loss, dpi=150, bbox_inches="tight")
    print(f"✅ Saved training/validation loss curves to {output_path_loss}")
    plt.close()
    
    # Plot 2: Validation Log MAE
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    for idx, run_tag in enumerate(OPTIMAL_RETRAINED_RUNS):
        history = load_loss_history(run_tag)
        if history is None:
            continue
        
        size = run_tag.split('_')[0].replace('x', '')
        color = colors[idx]
        
        if "val_log_mae" in history.columns:
            ax2.plot(history["epoch"], history["val_log_mae"], 
                    color=color, linewidth=2, label=f"x{size}K", alpha=0.8)
    
    ax2.set_xlabel("Epoch", fontsize=14)
    ax2.set_ylabel("Log MAE (dex)", fontsize=14)
    ax2.set_title("Validation Log MAE", fontsize=16, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, ncol=2, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path_log_mae, dpi=150, bbox_inches="tight")
    print(f"✅ Saved validation log MAE curves to {output_path_log_mae}")
    plt.close()


def plot_performance_vs_size(output_path: Path):
    """Plot performance metrics vs dataset size (baseline + x4800_improved overlaid)."""
    if not COMPARISON_CSV.exists():
        print(f"⚠️  {COMPARISON_CSV} not found")
        return

    df = pd.read_csv(COMPARISON_CSV)

    # Baseline: optimal_retrained runs
    df_optimal = df[df["dataset"].str.contains("_optimal_retrained", na=False)].copy()
    df_optimal = df_optimal.sort_values("total_samples")
    df_improved = df[df["dataset"] == "x4800_improved"]

    if df_optimal.empty:
        print("⚠️  No optimal_retrained runs found in comparison_metrics.csv")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test Loss - log scale, units: log_ratio
    valid_test = ~df_optimal["test_loss"].isna() & (df_optimal["test_loss"] > 0)
    if valid_test.sum() > 0:
        axes[0, 0].semilogy(df_optimal[valid_test]["total_samples"] / 1000,
                           df_optimal[valid_test]["test_loss"],
                           marker='o', linewidth=2, markersize=10, color='steelblue', label='Baseline')
    if not df_improved.empty:
        axes[0, 0].semilogy(4800, df_improved["test_loss"].iloc[0], marker='s', markersize=12,
                           color='forestgreen', label='x4800_improved', linestyle='')
    axes[0, 0].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[0, 0].set_ylabel("Test Loss (log_ratio)", fontsize=12)
    axes[0, 0].set_title("Test Loss vs Dataset Size", fontsize=14, fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Log MAE - log scale, units: dex
    valid_mae = ~df_optimal["log_mae"].isna() & (df_optimal["log_mae"] > 0)
    if valid_mae.sum() > 0:
        axes[0, 1].semilogy(df_optimal[valid_mae]["total_samples"] / 1000,
                           df_optimal[valid_mae]["log_mae"],
                           marker='o', linewidth=2, markersize=10, color='coral', label='Baseline')
    if not df_improved.empty:
        axes[0, 1].semilogy(4800, df_improved["log_mae"].iloc[0], marker='s', markersize=12,
                           color='forestgreen', label='x4800_improved', linestyle='')
    axes[0, 1].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[0, 1].set_ylabel("Log MAE (dex)", fontsize=12)
    axes[0, 1].set_title("Log MAE vs Dataset Size", fontsize=14, fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Log R² - linear scale, units: unitless (0-1)
    valid_r2 = ~df_optimal["log_r2"].isna()
    if valid_r2.sum() > 0:
        axes[1, 0].plot(df_optimal[valid_r2]["total_samples"] / 1000,
                       df_optimal[valid_r2]["log_r2"],
                       marker='o', linewidth=2, markersize=10, color='green', label='Baseline')
    if not df_improved.empty:
        axes[1, 0].plot(4800, df_improved["log_r2"].iloc[0], marker='s', markersize=12,
                       color='forestgreen', label='x4800_improved', linestyle='')
    axes[1, 0].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[1, 0].set_ylabel("Log R²", fontsize=12)
    axes[1, 0].set_title("Log R² vs Dataset Size", fontsize=14, fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Validation Loss - log scale, units: log_ratio
    valid_val = ~df_optimal["val_loss"].isna() & (df_optimal["val_loss"] > 0)
    if valid_val.sum() > 0:
        axes[1, 1].semilogy(df_optimal[valid_val]["total_samples"] / 1000,
                           df_optimal[valid_val]["val_loss"],
                           marker='o', linewidth=2, markersize=10, color='purple', label='Baseline')
    if not df_improved.empty:
        axes[1, 1].semilogy(4800, df_improved["val_loss"].iloc[0], marker='s', markersize=12,
                           color='forestgreen', label='x4800_improved', linestyle='')
    axes[1, 1].set_xlabel("Training Samples (×1000)", fontsize=12)
    axes[1, 1].set_ylabel("Validation Loss (log_ratio)", fontsize=12)
    axes[1, 1].set_title("Validation Loss vs Dataset Size", fontsize=14, fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved performance vs size plot to {output_path}")
    plt.close()


def plot_scatter_optimal_model(output_path: Path, run_tag: str = "x800_optimal_retrained"):
    """Generate scatter plot (predicted vs true) for the optimal model."""
    # Try to find the best model (largest dataset)
    best_mod = None
    best_run = None
    
    for run in reversed(OPTIMAL_RETRAINED_RUNS):
        mod = load_best_model_module(run)
        if mod is not None:
            best_mod = mod
            best_run = run
            break
    
    if best_mod is None:
        print(f"⚠️  Could not find optimal model module")
        return
    
    print(f"  Using model: {best_run}")
    
    # Load dataset
    size = best_run.split('_')[0].replace('x', '')
    csv_path = BASE_DIR / "data" / "datasets" / f"all_gas_fastchem_x{size}.csv"
    
    if not csv_path.exists():
        print(f"⚠️  Dataset not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Get test indices
    splits = getattr(best_mod, "SPLITS", {})
    test_idx = np.array(splits.get("test_idx", []), dtype=int)
    
    if len(test_idx) == 0:
        print("⚠️  No test indices found")
        return
    
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    # Get target columns
    target_cols = best_mod.TARGET_COLS
    
    # Normalize inputs and run inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = best_mod.load_model(device=device)
    model.eval()
    
    X_test = best_mod.normalize_inputs(df_test)
    
    with torch.no_grad():
        pred_scaled = best_mod.forward_autoencoder(model, X_test)
        pred_scaled = pred_scaled.cpu().numpy()
    
    y_pred = best_mod.denormalize_targets(pred_scaled)
    y_true = df_test[target_cols].to_numpy(dtype=np.float64, copy=True)
    
    # Clip to avoid numerical issues
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    
    # Flatten for overall scatter plot
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    # Filter valid values
    mask = (y_true_flat > 1e-30) & (y_pred_flat > 1e-30)
    x_plot = y_true_flat[mask]
    y_plot = y_pred_flat[mask]
    
    if len(x_plot) == 0:
        print("⚠️  No valid data points for scatter plot")
        return
    
    # Compute metrics
    log_x = np.log10(x_plot)
    log_y = np.log10(y_plot)
    log_mae = mean_absolute_error(log_x, log_y)
    log_r2 = r2_score(log_x, log_y)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Subsample for performance if too many points
    MAX_SCATTER = 200_000
    if len(x_plot) > MAX_SCATTER:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_plot), MAX_SCATTER, replace=False)
        x_plot, y_plot = x_plot[idx], y_plot[idx]
        log_x, log_y = log_x[idx], log_y[idx]
    
    try:
        from scipy.stats import gaussian_kde
        dens = gaussian_kde(np.vstack([log_x, log_y]))(np.vstack([log_x, log_y]))
        order = dens.argsort()
        ax.scatter(x_plot[order], y_plot[order], c=dens[order], 
                  cmap="viridis", s=6, alpha=0.6, linewidths=0)
    except Exception:
        ax.scatter(x_plot, y_plot, s=6, alpha=0.3, c='steelblue')
    
    # Add 1:1 line
    lims = [max(x_plot.min(), y_plot.min()), min(x_plot.max(), y_plot.max())]
    ax.plot(lims, lims, 'k--', linewidth=2, label='1:1 line', zorder=10)
    
    # Add ±0.1 dex lines
    xx = np.geomspace(lims[0], lims[1], 100)
    ax.plot(xx, xx * 10**0.1, 'r--', alpha=0.5, linewidth=1, label='±0.1 dex')
    ax.plot(xx, xx * 10**(-0.1), 'r--', alpha=0.5, linewidth=1)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"True number density (cm$^{-3}$)", fontsize=12)
    ax.set_ylabel(r"Predicted number density (cm$^{-3}$)", fontsize=12)
    ax.set_title(f"Predicted vs True: {best_run}\nLog MAE={log_mae:.4f} dex, Log R²={log_r2:.6f}", 
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved scatter plot to {output_path}")
    plt.close()


def plot_aafe_per_species(output_path: Path):
    """Average absolute fractional error per species for the best model."""
    best_mod = None
    best_run = None
    for run in BEST_MODEL_RUNS:
        mod = load_best_model_module(run)
        if mod is not None:
            best_mod = mod
            best_run = run
            break
    if best_mod is None:
        print("⚠️  Could not find best model for AAFE plot")
        return

    size = best_run.split("_")[0].replace("x", "")
    csv_path = BASE_DIR / "data" / "datasets" / f"all_gas_fastchem_x{size}.csv"
    if not csv_path.exists():
        print(f"⚠️  Dataset not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    splits = getattr(best_mod, "SPLITS", {})
    test_idx = np.array(splits.get("test_idx", []), dtype=int)
    if len(test_idx) == 0:
        print("⚠️  No test indices found")
        return

    df_test = df.iloc[test_idx].reset_index(drop=True)
    target_cols = best_mod.TARGET_COLS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = best_mod.load_model(device=device)
    model.eval()

    X_test = best_mod.normalize_inputs(df_test)
    with torch.no_grad():
        pred_scaled = best_mod.forward_autoencoder(model, X_test)
        pred_scaled = pred_scaled.cpu().numpy()
    y_pred = best_mod.denormalize_targets(pred_scaled)
    y_true = df_test[target_cols].to_numpy(dtype=np.float64, copy=True)
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)

    eps = 1e-30
    aafe_per_species = []
    for i, sp in enumerate(target_cols):
        denom = np.maximum(y_true[:, i], eps)
        frac_err = np.abs(y_pred[:, i] - y_true[:, i]) / denom
        aafe_per_species.append((sp, float(np.mean(frac_err))))

    species, aafe = zip(*sorted(aafe_per_species, key=lambda x: x[1], reverse=True))
    aafe = np.array(aafe)
    aafe_global = float(np.mean(aafe))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["red" if v > aafe_global else "steelblue" for v in aafe]
    ax.barh(range(len(species)), aafe, color=colors, alpha=0.7)
    ax.set_yticks(range(len(species)))
    ax.set_yticklabels(species, fontsize=8)
    ax.set_xlabel("Average Absolute Fractional Error (|pred−true|/true)", fontsize=11)
    ax.set_title(f"AAFE per Species: {best_run}\nGlobal average = {aafe_global:.4f} (Red = Above Average)", fontsize=12)
    ax.axvline(aafe_global, color="black", linestyle="--", linewidth=1, label=f"Global AAFE = {aafe_global:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved AAFE per species to {output_path}")
    plt.close()


def plot_model_comparison_bar(output_path: Path):
    """Bar chart comparing all optimal_retrained models + x4800_improved."""
    if not COMPARISON_CSV.exists():
        print(f"⚠️  {COMPARISON_CSV} not found")
        return

    df = pd.read_csv(COMPARISON_CSV)

    # Include optimal_retrained runs + x4800_improved
    df_optimal = df[df["dataset"].str.contains("_optimal_retrained", na=False)].copy()
    df_optimal = df_optimal.sort_values("total_samples")
    df_improved = df[df["dataset"] == "x4800_improved"]
    if not df_improved.empty:
        df_optimal = pd.concat([df_optimal, df_improved], ignore_index=True)

    if df_optimal.empty:
        print("⚠️  No optimal_retrained runs found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(df_optimal))
    width = 0.6
    # Color: baseline blue, improved green
    colors = ["steelblue"] * len(df_optimal)
    for i, ds in enumerate(df_optimal["dataset"]):
        if ds == "x4800_improved":
            colors[i] = "forestgreen"

    # Test Loss
    valid = ~df_optimal["test_loss"].isna()
    if valid.sum() > 0:
        axes[0].bar(x[valid], df_optimal.loc[valid, "test_loss"], width,
                   color=[c for c, v in zip(colors, valid) if v], alpha=0.8)
    axes[0].set_ylabel("Test Loss (log_ratio)", fontsize=12)
    axes[0].set_title("Test Loss Comparison", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    labels = [tag.replace("_optimal_retrained", "").replace("x4800_improved", "x4800 (improved)") for tag in df_optimal["dataset"]]
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].grid(axis="y", alpha=0.3)

    # Log MAE
    valid = ~df_optimal["log_mae"].isna()
    if valid.sum() > 0:
        axes[1].bar(x[valid], df_optimal.loc[valid, "log_mae"], width,
                   color=[c for c, v in zip(colors, valid) if v], alpha=0.8)
    axes[1].set_ylabel("Log MAE (dex)", fontsize=12)
    axes[1].set_title("Log MAE Comparison", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.3)

    # Log R²
    valid = ~df_optimal["log_r2"].isna()
    if valid.sum() > 0:
        axes[2].bar(x[valid], df_optimal.loc[valid, "log_r2"], width,
                   color=[c for c, v in zip(colors, valid) if v], alpha=0.8)
    axes[2].set_ylabel("Log R²", fontsize=12)
    axes[2].set_title("Log R² Comparison", fontsize=14, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved model comparison to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive plot suite")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR,
                       help="Output directory for plots")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GENERATING COMPREHENSIVE PLOT SUITE")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print()
    
    # 1. Loss curves for all sizes (split into two plots)
    print("1. Generating loss curves for all sizes...")
    plot_loss_curves_all_sizes(
        args.output_dir / "loss_curves_full_suite.png",
        args.output_dir / "log_mae_curves_full_suite.png"
    )
    
    # 2. Performance vs dataset size
    print("\n2. Generating performance vs dataset size plot...")
    plot_performance_vs_size(args.output_dir / "performance_vs_size_full_suite.png")
    
    # 3. Scatter plot for optimal model
    print("\n3. Generating scatter plot (predicted vs true) for optimal model...")
    plot_scatter_optimal_model(args.output_dir / "scatter_optimal_model.png")

    # 4. AAFE per species (best model)
    print("\n4. Generating AAFE per species plot...")
    plot_aafe_per_species(args.output_dir / "AAFE_per_species.png")

    # 5. Model comparison bar chart
    print("\n5. Generating model comparison bar chart...")
    plot_model_comparison_bar(args.output_dir / "model_comparison_bar.png")

    # 6. Baseline vs improved comparison (x4800)
    print("\n6. Generating baseline vs improved comparison plots...")
    try:
        from plot_baseline_vs_improved import plot_baseline_vs_improved_bar, plot_baseline_vs_improved_performance_curve
        plot_baseline_vs_improved_bar(args.output_dir / "baseline_vs_improved_bar.png")
        plot_baseline_vs_improved_performance_curve(args.output_dir / "baseline_vs_improved_performance.png")
    except Exception as e:
        print(f"⚠️  Baseline vs improved plots skipped: {e}")

    print("\n" + "="*80)
    print("✅ ALL PLOTS GENERATED!")
    print("="*80)
    print(f"\nPlots saved to: {args.output_dir}")
    print("  - loss_curves_full_suite.png (training & validation loss)")
    print("  - log_mae_curves_full_suite.png (validation log MAE)")
    print("  - performance_vs_size_full_suite.png")
    print("  - scatter_optimal_model.png")
    print("  - AAFE_per_species.png")
    print("  - model_comparison_bar.png")
    print("  - baseline_vs_improved_bar.png")
    print("  - baseline_vs_improved_performance.png")


if __name__ == "__main__":
    main()
