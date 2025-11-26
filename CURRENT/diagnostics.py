#!/usr/bin/env python3
"""
Diagnostics suite for the FlowMap autoencoder (UPDATED_VERS).

Mirrors the NEW_VERS diagnostics so results can be compared side by side.
Produces parity plots, per-species metrics, error distributions, and summary text.
"""

from __future__ import annotations

import os
import sys
import importlib.util
import logging
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

# Optional dependency
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found; KDE density coloring disabled")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CSV_PATH = os.environ.get(
    "CSV_PATH",
    "/Users/yashnilmohanty/Desktop/chemCalculations/NEW_VERS/all_gas_v10_no_stripe_clean.csv",
)
BEST_MODULE = os.environ.get("BEST_MODULE", "runs_autoencoder/best_model.py")
OUT_DIR = Path(os.environ.get("OUT_DIR", "runs_autoencoder/diagnostics"))

FIGSIZE_LARGE = (15, 6)
FIGSIZE_SINGLE = (8, 6)
DPI = 150
CLIP = 1e-10
JITTER_DEX = 0.03

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("diagnostics_autoencoder")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def load_module_from_path(path: str):
    path = os.path.abspath(path)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt(x, decimals: int = 3) -> str:
    try:
        return f"{float(x):.{decimals}e}"
    except Exception:
        return str(x)


# -----------------------------------------------------------------------------
# Main flow
# -----------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", OUT_DIR)

    log.info("Loading best_model module from %s", BEST_MODULE)
    best_mod = load_module_from_path(BEST_MODULE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device.type)

    model = best_mod.load_model(device=device)
    model.eval()

    input_cols: List[str] = best_mod.INPUT_COLS
    target_cols: List[str] = best_mod.TARGET_COLS
    log.info("Model: %d inputs → %d outputs", len(input_cols), len(target_cols))

    log.info("Loading CSV: %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    log.info("Loaded: %d rows × %d cols", df.shape[0], df.shape[1])

    splits = getattr(best_mod, "SPLITS", {})
    test_idx = np.array(splits.get("test_idx", []), dtype=int)
    if test_idx.size == 0:
        log.warning("No test indices found; defaulting to entire dataset")
        test_idx = np.arange(len(df))

    df_test = df.iloc[test_idx].reset_index(drop=True)
    log.info("Using test set: %d samples", len(df_test))

    log.info("Normalizing inputs and running inference...")
    X_test = best_mod.normalize_inputs(df_test)

    with torch.no_grad():
        pred_scaled = best_mod.forward_autoencoder(model, X_test)
        pred_scaled = pred_scaled.cpu().numpy()

    y_pred = best_mod.denormalize_targets(pred_scaled)
    y_true = df_test[target_cols].to_numpy(dtype=np.float64, copy=True)

    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    log.info("Predictions complete. Shape: %s", y_pred.shape)

    # -------------------------------------------------------------------------
    # Global metrics
    # -------------------------------------------------------------------------
    log.info("Computing global metrics...")

    mae_linear = mean_absolute_error(y_true, y_pred)
    r2_linear = r2_score(y_true, y_pred, multioutput="variance_weighted")

    y_true_log = np.log10(y_true + CLIP)
    y_pred_log = np.log10(y_pred + CLIP)
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log, multioutput="variance_weighted")

    log.info("GLOBAL METRICS:")
    log.info("  Linear MAE: %s", fmt(mae_linear))
    log.info("  Linear R²:  %.4f", r2_linear)
    log.info("  Log MAE:    %s", fmt(mae_log))
    log.info("  Log R²:     %.4f", r2_log)

    with open(OUT_DIR / "global_metrics.txt", "w") as f:
        f.write(f"Linear MAE:  {mae_linear:.6e}\n")
        f.write(f"Linear R²:   {r2_linear:.6f}\n")
        f.write(f"Log MAE:     {mae_log:.6e}\n")
        f.write(f"Log R²:      {r2_log:.6f}\n")
        f.write(f"Test samples: {len(y_true)}\n")
        f.write(f"Species:     {len(target_cols)}\n")

    # -------------------------------------------------------------------------
    # Per-species metrics
    # -------------------------------------------------------------------------
    log.info("Computing per-species errors...")
    per_species = []
    for i, sp in enumerate(target_cols):
        mae_sp = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2_sp = r2_score(y_true[:, i], y_pred[:, i])
        max_abun = y_true[:, i].max()
        mean_abun = y_true[:, i].mean()
        per_species.append(
            {
                "species": sp,
                "MAE": mae_sp,
                "R2": r2_sp,
                "max_abundance": max_abun,
                "mean_abundance": mean_abun,
            }
        )

    df_species = pd.DataFrame(per_species)
    df_species.sort_values("MAE", ascending=False).to_csv(
        OUT_DIR / "per_species_errors.csv", index=False
    )

    log.info("Top-5 species by error:")
    for _, row in df_species.nlargest(5, "MAE").iterrows():
        log.info("  %s: MAE=%s", row["species"], fmt(row["MAE"]))

    # -------------------------------------------------------------------------
    # Parity plots (top-10 by abundance)
    # -------------------------------------------------------------------------
    log.info("Generating parity plots for top-10 species...")
    top10 = df_species.nlargest(10, "mean_abundance")["species"].values

    fig, axes = plt.subplots(2, 5, figsize=FIGSIZE_LARGE)
    axes = axes.ravel()

    for ax, sp in zip(axes, top10):
        idx = target_cols.index(sp)
        x_vals = y_true[:, idx]
        y_vals = y_pred[:, idx]
        mask = (x_vals > CLIP) & (y_vals > CLIP)
        x_plot = x_vals[mask]
        y_plot = y_vals[mask]

        if len(x_plot) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.set_title(sp)
            continue

        if JITTER_DEX > 0:
            noise = 10 ** np.random.uniform(-JITTER_DEX, JITTER_DEX, size=len(x_plot))
            x_plot = x_plot * noise
            y_plot = y_plot * noise

        if HAS_SCIPY and len(x_plot) > 10:
            lx, ly = np.log10(x_plot), np.log10(y_plot)
            try:
                dens = gaussian_kde(np.vstack([lx, ly]))(np.vstack([lx, ly]))
                order = dens.argsort()
                ax.scatter(
                    x_plot[order],
                    y_plot[order],
                    c=dens[order],
                    cmap="viridis",
                    s=5,
                    alpha=0.7,
                    linewidths=0,
                )
            except Exception:
                ax.scatter(x_plot, y_plot, s=5, alpha=0.5, c="blue")
        else:
            ax.scatter(x_plot, y_plot, s=5, alpha=0.5, c="blue")

        lims = [CLIP, 1.0]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
        xx = np.geomspace(CLIP, 1.0, 100)
        ax.fill_between(xx, 0.9 * xx, 1.1 * xx, color="gray", alpha=0.2, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(CLIP, 1.0)
        ax.set_ylim(CLIP, 1.0)
        ax.set_xlabel("True", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.set_title(sp, fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "parity_top10.png", dpi=DPI)
    plt.close()
    log.info("✓ Saved: parity_top10.png")

    # Overall parity
    log.info("Generating overall parity plot...")
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    mask = (y_true_flat > CLIP) & (y_pred_flat > CLIP)
    x_plot = y_true_flat[mask]
    y_plot = y_pred_flat[mask]

    if len(x_plot) > 50000:
        idx_sample = np.random.choice(len(x_plot), 50000, replace=False)
        x_plot = x_plot[idx_sample]
        y_plot = y_plot[idx_sample]

    if HAS_SCIPY and len(x_plot) > 10:
        lx, ly = np.log10(x_plot), np.log10(y_plot)
        try:
            dens = gaussian_kde(np.vstack([lx, ly]))(np.vstack([lx, ly]))
            order = dens.argsort()
            scatter = ax.scatter(
                x_plot[order],
                y_plot[order],
                c=dens[order],
                cmap="viridis",
                s=3,
                alpha=0.6,
                linewidths=0,
            )
            plt.colorbar(scatter, ax=ax, label="Density")
        except Exception:
            ax.hexbin(
                x_plot, y_plot, xscale="log", yscale="log", gridsize=100, cmap="viridis", mincnt=1
            )
    else:
        ax.hexbin(
            x_plot, y_plot, xscale="log", yscale="log", gridsize=100, cmap="viridis", mincnt=1
        )

    lims = [CLIP, 1.0]
    ax.plot(lims, lims, "k--", lw=1.5, alpha=0.8, label="1:1")
    xx = np.geomspace(CLIP, 1.0, 100)
    ax.fill_between(xx, 0.9 * xx, 1.1 * xx, color="gray", alpha=0.3, label="±10%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(CLIP, 1.0)
    ax.set_ylim(CLIP, 1.0)
    ax.set_xlabel("True Abundance", fontsize=12)
    ax.set_ylabel("Predicted Abundance", fontsize=12)
    ax.set_title("Overall Parity: All Species Pooled", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "parity_overall.png", dpi=DPI)
    plt.close()
    log.info("✓ Saved: parity_overall.png")

    # MAE per species plot
    log.info("Generating MAE per species plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    df_sorted = df_species.sort_values("MAE", ascending=False)
    colors = ["red" if mae > mae_linear else "steelblue" for mae in df_sorted["MAE"]]
    ax.barh(range(len(df_sorted)), df_sorted["MAE"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["species"], fontsize=8)
    ax.set_xlabel("Mean Absolute Error", fontsize=11)
    ax.set_title("MAE per Species (Red = Above Global Average)", fontsize=12)
    ax.axvline(mae_linear, color="black", linestyle="--", linewidth=1, label=f"Global MAE = {mae_linear:.3e}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "MAE_per_species.png", dpi=DPI)
    plt.close()
    log.info("✓ Saved: MAE_per_species.png")

    # Residual vs observed
    log.info("Generating residual vs observed plot...")
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    residuals = y_pred_flat[mask] - y_true_flat[mask]
    if len(x_plot) > 30000:
        idx_sample = np.random.choice(len(x_plot), 30000, replace=False)
        x_res = x_plot[idx_sample]
        res_plot = residuals[idx_sample]
    else:
        x_res = x_plot
        res_plot = residuals

    ax.hexbin(x_res, res_plot, xscale="log", gridsize=80, cmap="RdBu_r", mincnt=1)
    ax.axhline(0, color="black", linestyle="-", linewidth=1.5, alpha=0.8)
    ax.axhline(mae_linear, color="red", linestyle="--", linewidth=1, alpha=0.6, label="+MAE")
    ax.axhline(-mae_linear, color="red", linestyle="--", linewidth=1, alpha=0.6, label="-MAE")
    ax.set_xscale("log")
    ax.set_xlabel("True Abundance", fontsize=12)
    ax.set_ylabel("Residual (Pred - True)", fontsize=12)
    ax.set_title("Residuals vs Observed Abundance", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "residual_vs_observed.png", dpi=DPI)
    plt.close()
    log.info("✓ Saved: residual_vs_observed.png")

    # Error distribution histograms
    log.info("Generating error distribution histograms...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    all_residuals = (y_pred - y_true).ravel()
    axes[0].hist(all_residuals, bins=100, alpha=0.7, color="steelblue", edgecolor="black")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0].axvline(mae_linear, color="orange", linestyle="--", linewidth=1, label=f"MAE={mae_linear:.3e}")
    axes[0].axvline(-mae_linear, color="orange", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Residual (Pred - True)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Linear Space Residuals", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    log_residuals = (y_pred_log - y_true_log).ravel()
    log_residuals = log_residuals[np.isfinite(log_residuals)]
    axes[1].hist(log_residuals, bins=100, alpha=0.7, color="coral", edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1].axvline(mae_log, color="orange", linestyle="--", linewidth=1, label=f"MAE={mae_log:.3e}")
    axes[1].axvline(-mae_log, color="orange", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual (log10 scale)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Log Space Residuals", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "error_distribution.png", dpi=DPI)
    plt.close()
    log.info("✓ Saved: error_distribution.png")

    # Worst samples
    log.info("Analyzing worst predictions...")
    sample_mae = np.mean(np.abs(y_pred - y_true), axis=1)
    worst_indices = sample_mae.argsort()[-100:][::-1]
    worst_samples = df_test.iloc[worst_indices].copy()
    worst_samples["sample_MAE"] = sample_mae[worst_indices]
    worst_samples.to_csv(OUT_DIR / "worst100_samples.csv", index=False)

    log.info("Top-5 worst predictions:")
    for idx in worst_indices[:5]:
        T = df_test.iloc[idx]["T_K"]
        P = df_test.iloc[idx]["P_bar"]
        mae_val = sample_mae[idx]
        log.info("  Sample %d: T=%.0f K, P=%.2e bar, MAE=%s", idx, T, P, fmt(mae_val))

    # Species histograms
    log.info("Generating individual species histograms...")
    top5_species = df_species.nlargest(5, "mean_abundance")["species"].values
    for sp in top5_species:
        idx = target_cols.index(sp)
        fig, ax = plt.subplots(figsize=(8, 5))
        obs_vals = y_true[:, idx]
        obs_vals_nonzero = obs_vals[obs_vals > CLIP]
        if len(obs_vals_nonzero) > 0:
            ax.hist(
                obs_vals_nonzero,
                bins=np.logspace(np.log10(CLIP), 0, 80),
                alpha=0.7,
                color="steelblue",
                edgecolor="black",
            )
            ax.set_xscale("log")
            ax.set_xlabel("Abundance", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(f"Distribution: {sp}", fontsize=12)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = sp.replace("/", "_").replace("+", "p").replace("-", "m")
        plt.savefig(OUT_DIR / f"hist_obs_{safe_name}.png", dpi=DPI)
        plt.close()
    log.info("✓ Saved: hist_obs_*.png for top-5 species")

    # Summary report
    log.info("Writing summary report...")
    with open(OUT_DIR / "diagnostic_summary.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("UPDATED_VERS AUTOENCODER DIAGNOSTIC SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("GLOBAL METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Linear MAE:        {mae_linear:.6e}\n")
        f.write(f"Linear R²:         {r2_linear:.6f}\n")
        f.write(f"Log MAE (dex):     {mae_log:.6e}\n")
        f.write(f"Log R²:            {r2_log:.6f}\n")
        f.write(f"Test samples:      {len(y_true):,}\n")
        f.write(f"Species predicted: {len(target_cols)}\n\n")

        f.write("TOP-5 SPECIES BY ABUNDANCE\n")
        f.write("-" * 70 + "\n")
        for _, row in df_species.nlargest(5, "mean_abundance").iterrows():
            f.write(
                f"{row['species']:15s}: mean={row['mean_abundance']:.3e}, "
                f"MAE={row['MAE']:.3e}, R²={row['R2']:.4f}\n"
            )

        f.write("\nTOP-5 SPECIES BY ERROR\n")
        f.write("-" * 70 + "\n")
        for _, row in df_species.nlargest(5, "MAE").iterrows():
            f.write(
                f"{row['species']:15s}: MAE={row['MAE']:.3e}, "
                f"mean_abun={row['mean_abundance']:.3e}\n"
            )

        f.write("\n" + "=" * 70 + "\n")
        f.write("PLOTS GENERATED\n")
        f.write("=" * 70 + "\n")
        f.write("• parity_top10.png - Parity plots for 10 most abundant species\n")
        f.write("• parity_overall.png - All species pooled parity plot\n")
        f.write("• MAE_per_species.png - Error breakdown\n")
        f.write("• residual_vs_observed.png - Residual analysis\n")
        f.write("• error_distribution.png - Error histograms\n")
        f.write("• hist_obs_*.png - Individual species distributions (top-5)\n")
        f.write("• per_species_errors.csv - Complete error table\n")
        f.write("• worst100_samples.csv - Samples with highest errors\n")
        f.write("=" * 70 + "\n")

    log.info("=" * 70)
    log.info("DIAGNOSTICS COMPLETE")
    log.info("=" * 70)
    log.info("Generated files in: %s", OUT_DIR)


if __name__ == "__main__":
    main()


