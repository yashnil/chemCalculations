#!/usr/bin/env python3
"""
plot_compare_x4800_three.py
===========================

Comparative plots between the three x4800 models:
  - x4800_optimal_retrained (baseline FlowMap)
  - x4800_improved (AdamW FlowMap, best)
  - x4800_mlp (6×1024 MLP)

Generates:
  1. compare_x4800_three_metrics.png - Bar chart of test_loss, log_mae, log_r2
  2. compare_x4800_three_parity.png - 3-panel parity plot (predicted vs true)
  3. compare_x4800_three_per_species.png - Per-species Log MAE comparison

Usage:
    python scripts/plot_compare_x4800_three.py [--output-dir plots]
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

X4800_MODELS = [
    ("x4800_optimal_retrained", "x4800 baseline", "steelblue"),
    ("x4800_improved", "x4800 improved", "forestgreen"),
    ("x4800_mlp", "x4800 MLP", "coral"),
]

if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

plt.style.use("seaborn-v0_8-whitegrid")


def load_best_model_module(run_tag: str):
    """Load the best_model.py module for a run."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    best_path = run_dir / "best_model.py"
    if not best_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("best_model", best_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_predictions(run_tag: str, df_test: pd.DataFrame, target_cols: list):
    """Run model and return (y_true, y_pred) on df_test."""
    mod = load_best_model_module(run_tag)
    if mod is None:
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mod.load_model(device=device)
    model.eval()
    X = mod.normalize_inputs(df_test)
    with torch.no_grad():
        pred_scaled = mod.forward_autoencoder(model, X)
        pred_scaled = pred_scaled.cpu().numpy()
    y_pred = mod.denormalize_targets(pred_scaled)
    y_true = df_test[target_cols].to_numpy(dtype=np.float64, copy=True)
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return y_true, y_pred


def plot_metrics_bar(output_path: Path):
    """Bar chart: test_loss, log_mae, log_r2 for the 3 models."""
    if not COMPARISON_CSV.exists():
        print(f"⚠️  {COMPARISON_CSV} not found")
        return

    df = pd.read_csv(COMPARISON_CSV)
    tag_to_color = {t: c for t, _, c in X4800_MODELS}
    rows = []
    for tag, label, _ in X4800_MODELS:
        r = df[df["dataset"] == tag]
        if not r.empty:
            rows.append({"tag": tag, "label": label, **r.iloc[0].to_dict()})

    if len(rows) < 2:
        print("⚠️  Need at least 2 models in comparison_metrics.csv")
        return

    metrics = ["test_loss", "log_mae", "log_r2"]
    labels = ["Test Loss\n(log_ratio)", "Log MAE\n(dex)", "Log R²"]
    x = np.arange(len(rows))
    colors = [tag_to_color[r["tag"]] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, ylabel in zip(axes, metrics, labels):
        vals = [r[metric] for r in rows]
        bars = ax.bar(x, vals, width=0.6, color=colors, alpha=0.85, edgecolor="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"].replace(" ", "\n") for r in rows], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel.replace("\n", " "), fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            h = bar.get_height()
            fmt = f"{v:.4f}" if metric != "log_r2" else f"{v:.5f}"
            ax.annotate(fmt, xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
        if metric in ("test_loss", "log_mae"):
            ax.set_yscale("log")

    plt.suptitle("x4800 Model Comparison: Baseline vs Improved vs MLP", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved metrics bar chart to {output_path}")
    plt.close()


def load_test_data_and_predictions():
    """Load test data and run inference for all 3 models. Returns (df_test, target_cols, predictions) or None."""
    mod_ref = load_best_model_module("x4800_improved")
    if mod_ref is None:
        print("⚠️  x4800_improved not found")
        return None

    csv_path = BASE_DIR / "data" / "datasets" / "all_gas_fastchem_x4800.csv"
    if not csv_path.exists():
        print("⚠️  Dataset not found")
        return None

    df = pd.read_csv(csv_path)
    splits = getattr(mod_ref, "SPLITS", {})
    test_idx = np.array(splits.get("test_idx", []), dtype=int)
    if len(test_idx) == 0:
        print("⚠️  No test indices")
        return None

    df_test = df.iloc[test_idx].reset_index(drop=True)
    target_cols = mod_ref.TARGET_COLS

    predictions = {}
    for tag, label, _ in X4800_MODELS:
        print(f"  Loading {tag}...")
        y_true, y_pred = get_predictions(tag, df_test, target_cols)
        if y_true is not None:
            predictions[tag] = (y_true, y_pred)

    return df_test, target_cols, predictions


def plot_parity_three(output_path: Path, predictions: dict | None):
    """3-panel parity plot: predicted vs true for each model."""
    if predictions is None or not predictions:
        print("⚠️  No predictions, skipping parity plot")
        return

    mod_ref = load_best_model_module("x4800_improved")
    if mod_ref is None:
        return
    target_cols = mod_ref.TARGET_COLS

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    MAX_POINTS = 80_000

    for ax, (tag, label, color) in zip(axes, X4800_MODELS):
        if tag not in predictions:
            ax.text(0.5, 0.5, f"{label}\n(not found)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xscale("log")
            ax.set_yscale("log")
            continue

        y_true, y_pred = predictions[tag]
        if y_true is None:
            ax.text(0.5, 0.5, f"{label}\n(not found)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xscale("log")
            ax.set_yscale("log")
            continue

        yt_flat = y_true.reshape(-1)
        yp_flat = y_pred.reshape(-1)
        mask = (yt_flat > 1e-30) & (yp_flat > 1e-30)
        x_plot = yt_flat[mask]
        y_plot = yp_flat[mask]

        if len(x_plot) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            continue

        log_x = np.log10(x_plot)
        log_y = np.log10(y_plot)
        log_mae = mean_absolute_error(log_x, log_y)
        log_r2 = r2_score(log_x, log_y)

        if len(x_plot) > MAX_POINTS:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(x_plot), MAX_POINTS, replace=False)
            x_plot, y_plot = x_plot[idx], y_plot[idx]

        ax.scatter(x_plot, y_plot, s=4, alpha=0.4, c=color, edgecolors="none")
        lims = [max(x_plot.min(), y_plot.min(), 1e-30), min(x_plot.max(), y_plot.max())]
        ax.plot(lims, lims, "k--", linewidth=1.5, label="1:1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"True (cm$^{-3}$)", fontsize=10)
        ax.set_ylabel(r"Predicted (cm$^{-3}$)", fontsize=10)
        ax.set_title(f"{label}\nLog MAE={log_mae:.4f}, Log R²={log_r2:.4f}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    plt.suptitle("Predicted vs True (same test set)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved 3-panel parity plot to {output_path}")
    plt.close()


def plot_per_species_comparison(output_path: Path, df_test: pd.DataFrame, target_cols: list, predictions: dict):
    """Per-species Log MAE comparison across the 3 models."""
    if len(predictions) < 2:
        print("⚠️  Need at least 2 models for per-species comparison")
        return

    results = {}
    for tag, label, _ in X4800_MODELS:
        if tag not in predictions:
            continue
        y_true, y_pred = predictions[tag]
        log_mae_per_sp = []
        for i in range(len(target_cols)):
            yt = y_true[:, i]
            yp = y_pred[:, i]
            mask = (yt > 1e-30) & (yp > 1e-30)
            if mask.sum() > 10:
                log_mae_per_sp.append(mean_absolute_error(np.log10(yt[mask]), np.log10(yp[mask])))
            else:
                log_mae_per_sp.append(np.nan)
        results[tag] = {"label": label, "log_mae": np.array(log_mae_per_sp)}

    if len(results) < 2:
        print("⚠️  Need at least 2 models for per-species comparison")
        return

    # Sort species by mean abundance (from true values)
    mean_abun = df_test[target_cols].mean().values
    order = np.argsort(-mean_abun)[:20]  # Top 20 species

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(order))
    width = 0.25
    colors = {"x4800_optimal_retrained": "steelblue", "x4800_improved": "forestgreen", "x4800_mlp": "coral"}

    for i, (tag, data) in enumerate(results.items()):
        offset = (i - len(results) / 2 + 0.5) * width
        vals = data["log_mae"][order]
        vals = np.where(np.isfinite(vals) & (vals > 0), vals, 1e-6)  # avoid zeros for log scale
        ax.bar(x + offset, vals, width=0.8 * width, label=data["label"], color=colors.get(tag, "gray"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([target_cols[j] for j in order], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Log MAE (dex)", fontsize=12)
    ax.set_xlabel("Species (top 20 by abundance)", fontsize=12)
    ax.set_title("Per-Species Log MAE: x4800 Baseline vs Improved vs MLP", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved per-species comparison to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("x4800 THREE-WAY COMPARISON PLOTS")
    print("=" * 70)
    print(f"Models: x4800_optimal_retrained, x4800_improved, x4800_mlp")
    print(f"Output: {args.output_dir}")
    print()

    plot_metrics_bar(args.output_dir / "compare_x4800_three_metrics.png")

    data = load_test_data_and_predictions()
    if data is not None:
        df_test, target_cols, predictions = data
        plot_parity_three(args.output_dir / "compare_x4800_three_parity.png", predictions)
        plot_per_species_comparison(
            args.output_dir / "compare_x4800_three_per_species.png", df_test, target_cols, predictions
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
