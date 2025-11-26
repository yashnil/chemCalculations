#!/usr/bin/env python3
# plot_test_autoencoder.py
#
# Mirrors NEW_VERS/plot.py but targets the FlowMap autoencoder so that plots
# and scalar diagnostics are directly comparable.

from __future__ import annotations

import os
import time
import importlib.util
import logging
from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

try:
    plt.style.use("science.mplstyle")
except OSError:
    pass

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CSV_PATH = os.environ.get(
    "CSV_PATH",
    "/Users/yashnilmohanty/Desktop/chemCalculations/NEW_VERS/all_gas_v10_no_stripe_clean.csv",
)
BEST_MODULE = os.environ.get("BEST_MODULE", os.path.join("runs_autoencoder", "best_model.py"))
OUT_PNG = os.environ.get("OUT_PNG", os.path.join("runs_autoencoder", "pred_vs_true_test.png"))

LOG10_AXES = True
CLAMP_SCALED_BEFORE_DENORM = True
CLAMP_MIN_SCALED = -1.0
CLAMP_MAX_SCALED = 0.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("plot_autoencoder")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fmt(x) -> str:
    try:
        return f"{float(x):.3e}"
    except Exception:
        return str(x)


def load_module_from_path(path: str):
    path = os.path.abspath(path)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def targets_to_scaled(
    y_linear: np.ndarray, zero_floor: float, log_eps: float, log_scale: float
) -> np.ndarray:
    y = y_linear.copy()
    y[y < zero_floor] = 0.0
    y = np.log10(np.maximum(y, log_eps))
    y /= log_scale
    return y


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    best_mod = load_module_from_path(BEST_MODULE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device.type)

    input_cols: List[str] = best_mod.INPUT_COLS
    target_cols: List[str] = best_mod.TARGET_COLS

    log.info("Normalization (inputs):")
    log.info("  - T_K: x / %s", fmt(best_mod.TEMP_DIVISOR))
    log.info("  - P_bar: log10(safe(x)) / %s", fmt(best_mod.INPUT_LOG_SCALE))
    if "fZ_dex" in input_cols:
        log.info("  - fZ_dex: x / %s", fmt(best_mod.INPUT_LOG_SCALE))
    if "fZ" in input_cols:
        log.info("  - fZ: log10(safe(x)) / %s", fmt(best_mod.INPUT_LOG_SCALE))
    log.info(
        "  - abund_*_dex: (epsilon - %.1f) / %.0f",
        float(best_mod.ABUND_EPSILON_OFFSET),
        float(best_mod.ABUND_DEX_SCALE),
    )

    log.info("Normalization (targets):")
    log.info(
        "  (x < %s) -> 0, log10(max(x, %s)), divide by %s",
        fmt(best_mod.TARGET_ZERO_FLOOR),
        fmt(best_mod.LOG_EPS),
        fmt(best_mod.TARGET_LOG_SCALE),
    )

    model = best_mod.load_model(device=device)

    csv_path = os.path.abspath(CSV_PATH)
    log.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("Loaded: %d rows × %d cols", df.shape[0], df.shape[1])

    splits = getattr(best_mod, "SPLITS", None)
    if not splits or "test_idx" not in splits:
        raise RuntimeError("best_model.py is missing test indices; re-train to export splits.")
    test_idx = np.asarray(splits["test_idx"], dtype=int)
    df_test = df.iloc[test_idx].copy()

    log.info("Using TEST split only: rows=%d (of %d total)", len(df_test), len(df))

    X_norm = best_mod.normalize_inputs(df_test)
    X_norm = X_norm.to(device)

    def _sync_if_cuda():
        if device.type == "cuda":
            torch.cuda.synchronize()

    _sync_if_cuda()
    t0 = time.perf_counter()
    with torch.no_grad():
        y_scaled_pred_tensor = best_mod.forward_autoencoder(model, X_norm)
    _sync_if_cuda()
    t1 = time.perf_counter()
    batch_time = t1 - t0

    y_scaled_pred = y_scaled_pred_tensor.cpu().numpy().astype(np.float64)
    n_samples = y_scaled_pred.shape[0]
    log.info(
        "Inference timing (batch): total=%.6f s | avg/sample=%.6f s | samples=%d",
        batch_time,
        batch_time / float(max(1, n_samples)),
        n_samples,
    )

    y_true_linear = df_test[target_cols].to_numpy(dtype=np.float64)
    y_scaled_true = targets_to_scaled(
        y_true_linear,
        zero_floor=float(best_mod.TARGET_ZERO_FLOOR),
        log_eps=float(best_mod.LOG_EPS),
        log_scale=float(best_mod.TARGET_LOG_SCALE),
    )

    mse_scaled = float(np.mean((y_scaled_pred - y_scaled_true) ** 2))

    if CLAMP_SCALED_BEFORE_DENORM:
        y_scaled_for_denorm = np.clip(y_scaled_pred, CLAMP_MIN_SCALED, CLAMP_MAX_SCALED)
        log.info(
            "Clamping scaled predictions to [%.1f, %.1f] before denorm.",
            CLAMP_MIN_SCALED,
            CLAMP_MAX_SCALED,
        )
    else:
        y_scaled_for_denorm = y_scaled_pred
        log.info("No clamping before denorm (linear predictions may exceed [1e-30, 1]).")

    y_pred_linear = best_mod.denormalize_targets(y_scaled_for_denorm).astype(np.float64)
    y_true_linear_floor = y_true_linear.copy()
    y_true_linear_floor[y_true_linear_floor < float(best_mod.TARGET_ZERO_FLOOR)] = 0.0
    mse_linear = float(np.mean((y_pred_linear - y_true_linear_floor) ** 2))

    log.info("MSE (scaled training space): %s", fmt(mse_scaled))
    log.info("MSE (linear space)         : %s", fmt(mse_linear))

    y_pred_flat = y_pred_linear.reshape(-1)
    y_true_flat = y_true_linear_floor.reshape(-1)

    if LOG10_AXES:
        mask = (y_true_flat > 0.0) & (y_pred_flat > 0.0)
        x_plot = y_true_flat[mask]
        y_plot = y_pred_flat[mask]
    else:
        x_plot = y_true_flat
        y_plot = y_pred_flat

    if x_plot.size and y_plot.size:
        vmin = float(min(x_plot.min(), y_plot.min()))
        vmax = float(max(x_plot.max(), y_plot.max()))
    else:
        vmin, vmax = 1e-30, 1.0
    if not np.isfinite(vmin):
        vmin = 1e-30
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin * 10.0

    plt.figure(figsize=(7.5, 7.0))
    plt.scatter(x_plot, y_plot, s=8, alpha=0.6)
    plt.plot([vmin, vmax], [vmin, vmax], linewidth=3.0, zorder=10, color="black")
    plt.xlabel("FastChem Abundance")
    plt.ylabel("Predicted Abundance")
    if LOG10_AXES and x_plot.size:
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(1e-30, 1)
        plt.ylim(1e-30, 1)
    plt.tight_layout()

    out_path = os.path.abspath(OUT_PNG)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    log.info("Saved plot: %s", out_path)


if __name__ == "__main__":
    main()


