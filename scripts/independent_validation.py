#!/usr/bin/env python3
"""
independent_validation.py
=========================

Validate the best ML emulator (x4800) against FastChem on conditions that
are *completely independent* of the training data.

Validation scenarios:
  1. Hot Jupiter T-P profile (Madhusudhan & Seager 2009-style) at solar composition
  2. Systematic T-P grid at solar composition (regular grid, not random)
  3. C/O ratio sweep (0.1 to 2.0) at fixed T, P
  4. Metallicity sweep (0.01× to 100× solar) at fixed T, P
  5. Cool dwarf T-P profile (T Dwarf-like)

All conditions are physically motivated and structurally different from the
empirical resampling + jitter used in training data generation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "results" / "runs" / "runs_autoencoder_x4800_optimal_retrained"))

import best_model as bm

try:
    import pyfastchem
except ImportError:
    from pyfastchem import FastChem  # noqa: F401
    import pyfastchem

LOGK = BASE_DIR / "data" / "fastchem_data" / "Kitzmann2023" / "logK.dat"
LOGK_COND = BASE_DIR / "data" / "fastchem_data" / "Kitzmann2023" / "logK_condensates.dat"
ELEM_ABUND = BASE_DIR / "data" / "fastchem_data" / "lodders_2003_extended.dat"
OUT_DIR = BASE_DIR / "plots" / "independent_validation"

SOLAR = {"H": 12.00, "O": 8.69, "C": 8.43, "N": 7.83, "S": 7.12}


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

def hot_jupiter_profile() -> pd.DataFrame:
    """Madhusudhan & Seager style T-P profile for a canonical hot Jupiter."""
    log_P = np.linspace(-6, 2, 60)
    P_bar = 10.0 ** log_P
    T_deep = 2200.0
    T_top = 800.0
    T_K = T_top + (T_deep - T_top) * (1 - np.exp(-P_bar / 0.1))
    T_K = np.clip(T_K, 800, 3000)
    rows = []
    for t, p in zip(T_K, P_bar):
        rows.append({"T_K": t, "P_bar": p, **{f"abund_{e}_dex": v for e, v in SOLAR.items()}})
    df = pd.DataFrame(rows)
    df["scenario"] = "hot_jupiter_profile"
    return df


def cool_dwarf_profile() -> pd.DataFrame:
    """T-dwarf-like atmosphere: cooler with deep high-pressure layers."""
    log_P = np.linspace(-4, 3, 50)
    P_bar = 10.0 ** log_P
    T_deep = 1800.0
    T_top = 900.0
    T_K = T_top + (T_deep - T_top) * (log_P - log_P.min()) / (log_P.max() - log_P.min())
    T_K = np.clip(T_K, 900, 3000)
    rows = []
    for t, p in zip(T_K, P_bar):
        rows.append({"T_K": t, "P_bar": p, **{f"abund_{e}_dex": v for e, v in SOLAR.items()}})
    df = pd.DataFrame(rows)
    df["scenario"] = "cool_dwarf_profile"
    return df


def systematic_tp_grid() -> pd.DataFrame:
    """Regular T-P grid at solar composition."""
    T_vals = np.linspace(800, 3000, 12)
    log_P_vals = np.linspace(-5, 3, 10)
    rows = []
    for T in T_vals:
        for lp in log_P_vals:
            rows.append({
                "T_K": T, "P_bar": 10.0 ** lp,
                **{f"abund_{e}_dex": v for e, v in SOLAR.items()},
            })
    df = pd.DataFrame(rows)
    df["scenario"] = "tp_grid_solar"
    return df


def co_ratio_sweep() -> pd.DataFrame:
    """Sweep C/O ratio from 0.1 to 2.0 at two representative (T, P) points."""
    co_ratios = np.linspace(0.1, 2.0, 20)
    conditions = [(1500, 0.1), (2500, 10.0)]
    rows = []
    for T, P in conditions:
        for co in co_ratios:
            C_dex = SOLAR["O"] + np.log10(co)
            rows.append({
                "T_K": T, "P_bar": P,
                "abund_H_dex": SOLAR["H"],
                "abund_O_dex": SOLAR["O"],
                "abund_C_dex": C_dex,
                "abund_N_dex": SOLAR["N"],
                "abund_S_dex": SOLAR["S"],
            })
    df = pd.DataFrame(rows)
    df["scenario"] = "co_ratio_sweep"
    return df


def metallicity_sweep() -> pd.DataFrame:
    """Sweep overall metallicity from 0.01× to 100× solar at two (T, P) points."""
    log_metal = np.linspace(-2, 2, 25)
    conditions = [(1500, 0.1), (2500, 10.0)]
    rows = []
    for T, P in conditions:
        for lm in log_metal:
            rows.append({
                "T_K": T, "P_bar": P,
                "abund_H_dex": SOLAR["H"],
                "abund_O_dex": SOLAR["O"] + lm,
                "abund_C_dex": SOLAR["C"] + lm,
                "abund_N_dex": SOLAR["N"] + lm,
                "abund_S_dex": SOLAR["S"] + lm,
            })
    df = pd.DataFrame(rows)
    df["scenario"] = "metallicity_sweep"
    return df


# ---------------------------------------------------------------------------
# FastChem runner
# ---------------------------------------------------------------------------

def run_fastchem_on_conditions(df: pd.DataFrame) -> np.ndarray:
    """Run FastChem on each row; return (n_rows, n_target_species) number densities."""
    engine = pyfastchem.FastChem(str(ELEM_ABUND), str(LOGK), str(LOGK_COND), 0)
    element_symbols = [engine.getElementSymbol(i) for i in range(engine.getElementNumber())]
    base_vec = np.array(engine.getElementAbundances(), dtype=np.float64, copy=True)
    gas_names = [engine.getGasSpeciesSymbol(i) for i in range(engine.getGasSpeciesNumber())]

    target_indices = []
    for sp in bm.TARGET_COLS:
        if sp in gas_names:
            target_indices.append(gas_names.index(sp))
        else:
            target_indices.append(-1)

    results = np.full((len(df), len(bm.TARGET_COLS)), np.nan, dtype=np.float64)
    n_fail = 0

    for row_idx in range(len(df)):
        eng = pyfastchem.FastChem(str(ELEM_ABUND), str(LOGK), str(LOGK_COND), 0)
        vec = base_vec.copy()
        for idx, sym in enumerate(element_symbols):
            col = f"abund_{sym}_dex"
            if col in df.columns:
                vec[idx] = 10.0 ** (df.iloc[row_idx][col] - 12.0)
        eng.setElementAbundances(vec.tolist())

        inp = pyfastchem.FastChemInput()
        out = pyfastchem.FastChemOutput()
        inp.temperature = np.array([df.iloc[row_idx]["T_K"]], dtype=np.float64)
        inp.pressure = np.array([df.iloc[row_idx]["P_bar"]], dtype=np.float64)

        flag = eng.calcDensities(inp, out)
        if flag != pyfastchem.FASTCHEM_SUCCESS:
            n_fail += 1
            continue

        densities = np.array(out.number_densities[0], dtype=np.float64)
        for sp_idx, gi in enumerate(target_indices):
            if gi >= 0:
                results[row_idx, sp_idx] = densities[gi]

    if n_fail:
        print(f"  [warn] FastChem failed for {n_fail}/{len(df)} conditions")
    return results


# ---------------------------------------------------------------------------
# ML model runner
# ---------------------------------------------------------------------------

def run_model_on_conditions(df: pd.DataFrame, model=None, device: torch.device | None = None) -> np.ndarray:
    """Run the x4800 ML emulator; return (n_rows, n_target_species) number densities."""
    if device is None:
        device = torch.device("cpu")
    if model is None:
        model = bm.load_model(device=device)
        model.eval()
    X = bm.normalize_inputs(df)
    if device.type == "mps":
        X = X.to(device)
    with torch.no_grad():
        y_scaled = bm.forward_autoencoder(model, X).cpu().numpy()
    return bm.denormalize_targets(y_scaled)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, floor: float = 1e-30) -> Dict:
    mask = (y_true > floor) & (y_pred > floor) & np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"log_mae": np.nan, "log_r2": np.nan, "log_max_err": np.nan, "n_valid": 0}
    lt = np.log10(y_true[mask])
    lp = np.log10(y_pred[mask])
    residuals = lp - lt
    log_mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((lt - lt.mean()) ** 2)
    log_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "log_mae": log_mae,
        "log_r2": log_r2,
        "log_max_err": np.max(np.abs(residuals)),
        "log_median_err": np.median(np.abs(residuals)),
        "mean_bias": np.mean(residuals),
        "n_valid": int(mask.sum()),
    }


def per_species_metrics(y_true: np.ndarray, y_pred: np.ndarray, floor: float = 1e-30) -> pd.DataFrame:
    rows = []
    for i, sp in enumerate(bm.TARGET_COLS):
        m = compute_metrics(y_true[:, i:i+1], y_pred[:, i:i+1], floor=floor)
        m["species"] = sp
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path):
    floor = 1e-30
    mask = (y_true > floor) & (y_pred > floor) & np.isfinite(y_true) & np.isfinite(y_pred)
    xt = y_true[mask]
    xp = y_pred[mask]
    if len(xt) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xt, xp, s=8, alpha=0.5, c="steelblue", edgecolors="none")
    lims = [max(xt.min(), xp.min()) * 0.5, min(xt.max(), xp.max()) * 2]
    ax.plot(lims, lims, "k--", lw=1.5, label="1:1")
    xx = np.geomspace(lims[0], lims[1], 100)
    ax.plot(xx, xx * 10**0.1, "r--", lw=0.8, alpha=0.5, label="±0.1 dex")
    ax.plot(xx, xx * 10**(-0.1), "r--", lw=0.8, alpha=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("FastChem (ground truth)", fontsize=13)
    ax.set_ylabel("ML Emulator prediction", fontsize=13)
    m = compute_metrics(y_true, y_pred)
    ax.set_title(f"{title}\nLog MAE = {m['log_mae']:.4f} dex  |  Log R² = {m['log_r2']:.6f}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_residual_histogram(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path):
    floor = 1e-30
    mask = (y_true > floor) & (y_pred > floor) & np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return
    residuals = np.log10(y_pred[mask]) - np.log10(y_true[mask])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuals, bins=100, color="steelblue", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", ls="--", lw=1.2)
    ax.axvline(np.mean(residuals), color="red", ls="-", lw=1.5, label=f"mean = {np.mean(residuals):.4f}")
    ax.set_xlabel("log₁₀(predicted) − log₁₀(true)  [dex]", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_profile_comparison(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                            species_list: List[str], title: str, path: Path):
    """Plot abundance profiles (vs pressure) for select species."""
    n_sp = len(species_list)
    fig, axes = plt.subplots(1, n_sp, figsize=(4.5 * n_sp, 6), sharey=True)
    if n_sp == 1:
        axes = [axes]

    sp_idx = {sp: i for i, sp in enumerate(bm.TARGET_COLS)}
    pressures = df["P_bar"].values

    for ax, sp in zip(axes, species_list):
        idx = sp_idx.get(sp)
        if idx is None:
            continue
        true_vals = y_true[:, idx]
        pred_vals = y_pred[:, idx]
        valid = (true_vals > 1e-30) & (pred_vals > 1e-30)
        ax.plot(true_vals[valid], pressures[valid], "k-", lw=2, label="FastChem")
        ax.plot(pred_vals[valid], pressures[valid], "r--", lw=2, label="ML Emulator")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_xlabel(f"{sp} number density", fontsize=11)
        ax.set_title(sp, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Pressure [bar]", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_sweep(sweep_vals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
               species_list: List[str], xlabel: str, title: str, path: Path,
               conditions: List[Tuple[float, float]], n_per_cond: int):
    """Plot species abundance vs sweep parameter for each condition."""
    sp_idx = {sp: i for i, sp in enumerate(bm.TARGET_COLS)}
    n_cond = len(conditions)
    n_sp = len(species_list)
    fig, axes = plt.subplots(n_cond, n_sp, figsize=(4.5 * n_sp, 4.5 * n_cond), squeeze=False)

    for ci, (T, P) in enumerate(conditions):
        sl = slice(ci * n_per_cond, (ci + 1) * n_per_cond)
        x = sweep_vals[sl]
        for si, sp in enumerate(species_list):
            idx = sp_idx.get(sp)
            if idx is None:
                continue
            ax = axes[ci, si]
            tv = y_true[sl, idx]
            pv = y_pred[sl, idx]
            valid = (tv > 1e-30) & (pv > 1e-30)
            if valid.sum() > 0:
                ax.plot(x[valid], tv[valid], "k-o", ms=4, lw=1.5, label="FastChem")
                ax.plot(x[valid], pv[valid], "r--s", ms=4, lw=1.5, label="ML Emulator")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_title(f"{sp} @ T={T}K, P={P}bar", fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _sync_mps():
    """Synchronize MPS to get accurate timing."""
    if hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _bench_scenario(name: str, df: pd.DataFrame, key_species: list,
                    all_metrics: list, has_mps: bool,
                    model_cpu=None, model_mps=None,
                    plot_fn=None):
    """Run FastChem + ML (CPU & optionally MPS GPU) on a scenario; return (y_true, y_pred_cpu)."""
    n = len(df)
    print(f"  Running FastChem ({n} conditions)...")
    t0 = time.time()
    y_true = run_fastchem_on_conditions(df)
    fc_time = time.time() - t0
    fc_per_eval_ms = (fc_time / n) * 1000

    # Pre-compute normalized inputs once
    X_cpu = bm.normalize_inputs(df)

    print(f"  Running ML emulator on CPU (inference only)...")
    # Warmup
    with torch.no_grad():
        _ = bm.forward_autoencoder(model_cpu, X_cpu)
    t0 = time.time()
    with torch.no_grad():
        y_scaled_cpu = bm.forward_autoencoder(model_cpu, X_cpu).cpu().numpy()
    ml_cpu_time = time.time() - t0
    cpu_per_eval_ms = (ml_cpu_time / n) * 1000
    y_pred_cpu = bm.denormalize_targets(y_scaled_cpu)

    ml_gpu_time = None
    gpu_per_eval_ms = None
    if has_mps and model_mps is not None:
        print(f"  Running ML emulator on MPS GPU (inference only)...")
        X_mps = X_cpu.to(torch.device("mps"))
        # Warmup
        with torch.no_grad():
            _ = bm.forward_autoencoder(model_mps, X_mps)
        _sync_mps()
        t0 = time.time()
        with torch.no_grad():
            _ = bm.forward_autoencoder(model_mps, X_mps)
        _sync_mps()
        ml_gpu_time = time.time() - t0
        gpu_per_eval_ms = (ml_gpu_time / n) * 1000

    m = compute_metrics(y_true, y_pred_cpu)
    m["scenario"] = name
    m["n_conditions"] = n
    m["fastchem_time_s"] = fc_time
    m["fastchem_ms_per_eval"] = fc_per_eval_ms
    m["ml_cpu_time_s"] = ml_cpu_time
    m["ml_cpu_ms_per_eval"] = cpu_per_eval_ms
    m["cpu_speedup"] = fc_per_eval_ms / max(cpu_per_eval_ms, 1e-9)
    if ml_gpu_time is not None:
        m["ml_gpu_time_s"] = ml_gpu_time
        m["ml_gpu_ms_per_eval"] = gpu_per_eval_ms
        m["gpu_speedup"] = fc_per_eval_ms / max(gpu_per_eval_ms, 1e-9)
    all_metrics.append(m)

    print(f"  Log MAE = {m['log_mae']:.4f} dex, Log R² = {m['log_r2']:.6f}")
    print(f"  FastChem: {fc_time:.2f}s ({fc_per_eval_ms:.2f} ms/eval)")
    print(f"  ML CPU:   {ml_cpu_time:.6f}s ({cpu_per_eval_ms:.4f} ms/eval) → {m['cpu_speedup']:.0f}× speedup")
    if ml_gpu_time is not None:
        print(f"  ML GPU:   {ml_gpu_time:.6f}s ({gpu_per_eval_ms:.4f} ms/eval) → {m['gpu_speedup']:.0f}× speedup")

    if plot_fn:
        plot_fn(y_true, y_pred_cpu)

    return y_true, y_pred_cpu


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    has_mps = torch.backends.mps.is_available()

    print("=" * 80)
    print("INDEPENDENT VALIDATION: ML Emulator vs FastChem")
    print("=" * 80)
    print(f"Model: x4800_optimal_retrained")
    print(f"Device: CPU" + (" + MPS GPU" if has_mps else " (MPS not available)"))
    print(f"Output: {OUT_DIR}")
    print()

    # Load models once upfront (exclude load time from benchmarks)
    print("Loading ML model on CPU...")
    model_cpu = bm.load_model(device=torch.device("cpu"))
    model_cpu.eval()
    model_mps = None
    if has_mps:
        print("Loading ML model on MPS GPU...")
        model_mps = bm.load_model(device=torch.device("mps"))
        model_mps.eval()
    print()

    key_species = ["H2O1", "C1O1", "C1H4", "H2", "N2", "H3N1", "C1O2", "H2S1"]
    all_metrics = []

    # --- Scenario 1: Hot Jupiter profile ---
    print("Scenario 1: Hot Jupiter T-P profile (solar composition)")
    df_hj = hot_jupiter_profile()
    def _plot_hj(yt, yp):
        plot_parity(yt, yp, "Hot Jupiter Profile: Predicted vs True",
                    OUT_DIR / "parity_hot_jupiter.png")
        plot_profile_comparison(df_hj, yt, yp, key_species[:4],
                               "Hot Jupiter T-P Profile", OUT_DIR / "profile_hot_jupiter.png")
    y_true_hj, y_pred_hj = _bench_scenario(
        "hot_jupiter_profile", df_hj, key_species, all_metrics, has_mps,
        model_cpu=model_cpu, model_mps=model_mps, plot_fn=_plot_hj)
    print()

    # --- Scenario 2: Cool dwarf profile ---
    print("Scenario 2: Cool dwarf T-P profile (solar composition)")
    df_cd = cool_dwarf_profile()
    def _plot_cd(yt, yp):
        plot_parity(yt, yp, "Cool Dwarf Profile: Predicted vs True",
                    OUT_DIR / "parity_cool_dwarf.png")
        plot_profile_comparison(df_cd, yt, yp, key_species[:4],
                               "Cool Dwarf T-P Profile", OUT_DIR / "profile_cool_dwarf.png")
    y_true_cd, y_pred_cd = _bench_scenario(
        "cool_dwarf_profile", df_cd, key_species, all_metrics, has_mps,
        model_cpu=model_cpu, model_mps=model_mps, plot_fn=_plot_cd)
    print()

    # --- Scenario 3: T-P grid ---
    print("Scenario 3: Systematic T-P grid (solar composition, 12x10 = 120 points)")
    df_grid = systematic_tp_grid()
    def _plot_grid(yt, yp):
        plot_parity(yt, yp, "T-P Grid (Solar): Predicted vs True",
                    OUT_DIR / "parity_tp_grid.png")
        plot_residual_histogram(yt, yp, "T-P Grid: Residual Distribution",
                               OUT_DIR / "residuals_tp_grid.png")
    y_true_grid, y_pred_grid = _bench_scenario(
        "tp_grid_solar", df_grid, key_species, all_metrics, has_mps,
        model_cpu=model_cpu, model_mps=model_mps, plot_fn=_plot_grid)
    print()

    # --- Scenario 4: C/O ratio sweep ---
    print("Scenario 4: C/O ratio sweep (0.1 to 2.0)")
    df_co = co_ratio_sweep()
    n_per = 20
    co_ratios = np.linspace(0.1, 2.0, n_per)
    def _plot_co(yt, yp):
        co_vals = np.tile(co_ratios, 2)
        plot_sweep(co_vals, yt, yp,
                   ["H2O1", "C1O1", "C1H4", "C1O2"], "C/O ratio",
                   "C/O Ratio Sweep", OUT_DIR / "sweep_co_ratio.png",
                   [(1500, 0.1), (2500, 10.0)], n_per)
        plot_parity(yt, yp, "C/O Sweep: Predicted vs True",
                    OUT_DIR / "parity_co_sweep.png")
    y_true_co, y_pred_co = _bench_scenario(
        "co_ratio_sweep", df_co, key_species, all_metrics, has_mps,
        model_cpu=model_cpu, model_mps=model_mps, plot_fn=_plot_co)
    print()

    # --- Scenario 5: Metallicity sweep ---
    print("Scenario 5: Metallicity sweep (0.01x to 100x solar)")
    df_met = metallicity_sweep()
    n_per_met = 25
    log_metal = np.linspace(-2, 2, n_per_met)
    def _plot_met(yt, yp):
        metal_vals = np.tile(log_metal, 2)
        plot_sweep(metal_vals, yt, yp,
                   ["H2O1", "C1O1", "C1H4", "H3N1"], "log₁₀([M/H])",
                   "Metallicity Sweep", OUT_DIR / "sweep_metallicity.png",
                   [(1500, 0.1), (2500, 10.0)], n_per_met)
        plot_parity(yt, yp, "Metallicity Sweep: Predicted vs True",
                    OUT_DIR / "parity_metallicity.png")
    y_true_met, y_pred_met = _bench_scenario(
        "metallicity_sweep", df_met, key_species, all_metrics, has_mps,
        model_cpu=model_cpu, model_mps=model_mps, plot_fn=_plot_met)
    print()

    # --- Combined parity plot ---
    print("Generating combined parity plot (all scenarios)...")
    y_true_all = np.vstack([y_true_hj, y_true_cd, y_true_grid, y_true_co, y_true_met])
    y_pred_all = np.vstack([y_pred_hj, y_pred_cd, y_pred_grid, y_pred_co, y_pred_met])
    m_all = compute_metrics(y_true_all, y_pred_all)
    m_all["scenario"] = "ALL_COMBINED"
    m_all["n_conditions"] = len(y_true_all)
    all_metrics.append(m_all)

    plot_parity(y_true_all, y_pred_all,
                "Independent Validation (All Scenarios Combined)",
                OUT_DIR / "parity_all_combined.png")
    plot_residual_histogram(y_true_all, y_pred_all,
                           "Independent Validation: Residual Distribution (All Scenarios)",
                           OUT_DIR / "residuals_all_combined.png")

    # --- Per-species breakdown ---
    print("Computing per-species metrics...")
    sp_df = per_species_metrics(y_true_all, y_pred_all)
    sp_df = sp_df.sort_values("log_mae", ascending=False)
    sp_df.to_csv(OUT_DIR / "per_species_validation.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#d62728" if v > m_all["log_mae"] else "#1f77b4" for v in sp_df["log_mae"]]
    ax.barh(range(len(sp_df)), sp_df["log_mae"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(sp_df)))
    ax.set_yticklabels(sp_df["species"], fontsize=9)
    ax.axvline(m_all["log_mae"], color="k", ls="--", lw=1.2, label=f"Overall: {m_all['log_mae']:.4f}")
    ax.set_xlabel("Log MAE (dex)", fontsize=12)
    ax.set_title("Per-Species Log MAE — Independent Validation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_species_log_mae.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Summary report ---
    print()
    print("=" * 80)
    print("INDEPENDENT VALIDATION SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(all_metrics)
    cols = ["scenario", "n_conditions", "log_mae", "log_r2", "log_max_err", "mean_bias",
            "fastchem_ms_per_eval", "ml_cpu_ms_per_eval", "cpu_speedup"]
    if has_mps:
        cols += ["ml_gpu_ms_per_eval", "gpu_speedup"]
    avail_cols = [c for c in cols if c in summary_df.columns]
    print(summary_df[avail_cols].to_string(index=False))
    summary_df.to_csv(OUT_DIR / "validation_summary.csv", index=False)

    print()
    print(f"Overall (all {m_all['n_conditions']} conditions, {m_all['n_valid']} valid points):")
    print(f"  Log MAE  = {m_all['log_mae']:.4f} dex")
    print(f"  Log R²   = {m_all['log_r2']:.6f}")
    print(f"  Max err  = {m_all['log_max_err']:.4f} dex")
    print(f"  Mean bias = {m_all['mean_bias']:.4f} dex")
    print()

    total_fc = sum(m.get("fastchem_time_s", 0) for m in all_metrics if m["scenario"] != "ALL_COMBINED")
    total_cpu = sum(m.get("ml_cpu_time_s", 0) for m in all_metrics if m["scenario"] != "ALL_COMBINED")
    print(f"Total timing across all {sum(m['n_conditions'] for m in all_metrics if m['scenario'] != 'ALL_COMBINED')} evaluations:")
    print(f"  FastChem total: {total_fc:.2f}s")
    print(f"  ML CPU total:   {total_cpu:.4f}s  ({total_fc/max(total_cpu,1e-9):.0f}× overall speedup)")
    if has_mps:
        total_gpu = sum(m.get("ml_gpu_time_s", 0) for m in all_metrics if m["scenario"] != "ALL_COMBINED" and "ml_gpu_time_s" in m)
        print(f"  ML GPU total:   {total_gpu:.4f}s  ({total_fc/max(total_gpu,1e-9):.0f}× overall speedup)")
    print()
    print(f"Plots and data saved to: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
