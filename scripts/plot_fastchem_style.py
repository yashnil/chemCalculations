#!/usr/bin/env python3
"""
plot_fastchem_style.py
======================

FastChem-paper-style plots: mixing ratios (number densities) vs T and P.

Mirrors Figure 1 of FastChem Cond (Kitzmann et al. 2023, 2309.02337):
  - Partial pressures / number densities of gas-phase species vs temperature
  - At fixed pressure (e.g. 0.5 bar, like Sharp & Huebner 1990)
  - Additional: 2D heatmap of abundance vs (T, P) for key species

Usage:
    python scripts/plot_fastchem_style.py [--model x4800_improved] [--output-dir plots]
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

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "results" / "runs"
PLOTS_DIR = BASE_DIR / "plots"

SOLAR = {"H": 12.00, "O": 8.69, "C": 8.43, "N": 7.83, "S": 7.12}


def load_best_model(run_tag: str, device: torch.device):
    """Load best_model.py from a run directory."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    best_path = run_dir / "best_model.py"
    if not best_path.exists():
        raise FileNotFoundError(f"Model not found: {best_path}")
    if str(BASE_DIR / "src") not in sys.path:
        sys.path.insert(0, str(BASE_DIR / "src"))
    spec = importlib.util.spec_from_file_location("best_model", best_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {best_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.load_model(device=device)
    model.eval()
    return mod, model


def make_tp_grid(
    t_min: float = 800,
    t_max: float = 3000,
    n_t: int = 50,
    p_fixed: float = 0.5,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Create DataFrame with T, P, and solar abundances for a 1D T-sweep at fixed P."""
    rng = rng or np.random.default_rng(42)
    T = np.linspace(t_min, t_max, n_t)
    n = len(T)
    df = pd.DataFrame({
        "T_K": T,
        "P_bar": np.full(n, p_fixed),
        "abund_H_dex": np.full(n, SOLAR["H"]),
        "abund_O_dex": SOLAR["O"] + rng.normal(0, 0.01, n),
        "abund_C_dex": SOLAR["C"] + rng.normal(0, 0.01, n),
        "abund_N_dex": SOLAR["N"] + rng.normal(0, 0.01, n),
        "abund_S_dex": SOLAR["S"] + rng.normal(0, 0.01, n),
    })
    return df


def make_tp_grid_2d(
    t_min: float = 800,
    t_max: float = 3000,
    p_min: float = 1e-4,
    p_max: float = 1e3,
    n_t: int = 40,
    n_p: int = 40,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Create 2D T-P grid with solar composition."""
    rng = rng or np.random.default_rng(42)
    T = np.linspace(t_min, t_max, n_t)
    P = np.logspace(np.log10(p_min), np.log10(p_max), n_p)
    TT, PP = np.meshgrid(T, P)
    TT = TT.ravel()
    PP = PP.ravel()
    n = len(TT)
    df = pd.DataFrame({
        "T_K": TT,
        "P_bar": PP,
        "abund_H_dex": np.full(n, SOLAR["H"]),
        "abund_O_dex": SOLAR["O"] + rng.normal(0, 0.01, n),
        "abund_C_dex": SOLAR["C"] + rng.normal(0, 0.01, n),
        "abund_N_dex": SOLAR["N"] + rng.normal(0, 0.01, n),
        "abund_S_dex": SOLAR["S"] + rng.normal(0, 0.01, n),
    })
    return df, T, P


def predict(mod, model: torch.nn.Module, df: pd.DataFrame, device: torch.device) -> np.ndarray:
    """Run model forward and return denormalized abundances (cm⁻³)."""
    X = mod.normalize_inputs(df)
    g = X.to(device) if hasattr(X, "to") else torch.as_tensor(X, dtype=torch.float32, device=device)
    y0 = torch.zeros((g.shape[0], len(mod.TARGET_COLS)), dtype=g.dtype, device=device)
    dt = torch.ones((g.shape[0], 1), dtype=g.dtype, device=device)
    with torch.no_grad():
        y_scaled = model(y0, dt, g)[:, 0, :].cpu().numpy()
    return mod.denormalize_targets(y_scaled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="x4800_improved", help="Run tag")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--p-fixed", type=float, default=0.5, help="Fixed pressure (bar) for T-sweep")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model}")
    mod, model = load_best_model(args.model, device)
    target_cols = mod.TARGET_COLS

    # Top species by typical abundance (FastChem/Sharp-Huebner style)
    # Use our naming: H2O1, C1O1=CO, C1H4=CH4, C1O2=CO2, H3N1=NH3
    key_species = ["H2", "H2O1", "N2", "C1O1", "C1H4", "O2", "C1O2", "H2S1", "H3N1"]
    plot_species = [s for s in key_species if s in target_cols]
    if not plot_species:
        plot_species = list(target_cols[:9])  # fallback to first 9

    # -------------------------------------------------------------------------
    # Plot 1: FastChem Fig 1 style — abundances vs T at fixed P
    # -------------------------------------------------------------------------
    print("Generating T-sweep at P=%.2f bar..." % args.p_fixed)
    df_1d = make_tp_grid(t_min=500, t_max=3000, n_t=80, p_fixed=args.p_fixed)
    y_pred = predict(mod, model, df_1d, device)
    T_vals = df_1d["T_K"].values

    n_plots = min(9, len(plot_species))
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for k, sp in enumerate(plot_species[:n_plots]):
        row, col = k // n_cols, k % n_cols
        ax = axes[row, col]
        idx = target_cols.index(sp)
        abun = y_pred[:, idx]
        ax.semilogy(T_vals, np.maximum(abun, 1e-30), "b-", linewidth=2)
        ax.set_xlabel("Temperature (K)", fontsize=10)
        ax.set_ylabel(r"Number density (cm$^{-3}$)", fontsize=10)
        ax.set_title(sp, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(T_vals.min(), T_vals.max())

    for k in range(n_plots, axes.size):
        row, col = k // n_cols, k % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Gas-phase abundances vs T at P = {args.p_fixed} bar (solar composition)\n"
        f"ML emulator: {args.model}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out1 = args.output_dir / "mixing_ratio_vs_T.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # -------------------------------------------------------------------------
    # Plot 2: 2D heatmap — one key species vs (T, P)
    # -------------------------------------------------------------------------
    sp_heatmap = "H2O1" if "H2O1" in target_cols else plot_species[0]
    print(f"Generating 2D heatmap for {sp_heatmap}...")
    df_2d, T_grid, P_grid = make_tp_grid_2d(
        t_min=800, t_max=3000, p_min=1e-3, p_max=1e2, n_t=50, n_p=50
    )
    y_2d = predict(mod, model, df_2d, device)
    idx_sp = target_cols.index(sp_heatmap)
    Z = y_2d[:, idx_sp].reshape(len(P_grid), len(T_grid))
    Z = np.maximum(Z, 1e-30)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(
        T_grid,
        P_grid,
        np.log10(Z),
        shading="auto",
        cmap="viridis",
    )
    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel("Pressure (bar)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(
        f"log₁₀(Number density) of {sp_heatmap} vs T and P\n"
        f"ML emulator: {args.model}",
        fontsize=13,
        fontweight="bold",
    )
    cbar = plt.colorbar(im, ax=ax, label=r"log₁₀(n / cm$^{-3}$)")
    plt.tight_layout()
    out2 = args.output_dir / "mixing_ratio_heatmap_TP.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
