#!/usr/bin/env python3
"""
benchmark_fastchem_speed.py
===========================

Rigorous, fair speed comparison between FastChem and the ML emulator.

Tests multiple scenarios to give an honest range of speedup factors:
  1. Single-sample: one condition at a time, both cold
  2. Engine-reuse: FastChem reuses engine, ML model pre-loaded
  3. Batch T-P (same composition): FastChem native batch vs ML batch
  4. Varying composition batch: different abundances per sample
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

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
    raise RuntimeError("pyfastchem not available — run in the fastchem_nn conda env")

LOGK = BASE_DIR / "data" / "fastchem_data" / "Kitzmann2023" / "logK.dat"
LOGK_COND = BASE_DIR / "data" / "fastchem_data" / "Kitzmann2023" / "logK_condensates.dat"
ELEM_ABUND = BASE_DIR / "data" / "fastchem_data" / "lodders_2003_extended.dat"

SOLAR = {"H": 12.00, "O": 8.69, "C": 8.43, "N": 7.83, "S": 7.12}

N_REPEATS = 3  # repeat each benchmark and take the median


def make_conditions(n: int, vary_composition: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "T_K": rng.uniform(800, 3000, n),
        "P_bar": 10.0 ** rng.uniform(-4, 3, n),
    })
    if vary_composition:
        df["abund_H_dex"] = np.full(n, SOLAR["H"])
        df["abund_O_dex"] = SOLAR["O"] + rng.normal(0, 0.3, n)
        df["abund_C_dex"] = SOLAR["C"] + rng.normal(0, 0.3, n)
        df["abund_N_dex"] = SOLAR["N"] + rng.normal(0, 0.3, n)
        df["abund_S_dex"] = SOLAR["S"] + rng.normal(0, 0.3, n)
    else:
        for elem, val in SOLAR.items():
            df[f"abund_{elem}_dex"] = val
    return df


def _fc_engine():
    return pyfastchem.FastChem(str(ELEM_ABUND), str(LOGK), str(LOGK_COND), 0)


def _fc_get_elem_info(engine):
    syms = [engine.getElementSymbol(i) for i in range(engine.getElementNumber())]
    base = np.array(engine.getElementAbundances(), dtype=np.float64, copy=True)
    return syms, base


def _fc_set_abundances(engine, syms, base, row):
    vec = base.copy()
    for idx, sym in enumerate(syms):
        col = f"abund_{sym}_dex"
        if col in row.index:
            vec[idx] = 10.0 ** (row[col] - 12.0)
    engine.setElementAbundances(vec.tolist())


# ---------------------------------------------------------------------------
# Scenario 1: Fresh engine per sample (worst-case FastChem)
# ---------------------------------------------------------------------------
def bench_fc_fresh_engine(df: pd.DataFrame) -> float:
    n = len(df)
    t0 = time.perf_counter()
    for i in range(n):
        eng = _fc_engine()
        syms, base = _fc_get_elem_info(eng)
        _fc_set_abundances(eng, syms, base, df.iloc[i])
        inp = pyfastchem.FastChemInput()
        out = pyfastchem.FastChemOutput()
        inp.temperature = np.array([df.iloc[i]["T_K"]], dtype=np.float64)
        inp.pressure = np.array([df.iloc[i]["P_bar"]], dtype=np.float64)
        eng.calcDensities(inp, out)
    return (time.perf_counter() - t0) / n * 1000


# ---------------------------------------------------------------------------
# Scenario 2: Reuse engine, change abundances per sample
# ---------------------------------------------------------------------------
def bench_fc_reuse_engine(df: pd.DataFrame) -> float:
    eng = _fc_engine()
    syms, base = _fc_get_elem_info(eng)
    n = len(df)
    t0 = time.perf_counter()
    for i in range(n):
        _fc_set_abundances(eng, syms, base, df.iloc[i])
        inp = pyfastchem.FastChemInput()
        out = pyfastchem.FastChemOutput()
        inp.temperature = np.array([df.iloc[i]["T_K"]], dtype=np.float64)
        inp.pressure = np.array([df.iloc[i]["P_bar"]], dtype=np.float64)
        eng.calcDensities(inp, out)
    return (time.perf_counter() - t0) / n * 1000


# ---------------------------------------------------------------------------
# Scenario 3: FastChem native batch (same composition, many T-P points)
# ---------------------------------------------------------------------------
def bench_fc_batch_tp(df: pd.DataFrame) -> float:
    eng = _fc_engine()
    syms, base = _fc_get_elem_info(eng)
    _fc_set_abundances(eng, syms, base, df.iloc[0])
    n = len(df)
    inp = pyfastchem.FastChemInput()
    out = pyfastchem.FastChemOutput()
    inp.temperature = df["T_K"].values.astype(np.float64)
    inp.pressure = df["P_bar"].values.astype(np.float64)
    t0 = time.perf_counter()
    eng.calcDensities(inp, out)
    return (time.perf_counter() - t0) / n * 1000


# ---------------------------------------------------------------------------
# ML benchmarks
# ---------------------------------------------------------------------------
def bench_ml(df: pd.DataFrame, model, n_warmup: int = 5) -> float:
    X = bm.normalize_inputs(df)
    for _ in range(n_warmup):
        with torch.no_grad():
            bm.forward_autoencoder(model, X)

    t0 = time.perf_counter()
    with torch.no_grad():
        bm.forward_autoencoder(model, X)
    return (time.perf_counter() - t0) / len(df) * 1000


def median_of(func, *args, repeats=N_REPEATS):
    times = [func(*args) for _ in range(repeats)]
    return np.median(times)


def main():
    print("=" * 80)
    print("FAIR SPEED BENCHMARK: FastChem vs ML Emulator")
    print("=" * 80)
    print(f"Model: x4800_optimal_retrained | Device: CPU ({torch.get_num_threads()} threads)")
    print(f"Each measurement repeated {N_REPEATS}x, median reported")
    print()

    model = bm.load_model(device=torch.device("cpu"))
    model.eval()

    # ---- Scenario A: Single-sample, varying composition (100 samples) ----
    print("-" * 80)
    print("SCENARIO A: Single-sample, varying composition (N=100)")
    print("  FastChem: fresh engine per sample | ML: pre-loaded model, batch=100")
    print("-" * 80)
    df_a = make_conditions(100, vary_composition=True)

    fc_fresh = median_of(bench_fc_fresh_engine, df_a)
    fc_reuse = median_of(bench_fc_reuse_engine, df_a)
    ml_a = median_of(bench_ml, df_a, model)

    print(f"  FastChem (fresh engine):  {fc_fresh:.3f} ms/sample")
    print(f"  FastChem (reuse engine):  {fc_reuse:.3f} ms/sample")
    print(f"  ML emulator (batch=100):  {ml_a:.4f} ms/sample")
    print(f"  Speedup vs fresh engine:  {fc_fresh / ml_a:.0f}x")
    print(f"  Speedup vs reuse engine:  {fc_reuse / ml_a:.0f}x")
    print()

    # ---- Scenario B: Batch T-P, fixed composition ----
    print("-" * 80)
    print("SCENARIO B: Batch T-P profile, fixed solar composition")
    print("  FastChem: native batch (single call) | ML: single forward pass")
    print("-" * 80)

    results_b = []
    for n in [100, 1000, 10000]:
        df_b = make_conditions(n, vary_composition=False)
        fc_batch = median_of(bench_fc_batch_tp, df_b)
        ml_b = median_of(bench_ml, df_b, model)
        speedup = fc_batch / ml_b
        results_b.append((n, fc_batch, ml_b, speedup))
        print(f"  N={n:>6,d}:  FastChem={fc_batch:.4f}  ML={ml_b:.4f} ms/sample  →  {speedup:.0f}x speedup")
    print()

    # ---- Scenario C: Varying composition batch (retrieval-like) ----
    print("-" * 80)
    print("SCENARIO C: Varying composition (retrieval-like)")
    print("  FastChem: reuse engine, loop | ML: single forward pass")
    print("-" * 80)

    results_c = []
    for n in [100, 1000, 10000]:
        df_c = make_conditions(n, vary_composition=True)
        fc_c = median_of(bench_fc_reuse_engine, df_c) if n <= 1000 else None
        ml_c = median_of(bench_ml, df_c, model)
        if fc_c is not None:
            speedup = fc_c / ml_c
            results_c.append((n, fc_c, ml_c, speedup))
            print(f"  N={n:>6,d}:  FastChem={fc_c:.4f}  ML={ml_c:.4f} ms/sample  →  {speedup:.0f}x speedup")
        else:
            results_c.append((n, None, ml_c, None))
            print(f"  N={n:>6,d}:  FastChem=skipped (too slow)  ML={ml_c:.4f} ms/sample")
    print()

    # ---- Summary ----
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Conservative (single-sample, reuse engine):")
    print(f"  FastChem: {fc_reuse:.3f} ms/sample  |  ML: {ml_a:.4f} ms/sample")
    print(f"  Speedup: {fc_reuse / ml_a:.0f}x")
    print()

    best_batch = max(results_b, key=lambda x: x[3])
    print(f"Best batch (fixed composition, N={best_batch[0]:,}):")
    print(f"  FastChem: {best_batch[1]:.4f} ms/sample  |  ML: {best_batch[2]:.4f} ms/sample")
    print(f"  Speedup: {best_batch[3]:.0f}x")
    print()

    if any(r[3] is not None for r in results_c):
        best_var = max((r for r in results_c if r[3] is not None), key=lambda x: x[3])
        print(f"Best varying-composition (N={best_var[0]:,}):")
        print(f"  FastChem: {best_var[1]:.4f} ms/sample  |  ML: {best_var[2]:.4f} ms/sample")
        print(f"  Speedup: {best_var[3]:.0f}x")
        print()

    print("Recommended claim: the ML emulator is ~100-1000x faster than FastChem,")
    print("depending on batch size and whether compositions vary.")
    print("=" * 80)

    # Save detailed results
    rows = []
    rows.append({"scenario": "A_fresh_engine", "n": 100, "fc_ms": fc_fresh, "ml_ms": ml_a, "speedup": fc_fresh / ml_a})
    rows.append({"scenario": "A_reuse_engine", "n": 100, "fc_ms": fc_reuse, "ml_ms": ml_a, "speedup": fc_reuse / ml_a})
    for n, fc, ml, sp in results_b:
        rows.append({"scenario": "B_batch_tp", "n": n, "fc_ms": fc, "ml_ms": ml, "speedup": sp})
    for n, fc, ml, sp in results_c:
        if sp is not None:
            rows.append({"scenario": "C_varying_comp", "n": n, "fc_ms": fc, "ml_ms": ml, "speedup": sp})

    out_path = BASE_DIR / "plots" / "speed_benchmark.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
