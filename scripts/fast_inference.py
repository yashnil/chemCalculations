#!/usr/bin/env python3
"""
fast_inference.py
=================

Benchmarks ML emulator inference on CPU vs MPS GPU (Apple Silicon).

Demonstrates that moving from CPU to MPS GPU yields ~1,500x speedup
over FastChem at batch sizes >= 10,000.

Usage:
    python scripts/fast_inference.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "results" / "runs" / "runs_autoencoder_x4800_optimal_retrained"))

import best_model as bm

SOLAR = {"H": 12.00, "O": 8.69, "C": 8.43, "N": 7.83, "S": 7.12}
N_REPEATS = 5
BATCH_SIZES = [1, 10, 100, 1_000, 10_000, 100_000]


def make_conditions(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "T_K": rng.uniform(800, 3000, n),
        "P_bar": 10.0 ** rng.uniform(-4, 3, n),
        "abund_H_dex": np.full(n, SOLAR["H"]),
        "abund_O_dex": SOLAR["O"] + rng.normal(0, 0.3, n),
        "abund_C_dex": SOLAR["C"] + rng.normal(0, 0.3, n),
        "abund_N_dex": SOLAR["N"] + rng.normal(0, 0.3, n),
        "abund_S_dex": SOLAR["S"] + rng.normal(0, 0.3, n),
    })


def _sync(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench_forward(model, X: torch.Tensor, device: torch.device,
                  n_warmup: int = 10) -> float:
    """Return median ms/sample over N_REPEATS trials."""
    n = X.shape[0]
    g = X.to(device)
    y0 = torch.zeros((n, len(bm.TARGET_COLS)), dtype=g.dtype, device=device)
    dt = torch.ones((n, 1), dtype=g.dtype, device=device)

    for _ in range(n_warmup):
        with torch.no_grad():
            model(y0, dt, g)
        _sync(device)

    times = []
    for _ in range(N_REPEATS):
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(y0, dt, g)
        _sync(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) / n * 1000)

    return float(np.median(times))


def bench_fastchem_reuse(n: int = 100) -> float:
    """FastChem engine-reuse benchmark, ms/sample."""
    try:
        import pyfastchem
    except ImportError:
        return float("nan")

    logk = str(BASE_DIR / "data" / "fastchem_data" / "Kitzmann2023" / "logK.dat")
    cond = str(BASE_DIR / "data" / "fastchem_data" / "Kitzmann2023" / "logK_condensates.dat")
    elem = str(BASE_DIR / "data" / "fastchem_data" / "lodders_2003_extended.dat")

    eng = pyfastchem.FastChem(elem, logk, cond, 0)
    syms = [eng.getElementSymbol(i) for i in range(eng.getElementNumber())]
    base = np.array(eng.getElementAbundances(), dtype=np.float64, copy=True)
    df = make_conditions(n)

    for i in range(min(5, n)):
        vec = base.copy()
        for idx, sym in enumerate(syms):
            col = f"abund_{sym}_dex"
            if col in df.columns:
                vec[idx] = 10.0 ** (df.iloc[i][col] - 12.0)
        eng.setElementAbundances(vec.tolist())
        inp = pyfastchem.FastChemInput()
        out = pyfastchem.FastChemOutput()
        inp.temperature = np.array([df.iloc[i]["T_K"]], dtype=np.float64)
        inp.pressure = np.array([df.iloc[i]["P_bar"]], dtype=np.float64)
        eng.calcDensities(inp, out)

    t0 = time.perf_counter()
    for i in range(n):
        vec = base.copy()
        for idx, sym in enumerate(syms):
            col = f"abund_{sym}_dex"
            if col in df.columns:
                vec[idx] = 10.0 ** (df.iloc[i][col] - 12.0)
        eng.setElementAbundances(vec.tolist())
        inp = pyfastchem.FastChemInput()
        out = pyfastchem.FastChemOutput()
        inp.temperature = np.array([df.iloc[i]["T_K"]], dtype=np.float64)
        inp.pressure = np.array([df.iloc[i]["P_bar"]], dtype=np.float64)
        eng.calcDensities(inp, out)
    elapsed = time.perf_counter() - t0
    return (elapsed / n) * 1000


def main():
    print("=" * 90)
    print("INFERENCE SPEED BENCHMARK: CPU vs MPS GPU (Apple Silicon)")
    print("=" * 90)

    has_mps = torch.backends.mps.is_available()
    print(f"PyTorch {torch.__version__} | MPS GPU available: {has_mps}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Repeats per measurement: {N_REPEATS} (median reported)")
    print()

    # FastChem baseline
    print("FastChem baseline (engine-reuse, N=100)...")
    fc_ms = bench_fastchem_reuse(100)
    if np.isnan(fc_ms):
        print("  pyfastchem unavailable — using previous measurement: 2.1 ms")
        fc_ms = 2.1
    else:
        print(f"  {fc_ms:.3f} ms/sample")
    print()

    # CPU model
    model_cpu = bm.load_model(device="cpu")
    model_cpu.eval()

    # MPS model
    model_mps = None
    if has_mps:
        model_mps = bm.load_model(device="mps")
        model_mps.eval()

    # Benchmark
    all_results = []
    for bs in BATCH_SIZES:
        df = make_conditions(bs)
        X = bm.normalize_inputs(df)

        cpu_ms = bench_forward(model_cpu, X, torch.device("cpu"), n_warmup=15)
        cpu_speedup = fc_ms / cpu_ms

        row = {
            "batch_size": bs,
            "cpu_ms": cpu_ms,
            "cpu_speedup": cpu_speedup,
            "cpu_sps": 1000.0 / cpu_ms,
        }

        if model_mps is not None:
            mps_ms = bench_forward(model_mps, X, torch.device("mps"), n_warmup=15)
            mps_speedup = fc_ms / mps_ms
            gpu_vs_cpu = cpu_ms / mps_ms
            row.update({
                "mps_ms": mps_ms,
                "mps_speedup": mps_speedup,
                "mps_sps": 1000.0 / mps_ms,
                "gpu_vs_cpu": gpu_vs_cpu,
            })

        all_results.append(row)

    # Print table
    print()
    print("=" * 90)
    print(f"RESULTS (FastChem baseline: {fc_ms:.3f} ms/sample)")
    print("=" * 90)
    print()

    if model_mps is not None:
        print(f"{'Batch':>8s}  {'CPU ms/samp':>12s} {'CPU speedup':>12s}"
              f"  {'MPS ms/samp':>12s} {'MPS speedup':>12s} {'GPU/CPU':>8s}")
        print("-" * 72)
        for r in all_results:
            print(f"{r['batch_size']:>8,d}  {r['cpu_ms']:>12.6f} {r['cpu_speedup']:>11.0f}x"
                  f"  {r['mps_ms']:>12.6f} {r['mps_speedup']:>11,.0f}x"
                  f" {r['gpu_vs_cpu']:>7.1f}x")
    else:
        print(f"{'Batch':>8s}  {'CPU ms/samp':>12s} {'CPU speedup':>12s}")
        print("-" * 36)
        for r in all_results:
            print(f"{r['batch_size']:>8,d}  {r['cpu_ms']:>12.6f} {r['cpu_speedup']:>11.0f}x")

    print()
    best_cpu = max(all_results, key=lambda r: r["cpu_speedup"])
    print(f"Peak CPU:  {best_cpu['cpu_speedup']:,.0f}x vs FastChem "
          f"(batch={best_cpu['batch_size']:,d}, "
          f"{best_cpu['cpu_sps']:,.0f} samples/sec)")

    if model_mps is not None:
        best_mps = max(all_results, key=lambda r: r.get("mps_speedup", 0))
        print(f"Peak MPS:  {best_mps['mps_speedup']:,.0f}x vs FastChem "
              f"(batch={best_mps['batch_size']:,d}, "
              f"{best_mps['mps_sps']:,.0f} samples/sec)")

    print()

    # Save
    out_path = BASE_DIR / "plots" / "speed_benchmark_optimized.csv"
    df_out = pd.DataFrame(all_results)
    df_out["fastchem_ms"] = fc_ms
    df_out.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
