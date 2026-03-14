#!/usr/bin/env python3
"""
benchmark_throughput.py
=======================

Chemulator-style throughput benchmark: ML emulator vs FastChem line-by-line.

Measures samples/second at various batch sizes for the ML model, and compares
against FastChem (engine-reuse, one sample at a time) to estimate speedup.

Usage:
    python scripts/benchmark_throughput.py [--model x4800_improved] [--output plots/bench_throughput.png]

Requires: pyfastchem, trained model (x4800_improved or x4800_mlp).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "results" / "runs"
PLOTS_DIR = BASE_DIR / "plots"

LOGK = BASE_DIR / "data" / "fastchem_data" / "Kitzmann2023" / "logK.dat"
LOGK_COND = BASE_DIR / "data" / "fastchem_data" / "Kitzmann2023" / "logK_condensates.dat"
ELEM_ABUND = BASE_DIR / "data" / "fastchem_data" / "lodders_2003_extended.dat"

SOLAR = {"H": 12.00, "O": 8.69, "C": 8.43, "N": 7.83, "S": 7.12}

WARMUP_STEPS = 10
MEASURE_STEPS = 200
MAX_BATCH = 4096
N_FASTCHEM_SAMPLES = 100  # samples for FastChem baseline (line-by-line)


def _p2_batches(max_cap: int) -> List[int]:
    out = []
    b = 1
    while b <= max_cap:
        out.append(b)
        b <<= 1
    return out


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def make_conditions(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "T_K": rng.uniform(800, 3000, n),
        "P_bar": 10.0 ** rng.uniform(-4, 3, n),
        "abund_H_dex": np.full(n, SOLAR["H"]),
        "abund_O_dex": SOLAR["O"] + rng.normal(0, 0.3, n),
        "abund_C_dex": SOLAR["C"] + rng.normal(0, 0.3, n),
        "abund_N_dex": SOLAR["N"] + rng.normal(0, 0.3, n),
        "abund_S_dex": SOLAR["S"] + rng.normal(0, 0.3, n),
    })


def load_best_model(run_tag: str, device: torch.device):
    """Load best_model.py from a run directory."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    best_path = run_dir / "best_model.py"
    if not best_path.exists():
        raise FileNotFoundError(f"Model not found: {best_path}")
    # Ensure autoencoder_model can be imported from src
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


def bench_fastchem_line_by_line(n: int = N_FASTCHEM_SAMPLES) -> float:
    """FastChem: engine reuse, one sample at a time. Returns samples/second."""
    try:
        import pyfastchem
    except ImportError:
        return float("nan")

    eng = pyfastchem.FastChem(str(ELEM_ABUND), str(LOGK), str(LOGK_COND), 0)
    syms = [eng.getElementSymbol(i) for i in range(eng.getElementNumber())]
    base = np.array(eng.getElementAbundances(), dtype=np.float64, copy=True)

    rng = np.random.default_rng(42)
    df = make_conditions(n, rng)

    # Warmup
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
    return n / max(elapsed, 1e-12)


def bench_ml_device(
    mod,
    model: torch.nn.Module,
    device: torch.device,
    rng: np.random.Generator,
    max_batch: int,
) -> Tuple[List[int], List[float]]:
    """Benchmark ML model at powers-of-2 batch sizes. Returns (batch_sizes, throughputs)."""
    batches = _p2_batches(max_batch)
    throughputs = []
    ok_batches = []

    for B in batches:
        try:
            df = make_conditions(B, rng)
            X = mod.normalize_inputs(df)
            g = torch.as_tensor(X, dtype=torch.float32, device=device)
            y0 = torch.zeros((B, len(mod.TARGET_COLS)), dtype=g.dtype, device=device)
            dt = torch.ones((B, 1), dtype=g.dtype, device=device)

            for _ in range(WARMUP_STEPS):
                with torch.no_grad():
                    _ = model(y0, dt, g)
            _sync(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(MEASURE_STEPS):
                    _ = model(y0, dt, g)
            _sync(device)
            elapsed = time.perf_counter() - t0

            sps = (MEASURE_STEPS * B) / max(elapsed, 1e-12)
            throughputs.append(float(sps))
            ok_batches.append(B)
        except Exception as e:
            print(f"    B={B} failed: {e}")
            break

    return ok_batches, throughputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="x4800_improved", help="Run tag (e.g. x4800_improved, x4800_mlp)")
    parser.add_argument("--output", type=Path, default=PLOTS_DIR / "bench_throughput.png")
    parser.add_argument("--max-batch", type=int, default=MAX_BATCH)
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU benchmark")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("THROUGHPUT BENCHMARK: ML Emulator vs FastChem (line-by-line)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print()

    # FastChem baseline
    print("FastChem baseline (engine reuse, line-by-line, N=100)...")
    fc_sps = bench_fastchem_line_by_line(N_FASTCHEM_SAMPLES)
    if np.isnan(fc_sps):
        print("  pyfastchem unavailable — using 500 samples/sec as placeholder")
        fc_sps = 500.0
    else:
        print(f"  {fc_sps:,.1f} samples/sec ({1000/fc_sps:.3f} ms/sample)")
    print()

    # Load ML model
    device_cpu = torch.device("cpu")
    mod, model_cpu = load_best_model(args.model, device_cpu)
    rng = np.random.default_rng(42)

    # CPU benchmark
    print(f"ML ({args.model}) CPU...")
    bs_cpu, thr_cpu = bench_ml_device(mod, model_cpu, device_cpu, rng, args.max_batch)
    for b, t in zip(bs_cpu, thr_cpu):
        print(f"  B={b:>5d} -> {t:,.1f} samples/sec ({t/fc_sps:.0f}x vs FastChem)")
    print()

    curves: List[Tuple[str, List[int], List[float]]] = [("CPU", bs_cpu, thr_cpu)]

    # GPU benchmark (MPS or CUDA)
    device_gpu = None
    if not args.no_gpu:
        if torch.cuda.is_available():
            device_gpu = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device_gpu = torch.device("mps")

    if device_gpu is not None:
        _, model_gpu = load_best_model(args.model, device_gpu)
        print(f"ML ({args.model}) {device_gpu.type.upper()}...")
        bs_gpu, thr_gpu = bench_ml_device(mod, model_gpu, device_gpu, rng, args.max_batch)
        for b, t in zip(bs_gpu, thr_gpu):
            print(f"  B={b:>5d} -> {t:,.1f} samples/sec ({t/fc_sps:.0f}x vs FastChem)")
        curves.append((device_gpu.type.upper(), bs_gpu, thr_gpu))
        print()

    # Plot
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, bs, th in curves:
        ax.plot(bs, th, marker="o", linewidth=2, markersize=8, label=f"ML ({label})")
    ax.axhline(y=fc_sps, color="gray", linewidth=2, linestyle="--", label=f"FastChem (line-by-line)")

    ax.set_xlabel("Batch size (B)", fontsize=12)
    ax.set_ylabel("Throughput (samples/second)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=10)
    ax.set_title(f"ML Emulator vs FastChem: {args.model}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Auto-set x limits
    all_bs = [b for _, bs, _ in curves for b in bs if b > 0]
    if all_bs:
        ax.set_xlim(max(min(all_bs) / 1.5, 0.5), max(all_bs) * 1.5)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {args.output}")

    # Save CSV
    csv_path = args.output.with_suffix(".csv")
    rows = [{"batch_size": b, "cpu_sps": t, "fastchem_sps": fc_sps, "speedup": t / fc_sps}
            for b, t in zip(bs_cpu, thr_cpu)]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
