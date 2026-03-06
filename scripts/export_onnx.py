#!/usr/bin/env python3
"""
export_onnx.py
==============

Exports the ML emulator to ONNX format and benchmarks ONNX Runtime
inference speed against PyTorch eager CPU and MPS GPU.

Usage:
    python scripts/export_onnx.py

Outputs:
    results/runs/runs_autoencoder_x4800_optimal_retrained/model.onnx
    plots/speed_benchmark_onnx.csv
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
ONNX_PATH = BASE_DIR / "results" / "runs" / "runs_autoencoder_x4800_optimal_retrained" / "model.onnx"


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


def export_to_onnx():
    """Export the model to ONNX format."""
    model = bm.load_model(device="cpu")
    model.eval()

    n_species = len(bm.TARGET_COLS)
    n_inputs = len(bm.INPUT_COLS)

    dummy_y0 = torch.zeros((1, n_species), dtype=torch.float32)
    dummy_dt = torch.ones((1, 1), dtype=torch.float32)
    dummy_g = torch.randn((1, n_inputs), dtype=torch.float32)

    print(f"Exporting to ONNX: {ONNX_PATH}")
    torch.onnx.export(
        model,
        (dummy_y0, dummy_dt, dummy_g),
        str(ONNX_PATH),
        input_names=["y0", "dt", "g"],
        output_names=["y_pred"],
        dynamic_axes={
            "y0": {0: "batch"},
            "dt": {0: "batch"},
            "g": {0: "batch"},
            "y_pred": {0: "batch"},
        },
        opset_version=17,
    )

    import onnx
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"ONNX model exported and validated ({size_mb:.1f} MB)")
    return ONNX_PATH


def bench_onnx(session, X_np: np.ndarray, n_warmup: int = 10) -> float:
    """Benchmark ONNX Runtime inference, return median ms/sample."""
    n = X_np.shape[0]
    n_species = len(bm.TARGET_COLS)
    y0 = np.zeros((n, n_species), dtype=np.float32)
    dt = np.ones((n, 1), dtype=np.float32)

    feeds = {"y0": y0, "dt": dt, "g": X_np}
    for _ in range(n_warmup):
        session.run(None, feeds)

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        session.run(None, feeds)
        t1 = time.perf_counter()
        times.append((t1 - t0) / n * 1000)

    return float(np.median(times))


def bench_pytorch(model, X: torch.Tensor, device: torch.device,
                  n_warmup: int = 10) -> float:
    n = X.shape[0]
    g = X.to(device)
    y0 = torch.zeros((n, len(bm.TARGET_COLS)), dtype=g.dtype, device=device)
    dt = torch.ones((n, 1), dtype=g.dtype, device=device)

    for _ in range(n_warmup):
        with torch.no_grad():
            model(y0, dt, g)
        if device.type == "mps":
            torch.mps.synchronize()

    times = []
    for _ in range(N_REPEATS):
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(y0, dt, g)
        if device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) / n * 1000)

    return float(np.median(times))


def main():
    import onnxruntime as ort

    print("=" * 90)
    print("ONNX EXPORT AND BENCHMARK")
    print("=" * 90)
    print(f"PyTorch {torch.__version__} | ONNX Runtime {ort.__version__}")
    print()

    # Export
    export_to_onnx()
    print()

    # Create ONNX session
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 8
    session = ort.InferenceSession(str(ONNX_PATH), sess_opts,
                                   providers=["CPUExecutionProvider"])
    print(f"ONNX Runtime session created (provider: {session.get_providers()[0]})")

    # PyTorch models
    model_cpu = bm.load_model(device="cpu")
    model_cpu.eval()

    has_mps = torch.backends.mps.is_available()
    model_mps = None
    if has_mps:
        model_mps = bm.load_model(device="mps")
        model_mps.eval()

    # Benchmark
    print()
    all_results = []
    for bs in BATCH_SIZES:
        df = make_conditions(bs)
        X = bm.normalize_inputs(df)
        X_np = X.numpy().astype(np.float32)

        cpu_ms = bench_pytorch(model_cpu, X, torch.device("cpu"), n_warmup=15)
        onnx_ms = bench_onnx(session, X_np, n_warmup=15)
        onnx_vs_cpu = cpu_ms / onnx_ms

        row = {
            "batch_size": bs,
            "pytorch_cpu_ms": cpu_ms,
            "onnx_cpu_ms": onnx_ms,
            "onnx_vs_pytorch": onnx_vs_cpu,
        }

        mps_ms = None
        if model_mps is not None:
            mps_ms = bench_pytorch(model_mps, X, torch.device("mps"), n_warmup=15)
            row["pytorch_mps_ms"] = mps_ms

        all_results.append(row)

        mps_str = f"  MPS={mps_ms:.6f}" if mps_ms is not None else ""
        print(f"  batch={bs:>7,d}: PyTorch CPU={cpu_ms:.6f}  "
              f"ONNX={onnx_ms:.6f}  "
              f"ONNX/PT={onnx_vs_cpu:.2f}x{mps_str}")

    # Summary
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()
    print(f"{'Batch':>8s}  {'PyTorch CPU':>14s}  {'ONNX Runtime':>14s}  {'ONNX speedup':>14s}", end="")
    if has_mps:
        print(f"  {'MPS GPU':>14s}", end="")
    print()
    print("-" * (58 + (16 if has_mps else 0)))

    for r in all_results:
        print(f"{r['batch_size']:>8,d}  {r['pytorch_cpu_ms']:>12.6f}ms  "
              f"{r['onnx_cpu_ms']:>12.6f}ms  {r['onnx_vs_pytorch']:>13.2f}x", end="")
        if has_mps and "pytorch_mps_ms" in r:
            print(f"  {r['pytorch_mps_ms']:>12.6f}ms", end="")
        print()

    print()
    best_onnx = max(all_results, key=lambda r: r["onnx_vs_pytorch"])
    print(f"Peak ONNX speedup over PyTorch CPU: {best_onnx['onnx_vs_pytorch']:.2f}x "
          f"(batch={best_onnx['batch_size']:,d})")

    # Accuracy check
    print()
    print("Accuracy verification (batch=1000)...")
    df_test = make_conditions(1000)
    X = bm.normalize_inputs(df_test)
    X_np = X.numpy().astype(np.float32)
    n_species = len(bm.TARGET_COLS)

    with torch.no_grad():
        y0 = torch.zeros((1000, n_species), dtype=torch.float32)
        dt = torch.ones((1000, 1), dtype=torch.float32)
        pt_out = model_cpu(y0, dt, X)[:, 0, :].numpy()

    onnx_out = session.run(None, {
        "y0": np.zeros((1000, n_species), dtype=np.float32),
        "dt": np.ones((1000, 1), dtype=np.float32),
        "g": X_np,
    })[0][:, 0, :]

    max_diff = np.max(np.abs(pt_out - onnx_out))
    mean_diff = np.mean(np.abs(pt_out - onnx_out))
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    if max_diff < 1e-5:
        print("  Accuracy: PASSED (numerically identical)")
    else:
        print(f"  Accuracy: WARNING (max diff = {max_diff:.2e})")

    # Save
    out_path = BASE_DIR / "plots" / "speed_benchmark_onnx.csv"
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print(f"ONNX model: {ONNX_PATH}")
    print("=" * 90)


if __name__ == "__main__":
    main()
