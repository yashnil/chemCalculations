#!/usr/bin/env python3
"""
Benchmark FastChem speed to verify the 7ms/eval claim.

Measures actual FastChem execution time for a representative set of conditions.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyfastchem

# Configuration
N_WARMUP = 5
N_TIMING_RUNS = 100
CHUNKSIZE = 1  # Single evaluations for accurate per-sample timing


def resolve_path(value: str | None, env_var: str) -> Path | None:
    if value:
        return Path(value).expanduser().resolve()
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return None


def infer_element_path(logk_path: Path) -> Path | None:
    candidates = [
        logk_path.parent.parent / "element_abundances" / "asplund_2009.dat",
        logk_path.parent.parent / "element_abundances" / "solar.abundances",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main():
    print("=" * 80)
    print("FASTCHEM SPEED BENCHMARK")
    print("=" * 80)
    print()
    
    # Resolve FastChem data paths
    logk_path = resolve_path(None, "FASTCHEM_LOGK")
    cond_path = resolve_path(None, "FASTCHEM_COND")
    elem_path = resolve_path(None, "FASTCHEM_ELEM")
    
    if not logk_path or not logk_path.exists():
        print("❌ Error: FASTCHEM_LOGK not set or file not found")
        print("   Set: export FASTCHEM_LOGK=/path/to/FastChem/input/logK/logK.dat")
        return
    
    if not cond_path or not cond_path.exists():
        print("❌ Error: FASTCHEM_COND not set or file not found")
        print("   Set: export FASTCHEM_COND=/path/to/FastChem/input/logK/logK_condensates.dat")
        return
    
    if not elem_path:
        elem_path = infer_element_path(logk_path)
    
    if not elem_path or not elem_path.exists():
        print("❌ Error: Element abundance file not found")
        print("   Set: export FASTCHEM_ELEM=/path/to/asplund_2009.dat")
        return
    
    print(f"✓ LogK file: {logk_path}")
    print(f"✓ Condensates file: {cond_path}")
    print(f"✓ Element abundances: {elem_path}")
    print()
    
    # Initialize FastChem
    print("Initializing FastChem...")
    try:
        fastchem = pyfastchem.FastChem(
            str(logk_path),
            str(cond_path),
            str(elem_path),
        )
        print("✓ FastChem initialized")
    except Exception as e:
        print(f"❌ Error initializing FastChem: {e}")
        return
    
    # Load test conditions from dataset
    csv_path = Path("data/datasets/all_gas_fastchem_x160.csv")
    if not csv_path.exists():
        print(f"❌ Error: Dataset not found at {csv_path}")
        return
    
    print(f"\nLoading test conditions from {csv_path.name}...")
    df = pd.read_csv(csv_path)
    
    # Use a representative sample (first N_TIMING_RUNS rows)
    test_df = df.head(N_TIMING_RUNS).copy()
    print(f"✓ Using {len(test_df)} test conditions")
    print(f"  T range: {test_df['T_K'].min():.0f} - {test_df['T_K'].max():.0f} K")
    print(f"  P range: {test_df['P_bar'].min():.1e} - {test_df['P_bar'].max():.1e} bar")
    print()
    
    # Prepare conditions
    temperatures = test_df['T_K'].values.astype(np.float64)
    pressures = test_df['P_bar'].values.astype(np.float64)
    
    # Element abundances (solar composition)
    abund_H = 10**(test_df.get('abund_H_dex', 12.0).values - 12.0)
    abund_C = 10**(test_df.get('abund_C_dex', 8.43).values - 12.0)
    abund_N = 10**(test_df.get('abund_N_dex', 7.83).values - 12.0)
    abund_O = 10**(test_df.get('abund_O_dex', 8.69).values - 12.0)
    abund_S = 10**(test_df.get('abund_S_dex', 7.12).values - 12.0)
    
    # Warmup
    print(f"Warming up ({N_WARMUP} evaluations)...")
    for i in range(N_WARMUP):
        try:
            fastchem.calcDensities(
                temperatures[i:i+1],
                pressures[i:i+1],
                abund_H[i:i+1],
                abund_C[i:i+1],
                abund_N[i:i+1],
                abund_O[i:i+1],
                abund_S[i:i+1],
            )
        except Exception as e:
            print(f"⚠️  Warmup {i+1} failed: {e}")
    
    print("✓ Warmup complete")
    print()
    
    # Timing runs
    print(f"Running timing benchmark ({N_TIMING_RUNS} evaluations)...")
    times = []
    
    for i in range(N_TIMING_RUNS):
        t0 = time.perf_counter()
        try:
            result = fastchem.calcDensities(
                temperatures[i:i+1],
                pressures[i:i+1],
                abund_H[i:i+1],
                abund_C[i:i+1],
                abund_N[i:i+1],
                abund_O[i:i+1],
                abund_S[i:i+1],
            )
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            times.append(elapsed_ms)
        except Exception as e:
            print(f"⚠️  Evaluation {i+1} failed: {e}")
    
    if not times:
        print("❌ Error: No successful evaluations")
        return
    
    times = np.array(times)
    
    # Statistics
    mean_ms = np.mean(times)
    median_ms = np.median(times)
    std_ms = np.std(times)
    min_ms = np.min(times)
    max_ms = np.max(times)
    p25_ms = np.percentile(times, 25)
    p75_ms = np.percentile(times, 75)
    
    print()
    print("=" * 80)
    print("FASTCHEM SPEED BENCHMARK RESULTS")
    print("=" * 80)
    print()
    print(f"Evaluations: {len(times)}")
    print()
    print("Per-sample timing:")
    print(f"  Mean:   {mean_ms:.3f} ms")
    print(f"  Median: {median_ms:.3f} ms")
    print(f"  Std:    {std_ms:.3f} ms")
    print(f"  Min:    {min_ms:.3f} ms")
    print(f"  Max:    {max_ms:.3f} ms")
    print(f"  25th %: {p25_ms:.3f} ms")
    print(f"  75th %: {p75_ms:.3f} ms")
    print()
    print(f"Throughput: {1000.0/mean_ms:.1f} samples/sec")
    print()
    
    # Compare to assumed 7ms
    assumed_ms = 7.0
    diff_pct = ((mean_ms - assumed_ms) / assumed_ms) * 100
    
    print("=" * 80)
    print("COMPARISON TO ASSUMED VALUE")
    print("=" * 80)
    print(f"Assumed:  {assumed_ms:.3f} ms/sample")
    print(f"Measured: {mean_ms:.3f} ms/sample")
    print(f"Difference: {diff_pct:+.1f}%")
    print()
    
    if abs(diff_pct) < 20:
        print("✅ Measured value is close to assumed 7ms (within 20%)")
    else:
        print(f"⚠️  Measured value differs significantly from assumed 7ms ({diff_pct:+.1f}%)")
        print("   Consider updating FASTCHEM_MS_PER_SAMPLE in inference_speed_test.py")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

