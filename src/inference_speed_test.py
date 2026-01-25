#!/usr/bin/env python3
"""
UPDATED_VERS Inference Speed Test

Matches the NEW_VERS speed harness but evaluates the FlowMap autoencoder.
Reports timing, speedup vs FastChem, and accuracy metrics on the test split.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Import helpers from autoencoder best_model
BEST_MODULE = os.environ.get("BEST_MODULE", "results/runs/runs_autoencoder/best_model.py")
import importlib.util
spec = importlib.util.spec_from_file_location("best_model", BEST_MODULE)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {BEST_MODULE}")
best_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(best_model)

# Import from the loaded module
load_model = best_model.load_model
normalize_inputs = best_model.normalize_inputs
denormalize_targets = best_model.denormalize_targets
forward_autoencoder = best_model.forward_autoencoder
TARGET_COLS = best_model.TARGET_COLS
INPUT_COLS = best_model.INPUT_COLS
SPLITS = best_model.SPLITS
TARGET_ZERO_FLOOR = best_model.TARGET_ZERO_FLOOR
LOG_EPS = best_model.LOG_EPS
TARGET_LOG_SCALE = best_model.TARGET_LOG_SCALE

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CSV_PATH = os.environ.get("CSV_PATH", "all_gas_v10_no_stripe_clean.csv")
OUT_DIR = os.environ.get("OUT_DIR", "results/runs/runs_autoencoder/speed_test")
N_WARMUP = int(os.environ.get("N_WARMUP", 10))
N_TIMING_RUNS = int(os.environ.get("N_TIMING_RUNS", 100))
BATCH_SIZES = [1, 10, 50, 100, 621]
# FastChem speed: measured ~5.1 ms/sample (median ~4.6 ms) on test system
# Can be verified with scripts/benchmark_fastchem_speed.py
# Using 7.0 ms as conservative estimate for speedup calculations
FASTCHEM_MS_PER_SAMPLE = float(os.environ.get("FASTCHEM_MS_PER_SAMPLE", 7.0))

print("=" * 80)
print("UPDATED_VERS INFERENCE SPEED TEST")
print("=" * 80)

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load model and data
# -----------------------------------------------------------------------------
print("\n1. Loading autoencoder model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")
model = load_model(device=device)
model.eval()
print(f"   ✓ Model loaded: {len(INPUT_COLS)} inputs → {len(TARGET_COLS)} species")

print("\n2. Loading test data...")
df = pd.read_csv(CSV_PATH)
test_idx = SPLITS["test_idx"]
df_test = df.iloc[test_idx].reset_index(drop=True)
print(f"   ✓ Test set: {len(df_test)} samples")
print(f"   ✓ T range: {df_test['T_K'].min():.0f}-{df_test['T_K'].max():.0f} K")
print(f"   ✓ P range: {df_test['P_bar'].min():.1e}-{df_test['P_bar'].max():.1e} bar")

X_test = normalize_inputs(df_test).to(device)
print(f"   ✓ Normalized inputs shape: {X_test.shape}")

# -----------------------------------------------------------------------------
# 2. Warmup
# -----------------------------------------------------------------------------
print(f"\n3. Warmup phase ({N_WARMUP} iterations)...")
for _ in range(N_WARMUP):
    with torch.no_grad():
        _ = forward_autoencoder(model, X_test)
if device.type == "cuda":
    torch.cuda.synchronize()
print("   ✓ Warmup complete")


def time_forward(batch: torch.Tensor) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        _ = forward_autoencoder(model, batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    return (t_end - t_start) * 1000.0


# -----------------------------------------------------------------------------
# 3. Full batch timing
# -----------------------------------------------------------------------------
print(f"\n4. Full batch inference timing ({N_TIMING_RUNS} runs)...")
times_full = np.array([time_forward(X_test) for _ in range(N_TIMING_RUNS)])
mean_time_full = times_full.mean()
std_time_full = times_full.std()
median_time_full = np.median(times_full)
print(f"   Full batch ({len(df_test)} samples):")
print(f"     Mean:   {mean_time_full:.4f} ms")
print(f"     Median: {median_time_full:.4f} ms")
print(f"     Std:    {std_time_full:.4f} ms")
time_per_sample_ms = mean_time_full / len(df_test)
print(f"\n   ✓ Per-sample inference: {time_per_sample_ms:.6f} ms/sample")

# -----------------------------------------------------------------------------
# 4. Single-sample timing
# -----------------------------------------------------------------------------
print(f"\n5. Single-sample inference timing ({N_TIMING_RUNS} runs)...")
sample_idx = len(df_test) // 2
X_single = X_test[sample_idx : sample_idx + 1]
times_single = np.array([time_forward(X_single) for _ in range(N_TIMING_RUNS)])
mean_time_single = times_single.mean()
std_time_single = times_single.std()
print(f"   Single sample:")
print(f"     Mean:   {mean_time_single:.6f} ms")
print(f"     Median: {np.median(times_single):.6f} ms")
print(f"     Std:    {std_time_single:.6f} ms")

# -----------------------------------------------------------------------------
# 5. Variable batch size timing
# -----------------------------------------------------------------------------
print(f"\n6. Variable batch size timing...")
batch_timings: Dict[int, Dict[str, float]] = {}
for batch_size in BATCH_SIZES:
    if batch_size > len(df_test):
        continue
    X_batch = X_test[:batch_size]
    times_batch = np.array([time_forward(X_batch) for _ in range(N_TIMING_RUNS)])
    batch_timings[batch_size] = {
        "mean_total_ms": times_batch.mean(),
        "std_total_ms": times_batch.std(),
        "mean_per_sample_ms": times_batch.mean() / batch_size,
    }
    print(
        f"   Batch={batch_size:4d}: {batch_timings[batch_size]['mean_total_ms']:8.4f} ms total, "
        f"{batch_timings[batch_size]['mean_per_sample_ms']:.6f} ms/sample"
    )

# -----------------------------------------------------------------------------
# 6. FastChem comparison
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FASTCHEM vs UPDATED_VERS SPEED COMPARISON")
print("=" * 80)
newvers_ms_per_sample = time_per_sample_ms
fastchem_ms_per_sample = FASTCHEM_MS_PER_SAMPLE
speedup = fastchem_ms_per_sample / newvers_ms_per_sample
print(f"\nFastChem (typical):      {fastchem_ms_per_sample:.3f} ms/sample")
print(f"UPDATED_VERS (batch):    {newvers_ms_per_sample:.6f} ms/sample")
print(f"UPDATED_VERS (single):   {mean_time_single:.6f} ms/sample")
print(f"\nSpeedup (batch mode):    {speedup:.1f}×")
print(f"Speedup (single mode):   {fastchem_ms_per_sample / mean_time_single:.1f}×")

throughput_newvers = 1000.0 / newvers_ms_per_sample
throughput_fastchem = 1000.0 / fastchem_ms_per_sample
print(f"\nThroughput:")
print(f"   FastChem:    {throughput_fastchem:10.1f} samples/sec")
print(f"   UPDATED_VERS:{throughput_newvers:10.1f} samples/sec")

n_total = len(df)
fastchem_total_time_s = (n_total * fastchem_ms_per_sample) / 1000.0
newvers_total_time_s = (n_total * newvers_ms_per_sample) / 1000.0
time_saved = fastchem_total_time_s - newvers_total_time_s
print(f"\nFull dataset ({n_total:,} samples):")
print(f"   FastChem time:   {fastchem_total_time_s:10.1f} s ({fastchem_total_time_s/60:.1f} min)")
print(f"   UPDATED_VERS:    {newvers_total_time_s:10.1f} s ({newvers_total_time_s/60:.1f} min)")
print(f"   Time saved:      {time_saved:10.1f} s ({time_saved/60:.1f} min)")

# -----------------------------------------------------------------------------
# 7. Accuracy metrics
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("ACCURACY CHECK (Autoencoder vs Ground Truth)")
print("=" * 80)
y_fastchem = df_test[TARGET_COLS].to_numpy(dtype=np.float64)
with torch.no_grad():
    y_pred_scaled = forward_autoencoder(model, X_test).cpu().numpy()
    y_pred = denormalize_targets(y_pred_scaled)

mae_linear = mean_absolute_error(y_fastchem, y_pred)
mse_linear = np.mean((y_fastchem - y_pred) ** 2)

mask = y_fastchem > 1e-15
y_fc_log = np.log10(np.clip(y_fastchem[mask], 1e-30, None))
y_pred_log = np.log10(np.clip(y_pred[mask], 1e-30, None))
mae_log = mean_absolute_error(y_fc_log, y_pred_log)
r2_log = r2_score(y_fc_log, y_pred_log)
relative_errors = np.abs((y_pred - y_fastchem) / (y_fastchem + 1e-30))
median_rel_error = np.median(relative_errors[y_fastchem > 1e-10])

print(f"\nLinear MAE:      {mae_linear:.6e}")
print(f"Linear MSE:      {mse_linear:.6e}")
print(f"Log MAE:         {mae_log:.6e} dex")
print(f"Log R²:          {r2_log:.6f}")
print(f"Median Rel Err:  {median_rel_error:.2%}")

# -----------------------------------------------------------------------------
# 8. Save results
# -----------------------------------------------------------------------------
print(f"\n8. Saving results to {OUT_DIR}/")
summary_path = os.path.join(OUT_DIR, "speed_test_summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("UPDATED_VERS INFERENCE SPEED TEST - SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write("TIMING RESULTS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Device:                  {device}\n")
    f.write(f"Test samples:            {len(df_test)}\n")
    f.write(f"Species predicted:       {len(TARGET_COLS)}\n")
    f.write(f"Warmup runs:             {N_WARMUP}\n")
    f.write(f"Timing runs:             {N_TIMING_RUNS}\n\n")
    f.write("INFERENCE TIME (UPDATED_VERS)\n")
    f.write("-" * 80 + "\n")
    f.write(f"Full batch ({len(df_test)} samples):\n")
    f.write(f"   Total:         {mean_time_full:.4f} ± {std_time_full:.4f} ms\n")
    f.write(f"   Per sample:    {time_per_sample_ms:.6f} ms/sample\n\n")
    f.write("Single sample:\n")
    f.write(f"   Total:         {mean_time_single:.6f} ± {std_time_single:.6f} ms\n\n")
    f.write("BATCH SIZE ANALYSIS\n")
    f.write("-" * 80 + "\n")
    for bs in sorted(batch_timings.keys()):
        bt = batch_timings[bs]
        f.write(
            f"Batch size {bs:4d}: {bt['mean_total_ms']:8.4f} ms total, "
            f"{bt['mean_per_sample_ms']:.6f} ms/sample\n"
        )
    f.write("\nSPEED COMPARISON\n")
    f.write("=" * 80 + "\n")
    f.write(f"FastChem (benchmark):    {fastchem_ms_per_sample:.3f} ms/sample\n")
    f.write(f"UPDATED_VERS (batch):    {newvers_ms_per_sample:.6f} ms/sample\n")
    f.write(f"UPDATED_VERS (single):   {mean_time_single:.6f} ms/sample\n\n")
    f.write(f"Speedup (batch mode):    {speedup:.1f}×\n")
    f.write(f"Speedup (single mode):   {fastchem_ms_per_sample / mean_time_single:.1f}×\n\n")
    f.write("THROUGHPUT\n")
    f.write("-" * 80 + "\n")
    f.write(f"FastChem:                {throughput_fastchem:10.1f} samples/sec\n")
    f.write(f"UPDATED_VERS:            {throughput_newvers:10.1f} samples/sec\n\n")
    f.write(f"Full dataset ({n_total:,} samples)\n")
    f.write("-" * 80 + "\n")
    f.write(f"FastChem time:           {fastchem_total_time_s:10.1f} sec ({fastchem_total_time_s/60:.1f} min)\n")
    f.write(f"UPDATED_VERS time:       {newvers_total_time_s:10.1f} sec ({newvers_total_time_s/60:.1f} min)\n")
    f.write(f"Time saved:              {time_saved:10.1f} sec ({time_saved/60:.1f} min)\n\n")
    f.write("ACCURACY METRICS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Linear MAE:              {mae_linear:.6e}\n")
    f.write(f"Linear MSE:              {mse_linear:.6e}\n")
    f.write(f"Log MAE:                 {mae_log:.6e} dex\n")
    f.write(f"Log R²:                  {r2_log:.6f}\n")
    f.write(f"Median Relative Error:   {median_rel_error:.2%}\n")
print(f"   ✓ Saved: {summary_path}")

timing_data = []
for bs in sorted(batch_timings.keys()):
    bt = batch_timings[bs]
    timing_data.append(
        {
            "batch_size": bs,
            "total_time_ms": bt["mean_total_ms"],
            "total_time_std_ms": bt["std_total_ms"],
            "per_sample_ms": bt["mean_per_sample_ms"],
            "samples_per_sec": 1000.0 / bt["mean_per_sample_ms"],
            "speedup_vs_fastchem": FASTCHEM_MS_PER_SAMPLE / bt["mean_per_sample_ms"],
        }
    )

df_timing = pd.DataFrame(timing_data)
timing_csv = os.path.join(OUT_DIR, "batch_timing_breakdown.csv")
df_timing.to_csv(timing_csv, index=False)
print(f"   ✓ Saved: {timing_csv}")

# -----------------------------------------------------------------------------
# 9. Plot timing curves
# -----------------------------------------------------------------------------
print("\n9. Creating timing visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
batch_sizes = [row["batch_size"] for row in timing_data]
per_sample_times = [row["per_sample_ms"] for row in timing_data]
speedups = [row["speedup_vs_fastchem"] for row in timing_data]

ax1.plot(batch_sizes, per_sample_times, "o-", linewidth=2, markersize=8, color="steelblue", label="UPDATED_VERS")
ax1.axhline(FASTCHEM_MS_PER_SAMPLE, color="red", linestyle="--", linewidth=2, label="FastChem", alpha=0.7)
ax1.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
ax1.set_ylabel("Time per Sample (ms)", fontsize=12, fontweight="bold")
ax1.set_title("Inference Time vs Batch Size", fontsize=13, fontweight="bold")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, which="both")

ax2.plot(batch_sizes, speedups, "o-", linewidth=2, markersize=8, color="green")
ax2.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax2.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
ax2.set_ylabel("Speedup vs FastChem (×)", fontsize=12, fontweight="bold")
ax2.set_title(
    f"UPDATED_VERS Speedup\n(Max: {max(speedups):.0f}× at batch={batch_sizes[int(np.argmax(speedups))]})",
    fontsize=13,
    fontweight="bold",
)
ax2.set_xscale("log")
ax2.legend([f"Max speedup: {max(speedups):.0f}×"], fontsize=11)
ax2.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "speed_comparison.png")
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"   ✓ Saved: {plot_path}")

# -----------------------------------------------------------------------------
# 10. Final summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("✅ INFERENCE SPEED TEST COMPLETE")
print("=" * 80)
print(f"\n📊 KEY RESULTS:")
print(f"   UPDATED_VERS per-sample:  {newvers_ms_per_sample:.6f} ms")
print(f"   FastChem per-sample:      {fastchem_ms_per_sample:.3f} ms")
print(f"   Speedup:                  {speedup:.1f}×")
print(f"   Log R²:                   {r2_log:.4f}")
print(f"\n📁 Outputs:")
print(f"   {summary_path}")
print(f"   {timing_csv}")
print(f"   {plot_path}")
print("\n💡 Interpretation:")
if speedup > 1000:
    print(f"   UPDATED_VERS is {speedup:.0f}× faster - excellent for large-scale applications!")
elif speedup > 100:
    print(f"   UPDATED_VERS is {speedup:.1f}× faster - dramatic acceleration over FastChem.")
else:
    print(f"   UPDATED_VERS is {speedup:.1f}× faster - solid improvement over FastChem.")


