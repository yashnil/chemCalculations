#!/usr/bin/env python3
"""
NEW_VERS Inference Speed Test

Measures exact per-point inference time for NEW_VERS and compares to FastChem benchmarks.
Provides detailed timing breakdown and speedup calculations.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict, List

# Import from best_model
sys.path.append('runs_mlp_NEW_VERS')
from best_model import load_model, normalize_inputs, denormalize_targets, TARGET_COLS, INPUT_COLS

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_PATH = 'all_gas_v10_no_stripe_clean.csv'
OUT_DIR = 'runs_mlp_NEW_VERS/speed_test'
N_WARMUP = 10         # Warmup iterations (discard timing)
N_TIMING_RUNS = 100   # Number of timing runs for accurate measurement
BATCH_SIZES = [1, 10, 50, 100, 621]  # Test different batch sizes

# FastChem benchmark (from v8 documentation and typical runs)
FASTCHEM_MS_PER_SAMPLE = 7.0  # Conservative estimate from baseline_checks

print("=" * 80)
print("NEW_VERS INFERENCE SPEED TEST")
print("=" * 80)

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1. LOAD MODEL AND DATA
# =============================================================================
print("\n1. Loading NEW_VERS model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")

model = load_model(device=device)
model.eval()
print(f"   ✓ Model loaded: {len(INPUT_COLS)} inputs → {len(TARGET_COLS)} species")

print("\n2. Loading test data...")
df = pd.read_csv(CSV_PATH)

# Get test split indices from checkpoint
checkpoint = torch.load('runs_mlp_NEW_VERS/best.pt', map_location='cpu', weights_only=True)
test_idx = checkpoint['config']['splits']['test_idx']
df_test = df.iloc[test_idx].reset_index(drop=True)

print(f"   ✓ Test set: {len(df_test)} samples")
print(f"   ✓ T range: {df_test['T_K'].min():.0f}-{df_test['T_K'].max():.0f} K")
print(f"   ✓ P range: {df_test['P_bar'].min():.1e}-{df_test['P_bar'].max():.1e} bar")

# Prepare normalized inputs
X_test = normalize_inputs(df_test).to(device)
print(f"   ✓ Normalized inputs shape: {X_test.shape}")

# =============================================================================
# 2. WARMUP PHASE
# =============================================================================
print(f"\n3. Warmup phase ({N_WARMUP} iterations)...")
for i in range(N_WARMUP):
    with torch.no_grad():
        _ = model(X_test)
if device.type == 'cuda':
    torch.cuda.synchronize()
print("   ✓ Warmup complete")

# =============================================================================
# 3. FULL BATCH INFERENCE TIMING
# =============================================================================
print(f"\n4. Full batch inference timing ({N_TIMING_RUNS} runs)...")

times_full = []
for _ in range(N_TIMING_RUNS):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    t_start = time.perf_counter()
    with torch.no_grad():
        y_pred_scaled = model(X_test)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    t_end = time.perf_counter()
    times_full.append((t_end - t_start) * 1000)  # Convert to ms

times_full = np.array(times_full)
mean_time_full = times_full.mean()
std_time_full = times_full.std()
median_time_full = np.median(times_full)

print(f"   Full batch ({len(df_test)} samples):")
print(f"     Mean:   {mean_time_full:.4f} ms")
print(f"     Median: {median_time_full:.4f} ms")
print(f"     Std:    {std_time_full:.4f} ms")

# Per-sample timing
time_per_sample_ms = mean_time_full / len(df_test)
print(f"\n   ✓ Per-sample inference: {time_per_sample_ms:.6f} ms/sample")

# =============================================================================
# 4. SINGLE-SAMPLE INFERENCE TIMING
# =============================================================================
print(f"\n5. Single-sample inference timing ({N_TIMING_RUNS} runs)...")

# Pick a random test sample
sample_idx = len(df_test) // 2
X_single = X_test[sample_idx:sample_idx+1]

times_single = []
for _ in range(N_TIMING_RUNS):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    t_start = time.perf_counter()
    with torch.no_grad():
        _ = model(X_single)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    t_end = time.perf_counter()
    times_single.append((t_end - t_start) * 1000)  # ms

times_single = np.array(times_single)
mean_time_single = times_single.mean()
std_time_single = times_single.std()

print(f"   Single sample:")
print(f"     Mean:   {mean_time_single:.6f} ms")
print(f"     Median: {np.median(times_single):.6f} ms")
print(f"     Std:    {std_time_single:.6f} ms")

# =============================================================================
# 5. VARIABLE BATCH SIZE TIMING
# =============================================================================
print(f"\n6. Variable batch size timing...")

batch_timings = {}
for batch_size in BATCH_SIZES:
    if batch_size > len(df_test):
        continue
    
    X_batch = X_test[:batch_size]
    
    times_batch = []
    for _ in range(N_TIMING_RUNS):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model(X_batch)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t_end = time.perf_counter()
        times_batch.append((t_end - t_start) * 1000)
    
    times_batch = np.array(times_batch)
    batch_timings[batch_size] = {
        'mean_total_ms': times_batch.mean(),
        'std_total_ms': times_batch.std(),
        'mean_per_sample_ms': times_batch.mean() / batch_size,
    }
    
    print(f"   Batch={batch_size:4d}: {batch_timings[batch_size]['mean_total_ms']:8.4f} ms total, "
          f"{batch_timings[batch_size]['mean_per_sample_ms']:.6f} ms/sample")

# =============================================================================
# 6. FASTCHEM COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("FASTCHEM vs NEW_VERS SPEED COMPARISON")
print("=" * 80)

# NEW_VERS timing (use full batch as most realistic)
newvers_ms_per_sample = time_per_sample_ms

# FastChem timing (from benchmarks)
fastchem_ms_per_sample = FASTCHEM_MS_PER_SAMPLE

# Speedup
speedup = fastchem_ms_per_sample / newvers_ms_per_sample

print(f"\nFastChem (typical):      {fastchem_ms_per_sample:.3f} ms/sample")
print(f"NEW_VERS (batch):        {newvers_ms_per_sample:.6f} ms/sample")
print(f"NEW_VERS (single):       {mean_time_single:.6f} ms/sample")
print(f"\nSpeedup (batch mode):    {speedup:.1f}×")
print(f"Speedup (single mode):   {fastchem_ms_per_sample / mean_time_single:.1f}×")

# Throughput
throughput_newvers = 1000.0 / newvers_ms_per_sample  # samples/second
throughput_fastchem = 1000.0 / fastchem_ms_per_sample

print(f"\nThroughput:")
print(f"   FastChem:    {throughput_fastchem:10.1f} samples/sec")
print(f"   NEW_VERS:    {throughput_newvers:10.1f} samples/sec")

# Time savings for full dataset
n_total = len(df)
fastchem_total_time_s = (n_total * fastchem_ms_per_sample) / 1000.0
newvers_total_time_s = (n_total * newvers_ms_per_sample) / 1000.0
time_saved = fastchem_total_time_s - newvers_total_time_s

print(f"\nFull dataset ({n_total:,} samples):")
print(f"   FastChem time:   {fastchem_total_time_s:10.1f} seconds ({fastchem_total_time_s/60:.1f} minutes)")
print(f"   NEW_VERS time:   {newvers_total_time_s:10.1f} seconds ({newvers_total_time_s/60:.1f} minutes)")
print(f"   Time saved:      {time_saved:10.1f} seconds ({time_saved/60:.1f} minutes)")

# =============================================================================
# 7. ACCURACY vs SPEED SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("ACCURACY CHECK (NEW_VERS vs Ground Truth)")
print("=" * 80)

# Get predictions and ground truth
y_fastchem = df_test[TARGET_COLS].values
with torch.no_grad():
    y_pred_scaled = model(X_test).cpu().numpy()
    y_pred = denormalize_targets(y_pred_scaled)

# Compute metrics
mae_linear = mean_absolute_error(y_fastchem, y_pred)
mse_linear = np.mean((y_fastchem - y_pred)**2)

# Log space
mask = y_fastchem > 1e-15
y_fc_log = np.log10(np.clip(y_fastchem[mask], 1e-30, None))
y_pred_log = np.log10(np.clip(y_pred[mask], 1e-30, None))
mae_log = mean_absolute_error(y_fc_log, y_pred_log)
r2_log = r2_score(y_fc_log, y_pred_log)

print(f"\nLinear MAE:      {mae_linear:.6e}")
print(f"Linear MSE:      {mse_linear:.6e}")
print(f"Log MAE:         {mae_log:.6e} dex")
print(f"Log R²:          {r2_log:.6f}")

# Relative error
relative_errors = np.abs((y_pred - y_fastchem) / (y_fastchem + 1e-30))
median_rel_error = np.median(relative_errors[y_fastchem > 1e-10])

print(f"Median Rel Err:  {median_rel_error:.2%}")

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================
print(f"\n8. Saving results to {OUT_DIR}/")

# Summary file
with open(os.path.join(OUT_DIR, 'speed_test_summary.txt'), 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("NEW_VERS INFERENCE SPEED TEST - SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("TIMING RESULTS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Device:                  {device}\n")
    f.write(f"Test samples:            {len(df_test)}\n")
    f.write(f"Species predicted:       {len(TARGET_COLS)}\n")
    f.write(f"Warmup runs:             {N_WARMUP}\n")
    f.write(f"Timing runs:             {N_TIMING_RUNS}\n\n")
    
    f.write("INFERENCE TIME (NEW_VERS)\n")
    f.write("-" * 80 + "\n")
    f.write(f"Full batch ({len(df_test)} samples):\n")
    f.write(f"   Total:         {mean_time_full:.4f} ± {std_time_full:.4f} ms\n")
    f.write(f"   Per sample:    {time_per_sample_ms:.6f} ms/sample\n\n")
    
    f.write(f"Single sample:\n")
    f.write(f"   Total:         {mean_time_single:.6f} ± {std_time_single:.6f} ms\n\n")
    
    f.write("BATCH SIZE ANALYSIS\n")
    f.write("-" * 80 + "\n")
    for bs in sorted(batch_timings.keys()):
        bt = batch_timings[bs]
        f.write(f"Batch size {bs:4d}: {bt['mean_total_ms']:8.4f} ms total, "
                f"{bt['mean_per_sample_ms']:.6f} ms/sample\n")
    
    f.write("\n\nSPEED COMPARISON: NEW_VERS vs FASTCHEM\n")
    f.write("=" * 80 + "\n")
    f.write(f"FastChem (benchmark):    {fastchem_ms_per_sample:.3f} ms/sample\n")
    f.write(f"NEW_VERS (batch):        {newvers_ms_per_sample:.6f} ms/sample\n")
    f.write(f"NEW_VERS (single):       {mean_time_single:.6f} ms/sample\n\n")
    
    f.write(f"Speedup (batch mode):    {speedup:.1f}×\n")
    f.write(f"Speedup (single mode):   {fastchem_ms_per_sample / mean_time_single:.1f}×\n\n")
    
    f.write("THROUGHPUT\n")
    f.write("-" * 80 + "\n")
    f.write(f"FastChem:                {throughput_fastchem:10.1f} samples/sec\n")
    f.write(f"NEW_VERS (batch):        {throughput_newvers:10.1f} samples/sec\n\n")
    
    f.write(f"FULL DATASET ({n_total:,} samples)\n")
    f.write("-" * 80 + "\n")
    f.write(f"FastChem time:           {fastchem_total_time_s:10.1f} sec ({fastchem_total_time_s/60:.1f} min)\n")
    f.write(f"NEW_VERS time:           {newvers_total_time_s:10.1f} sec ({newvers_total_time_s/60:.1f} min)\n")
    f.write(f"Time saved:              {time_saved:10.1f} sec ({time_saved/60:.1f} min)\n\n")
    
    f.write("\nACCURACY METRICS (vs Ground Truth)\n")
    f.write("=" * 80 + "\n")
    f.write(f"Linear MAE:              {mae_linear:.6e}\n")
    f.write(f"Linear MSE:              {mse_linear:.6e}\n")
    f.write(f"Log MAE:                 {mae_log:.6e} dex\n")
    f.write(f"Log R²:                  {r2_log:.6f}\n")
    f.write(f"Median Relative Error:   {median_rel_error:.2%}\n")
    
print(f"   ✓ Saved: {OUT_DIR}/speed_test_summary.txt")

# Detailed timing CSV
timing_data = []
for bs in sorted(batch_timings.keys()):
    bt = batch_timings[bs]
    timing_data.append({
        'batch_size': bs,
        'total_time_ms': bt['mean_total_ms'],
        'total_time_std_ms': bt['std_total_ms'],
        'per_sample_ms': bt['mean_per_sample_ms'],
        'samples_per_sec': 1000.0 / bt['mean_per_sample_ms'],
        'speedup_vs_fastchem': FASTCHEM_MS_PER_SAMPLE / bt['mean_per_sample_ms']
    })

df_timing = pd.DataFrame(timing_data)
df_timing.to_csv(os.path.join(OUT_DIR, 'batch_timing_breakdown.csv'), index=False)
print(f"   ✓ Saved: {OUT_DIR}/batch_timing_breakdown.csv")

# =============================================================================
# 9. CREATE TIMING VISUALIZATION
# =============================================================================
print("\n9. Creating timing visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Per-sample time vs batch size
batch_sizes = [bt['batch_size'] for bt in timing_data]
per_sample_times = [bt['per_sample_ms'] for bt in timing_data]

ax1.plot(batch_sizes, per_sample_times, 'o-', linewidth=2, markersize=8, color='steelblue', label='NEW_VERS')
ax1.axhline(FASTCHEM_MS_PER_SAMPLE, color='red', linestyle='--', linewidth=2, label='FastChem', alpha=0.7)
ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax1.set_ylabel('Time per Sample (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Inference Time vs Batch Size', fontsize=13, fontweight='bold')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, which='both')

# Plot 2: Speedup vs batch size
speedups = [bt['speedup_vs_fastchem'] for bt in timing_data]

ax2.plot(batch_sizes, speedups, 'o-', linewidth=2, markersize=8, color='green')
ax2.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup vs FastChem (×)', fontsize=12, fontweight='bold')
ax2.set_title(f'NEW_VERS Speedup\n(Max: {max(speedups):.0f}× at batch={batch_sizes[np.argmax(speedups)]})', 
              fontsize=13, fontweight='bold')
ax2.set_xscale('log')
ax2.legend([f'Max speedup: {max(speedups):.0f}×'], fontsize=11)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'speed_comparison.png'), dpi=200)
plt.close()
print(f"   ✓ Saved: {OUT_DIR}/speed_comparison.png")

# =============================================================================
# 10. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("✅ INFERENCE SPEED TEST COMPLETE")
print("=" * 80)

print(f"\n📊 KEY RESULTS:")
print(f"   NEW_VERS per-sample:  {newvers_ms_per_sample:.6f} ms")
print(f"   FastChem per-sample:  {fastchem_ms_per_sample:.3f} ms")
print(f"   Speedup:              {speedup:.1f}×")
print(f"   Log R²:               {r2_log:.4f}")

print(f"\n📁 Outputs:")
print(f"   {OUT_DIR}/speed_test_summary.txt")
print(f"   {OUT_DIR}/batch_timing_breakdown.csv")
print(f"   {OUT_DIR}/speed_comparison.png")

print("\n💡 Interpretation:")
if speedup > 1000:
    print(f"   NEW_VERS is {speedup:.0f}× faster - excellent for large-scale applications!")
elif speedup > 100:
    print(f"   NEW_VERS is {speedup:.0f}× faster - suitable for real-time inference!")
else:
    print(f"   NEW_VERS is {speedup:.0f}× faster - good speedup for iterative workflows!")

print("=" * 80)

