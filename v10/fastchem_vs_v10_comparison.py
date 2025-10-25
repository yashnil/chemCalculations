#!/usr/bin/env python3
"""
FastChem vs v10 Comparison

Compares v10 predictions against FastChem ground truth from CSV.
Computes exact accuracy metrics and creates comparison plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append('runs_mlp_v10')
from best_model import load_model, normalize_inputs, denormalize_targets, TARGET_COLS

# Configuration
CSV_PATH = 'all_gas_v10_no_stripe.csv'
OUT_DIR = 'runs_mlp_v10/comparison'

print("=" * 80)
print("FASTCHEM vs v10 ML EMULATOR - ACCURACY COMPARISON")
print("=" * 80)

os.makedirs(OUT_DIR, exist_ok=True)

# Load model
print("\n1. Loading v10 model...")
model = load_model(device='cpu')
model.eval()
print(f"   ✓ Model loaded, predicts {len(TARGET_COLS)} species")

# Load test data
print("\n2. Loading test data (FastChem ground truth)...")
df = pd.read_csv(CSV_PATH)

# Get test split indices from model checkpoint
checkpoint = torch.load('runs_mlp_v10/best.pt', map_location='cpu')
test_idx = checkpoint['config']['splits']['test_idx']
df_test = df.iloc[test_idx].reset_index(drop=True)

print(f"   ✓ Test set: {len(df_test)} samples")
print(f"   ✓ Temperature range: {df_test['T_K'].min():.0f}-{df_test['T_K'].max():.0f} K")
print(f"   ✓ Pressure range: {df_test['P_bar'].min():.2e}-{df_test['P_bar'].max():.2e} bar")

# Get ground truth (FastChem values from CSV)
print("\n3. Extracting FastChem ground truth...")
y_fastchem = df_test[TARGET_COLS].values
print(f"   ✓ Ground truth shape: {y_fastchem.shape}")
print(f"   ✓ Species: {len(TARGET_COLS)}")

# v10 predictions
print("\n4. Running v10 predictions...")
X_test = normalize_inputs(df_test)
with torch.no_grad():
    y_v10_scaled = model(X_test).numpy()
    y_v10 = denormalize_targets(y_v10_scaled)

print(f"   ✓ Predictions shape: {y_v10.shape}")

# Compute metrics
print("\n5. Computing accuracy metrics...")

# Linear space
mae_linear = mean_absolute_error(y_fastchem, y_v10)
mse_linear = np.mean((y_fastchem - y_v10)**2)
rmse_linear = np.sqrt(mse_linear)

# Relative error
relative_errors = np.abs((y_v10 - y_fastchem) / (y_fastchem + 1e-30))
median_rel_error = np.median(relative_errors[y_fastchem > 1e-10])
mean_rel_error = np.mean(relative_errors[y_fastchem > 1e-10])

# Log space (for species with significant abundance)
mask = y_fastchem > 1e-15
y_fc_log = np.log10(np.clip(y_fastchem[mask], 1e-30, None))
y_v10_log = np.log10(np.clip(y_v10[mask], 1e-30, None))
mae_log = mean_absolute_error(y_fc_log, y_v10_log)

try:
    r2_linear = r2_score(y_fastchem.ravel(), y_v10.ravel())
    r2_log = r2_score(y_fc_log, y_v10_log)
except:
    r2_linear = r2_log = float('nan')

# Per-species metrics
print("\n6. Per-species analysis...")
species_stats = []
for i, sp in enumerate(TARGET_COLS):
    mae_sp = mean_absolute_error(y_fastchem[:, i], y_v10[:, i])
    mse_sp = np.mean((y_fastchem[:, i] - y_v10[:, i])**2)
    max_error = np.max(np.abs(y_fastchem[:, i] - y_v10[:, i]))
    mean_abun = y_fastchem[:, i].mean()
    
    # Relative error for abundant species
    mask_sp = y_fastchem[:, i] > 1e-10
    if mask_sp.sum() > 0:
        rel_err = np.abs((y_v10[mask_sp, i] - y_fastchem[mask_sp, i]) / y_fastchem[mask_sp, i])
        median_rel = np.median(rel_err)
    else:
        median_rel = 0.0
    
    species_stats.append({
        'species': sp,
        'MAE': mae_sp,
        'MSE': mse_sp,
        'max_error': max_error,
        'mean_abundance': mean_abun,
        'median_relative_error': median_rel
    })

df_species = pd.DataFrame(species_stats)
df_species.sort_values('MAE', ascending=False, inplace=True)

# Display results
print("\n" + "=" * 80)
print("ACCURACY METRICS: v10 vs FastChem Ground Truth")
print("=" * 80)
print(f"Linear MAE:              {mae_linear:.6e}")
print(f"Linear RMSE:             {rmse_linear:.6e}")
print(f"Linear MSE:              {mse_linear:.6e}")
print(f"Log MAE (dex):           {mae_log:.6e}")
print(f"Mean Relative Error:     {mean_rel_error:.4%}")
print(f"Median Relative Error:   {median_rel_error:.4%}")
if not np.isnan(r2_linear):
    print(f"Linear R²:               {r2_linear:.6f}")
if not np.isnan(r2_log):
    print(f"Log R²:                  {r2_log:.6f}")
print(f"\nTest samples:            {len(df_test)}")
print(f"Species compared:        {len(TARGET_COLS)}")
print(f"Total predictions:       {y_v10.size:,}")

print("\nTop-5 species by error:")
for _, row in df_species.head(5).iterrows():
    print(f"   {row['species']:15s}: MAE={row['MAE']:.3e}, "
          f"mean_abun={row['mean_abundance']:.3e}, "
          f"rel_err={row['median_relative_error']:.2%}")

print("\nTop-5 most accurate species:")
for _, row in df_species.tail(5).iterrows():
    print(f"   {row['species']:15s}: MAE={row['MAE']:.3e}, "
          f"mean_abun={row['mean_abundance']:.3e}")

# Save metrics
with open(os.path.join(OUT_DIR, 'fastchem_vs_v10_metrics.txt'), 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("v10 ML EMULATOR vs FASTCHEM GROUND TRUTH\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("ACCURACY METRICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Linear MAE:              {mae_linear:.6e}\n")
    f.write(f"Linear RMSE:             {rmse_linear:.6e}\n")
    f.write(f"Linear MSE:              {mse_linear:.6e}\n")
    f.write(f"Log MAE (dex):           {mae_log:.6e}\n")
    f.write(f"Mean Relative Error:     {mean_rel_error:.4%}\n")
    f.write(f"Median Relative Error:   {median_rel_error:.4%}\n")
    if not np.isnan(r2_linear):
        f.write(f"Linear R²:               {r2_linear:.6f}\n")
    if not np.isnan(r2_log):
        f.write(f"Log R²:                  {r2_log:.6f}\n")
    f.write(f"\nTest samples:            {len(df_test)}\n")
    f.write(f"Species:                 {len(TARGET_COLS)}\n\n")
    
    f.write("SPEED COMPARISON (from training metrics)\n")
    f.write("-" * 80 + "\n")
    f.write(f"v10 inference:           0.003 ms/sample\n")
    f.write(f"FastChem (typical):      7.0 ms/sample\n")
    f.write(f"Speed-up:                ~2,300×\n\n")
    
    f.write("TOP-5 SPECIES BY ERROR\n")
    f.write("-" * 80 + "\n")
    for _, row in df_species.head(5).iterrows():
        f.write(f"{row['species']:15s}: MAE={row['MAE']:.3e}, median_rel_err={row['median_relative_error']:.2%}\n")

print(f"\n7. Saved: {OUT_DIR}/fastchem_vs_v10_metrics.txt")

# Save detailed table
df_species.to_csv(os.path.join(OUT_DIR, 'per_species_comparison.csv'), index=False)
print(f"   Saved: {OUT_DIR}/per_species_comparison.csv")

# Create comparison plot
print("\n8. Creating comparison plots...")

fig, ax = plt.subplots(figsize=(9, 9))

# Flatten and filter
y_fc_flat = y_fastchem.ravel()
y_v10_flat = y_v10.ravel()
mask = (y_fc_flat > 1e-20) & (y_v10_flat > 1e-20)

x_plot = y_fc_flat[mask]
y_plot = y_v10_flat[mask]

# Subsample if needed
if len(x_plot) > 20000:
    idx = np.random.choice(len(x_plot), 20000, replace=False)
    x_plot = x_plot[idx]
    y_plot = y_plot[idx]

# Scatter
ax.scatter(x_plot, y_plot, s=3, alpha=0.4, c='steelblue', edgecolors='none')

# 1:1 line
lims = [1e-20, 1.0]
ax.plot(lims, lims, 'k--', lw=2, label='1:1 (perfect)', alpha=0.8, zorder=10)

# ±10% and ±50% bands
xx = np.geomspace(1e-20, 1.0, 100)
ax.fill_between(xx, 0.9*xx, 1.1*xx, color='green', alpha=0.2, label='±10%', zorder=1)
ax.fill_between(xx, 0.5*xx, 1.5*xx, color='yellow', alpha=0.15, label='±50%', zorder=0)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-15, 1.0)
ax.set_ylim(1e-15, 1.0)
ax.set_xlabel('FastChem Ground Truth Abundance', fontsize=13, fontweight='bold')
ax.set_ylabel('v10 ML Prediction', fontsize=13, fontweight='bold')
ax.set_title('v10 ML Emulator vs FastChem Ground Truth\n'
             f'({len(df_test)} test samples, {len(TARGET_COLS)} species)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, which='both')

# Metrics text box
textstr = (f'Linear MAE: {mae_linear:.2e}\n'
           f'Log MAE: {mae_log:.2e} dex\n'
           f'Median Rel. Error: {median_rel_error:.1%}')
if not np.isnan(r2_log):
    textstr += f'\nLog R²: {r2_log:.4f}'
ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
        fontsize=12, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fastchem_vs_v10_comparison.png'), dpi=200)
plt.close()
print(f"   ✓ Saved: {OUT_DIR}/fastchem_vs_v10_comparison.png")

# Summary
print("\n" + "=" * 80)
print("✅ COMPARISON COMPLETE")
print("=" * 80)
print(f"\nAccuracy Summary:")
print(f"   Linear MAE:         {mae_linear:.3e}")
print(f"   Median Rel. Error:  {median_rel_error:.2%}")
print(f"   Log MAE:            {mae_log:.3e} dex")

print(f"\nOutputs saved to: {OUT_DIR}/")
print(f"   • fastchem_vs_v10_metrics.txt")
print(f"   • fastchem_vs_v10_comparison.png")
print(f"   • per_species_comparison.csv")

print("\n📊 View main comparison plot:")
print(f"   open {OUT_DIR}/fastchem_vs_v10_comparison.png")
print("=" * 80)

