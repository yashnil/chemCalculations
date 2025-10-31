#!/usr/bin/env python3
"""
Debug script to check v9 input transformations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"

# Load and preprocess like baseline_checks.py
df = pd.read_csv(CSV_PATH)

# Drop T-bin 0
df["T_bin"] = pd.qcut(df["temperature"], 5, labels=False, duplicates="drop")
df = df[df["T_bin"] != 0].reset_index(drop=True).drop(columns="T_bin")

print("=" * 60)
print("CHECKING V9 INPUT TRANSFORMATIONS")
print("=" * 60)

# Check raw compositions
print("\nRaw composition statistics:")
for elem in ["comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]:
    print(f"{elem:10s}: min={df[elem].min():.2e}, max={df[elem].max():.2e}, "
          f"mean={df[elem].mean():.2e}")
    zero_count = (df[elem] == 0).sum()
    if zero_count > 0:
        print(f"  ⚠️  {zero_count} zero values found!")

# Create v9 transformations
print("\n" + "=" * 60)
print("V9 TRANSFORMATIONS")
print("=" * 60)

T_max = df["temperature"].max()
print(f"\nT_max = {T_max:.2f} K")

X = pd.DataFrame()
X["temperature_norm"] = df["temperature"] / T_max
X["log_pressure"] = np.log10(df["pressure"])

# Log ratios - check for problems
print("\nComputing log ratios...")
X["log_O_H"] = np.log10(df["comp_O"] / df["comp_H"])
X["log_C_H"] = np.log10(df["comp_C"] / df["comp_H"])
X["log_N_H"] = np.log10(df["comp_N"] / df["comp_H"])
X["log_S_H"] = np.log10(df["comp_S"] / df["comp_H"])

# Check for problematic values
print("\n" + "=" * 60)
print("CHECKING FOR PROBLEMS")
print("=" * 60)

for col in X.columns:
    n_nan = X[col].isna().sum()
    n_inf = np.isinf(X[col]).sum()
    n_finite = np.isfinite(X[col]).sum()
    
    print(f"\n{col}:")
    print(f"  Finite: {n_finite}/{len(X)}")
    if n_nan > 0:
        print(f"  ⚠️  NaN: {n_nan}")
    if n_inf > 0:
        print(f"  ⚠️  Inf: {n_inf}")
    
    if np.isfinite(X[col]).any():
        finite_vals = X[col][np.isfinite(X[col])]
        print(f"  Range: [{finite_vals.min():.3f}, {finite_vals.max():.3f}]")
        print(f"  Mean ± Std: {finite_vals.mean():.3f} ± {finite_vals.std():.3f}")

# Compare with v8 transformations
print("\n" + "=" * 60)
print("COMPARISON: V8 vs V9 FEATURE RANGES")
print("=" * 60)

# v8 style
X_v8 = pd.DataFrame()
X_v8["temperature"] = df["temperature"]
X_v8["pressure"] = np.log10(df["pressure"])
for elem in ["comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]:
    X_v8[elem] = np.log10(df[elem]) + 9.0

print("\nV8 features (before StandardScaler):")
for col in X_v8.columns:
    finite_vals = X_v8[col][np.isfinite(X_v8[col])]
    print(f"  {col:15s}: [{finite_vals.min():8.2f}, {finite_vals.max():8.2f}]  "
          f"std={finite_vals.std():.2f}")

print("\nV9 features (before StandardScaler):")
for col in X.columns:
    if np.isfinite(X[col]).any():
        finite_vals = X[col][np.isfinite(X[col])]
        print(f"  {col:15s}: [{finite_vals.min():8.2f}, {finite_vals.max():8.2f}]  "
              f"std={finite_vals.std():.2f}")

# Plot distributions
print("\nGenerating distribution plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(X.columns):
    finite_vals = X[col][np.isfinite(X[col])]
    axes[i].hist(finite_vals, bins=50, alpha=0.7, edgecolor='black')
    axes[i].set_title(col)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Count')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v9_input_distributions.png', dpi=150)
print("Saved: v9_input_distributions.png")

# Summary
print("\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)

has_issues = False

# Check for infinities
total_inf = sum(np.isinf(X[col]).sum() for col in X.columns)
if total_inf > 0:
    print(f"❌ Found {total_inf} infinite values across all features")
    print("   Cause: log10(0) when element abundance is zero")
    has_issues = True

# Check for extreme ranges
for col in ["log_O_H", "log_C_H", "log_N_H", "log_S_H"]:
    if np.isfinite(X[col]).any():
        finite_vals = X[col][np.isfinite(X[col])]
        if finite_vals.max() - finite_vals.min() > 20:
            print(f"⚠️  {col} has extreme range: {finite_vals.max() - finite_vals.min():.1f}")
            has_issues = True

if not has_issues:
    print("✅ No obvious issues found, but performance is still poor")
    print("   This suggests the log-ratio representation itself may be problematic")
else:
    print("\n🔧 SUGGESTED FIX:")
    print("   Add small epsilon to avoid log(0):")
    print("   X['log_O_H'] = np.log10((df['comp_O'] + 1e-12) / (df['comp_H'] + 1e-12))")

print("\n" + "=" * 60)

