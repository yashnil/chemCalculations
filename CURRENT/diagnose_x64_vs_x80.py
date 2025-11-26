#!/usr/bin/env python3
"""
Diagnose why x80 performs worse than x64 by comparing:
1. Target species differences
2. Training configuration
3. Dataset distribution shifts
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Load summaries
x64_summary = json.load(open(BASE_DIR / "runs_autoencoder_x64" / "summary.json"))
x80_summary = json.load(open(BASE_DIR / "runs_autoencoder_x80" / "summary.json"))

print("=" * 80)
print("TARGET SPECIES COMPARISON")
print("=" * 80)

x64_targets = set(x64_summary["target_cols"])
x80_targets = set(x80_summary["target_cols"])

only_x64 = x64_targets - x80_targets
only_x80 = x80_targets - x64_targets
common = x64_targets & x80_targets

print(f"\nSpecies only in x64: {sorted(only_x64)}")
print(f"Species only in x80: {sorted(only_x80)}")
print(f"Common species: {len(common)}/{len(x64_targets)}")

if only_x64 or only_x80:
    print("\n⚠️  CRITICAL: Target species mismatch detected!")
    print("   This makes the loss values incomparable.")

# Load per-species errors
import pandas as pd

x64_errors = pd.read_csv(BASE_DIR / "runs_autoencoder_x64" / "diagnostics" / "per_species_errors.csv")
x80_errors = pd.read_csv(BASE_DIR / "runs_autoencoder_x80" / "diagnostics" / "per_species_errors.csv")

print("\n" + "=" * 80)
print("PER-SPECIES ERROR COMPARISON (for common species)")
print("=" * 80)

x64_dict = {row['species']: row for _, row in x64_errors.iterrows()}
x80_dict = {row['species']: row for _, row in x80_errors.iterrows()}

print(f"\n{'Species':<10} {'x64 R²':<10} {'x80 R²':<10} {'x64 MAE':<15} {'x80 MAE':<15}")
print("-" * 80)

for species in sorted(common):
    if species in x64_dict and species in x80_dict:
        x64_r2 = x64_dict[species]['R2']
        x80_r2 = x80_dict[species]['R2']
        x64_mae = x64_dict[species]['MAE']
        x80_mae = x80_dict[species]['MAE']
        r2_diff = x80_r2 - x64_r2
        marker = "⚠️" if r2_diff < -0.05 else "✓"
        print(f"{marker} {species:<10} {x64_r2:>8.4f}   {x80_r2:>8.4f}   {x64_mae:>12.2e}   {x80_mae:>12.2e}")

# Check for the problematic species
print("\n" + "=" * 80)
print("SPECIES-SPECIFIC ISSUES")
print("=" * 80)

if "S6" in only_x80:
    s6_info = x80_dict.get("S6", {})
    print(f"\n⚠️  S6 (only in x80): R² = {s6_info.get('R2', 'N/A'):.4f}, MAE = {s6_info.get('MAE', 'N/A'):.2e}")
    print("   This is a difficult sulfur chain species that may be driving the loss increase.")

if "O3S1" in only_x64:
    o3s1_info = x64_dict.get("O3S1", {})
    print(f"\n✓ O3S1 (only in x64): R² = {o3s1_info.get('R2', 'N/A'):.4f}, MAE = {o3s1_info.get('MAE', 'N/A'):.2e}")
    print("   This was replaced by S6 in x80, which is much harder to predict.")

# Training configuration
print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)

print(f"\nx64: {x64_summary['train_samples']} train, {x64_summary['val_samples']} val, {x64_summary['test_samples']} test")
print(f"x80: {x80_summary['train_samples']} train, {x80_summary['val_samples']} val, {x80_summary['test_samples']} test")

print(f"\nx64 val_loss: {x64_summary['val_loss']:.6f}")
print(f"x80 val_loss: {x80_summary['val_loss']:.6f}")
print(f"Difference: {x80_summary['val_loss'] - x64_summary['val_loss']:.6f} ({((x80_summary['val_loss'] - x64_summary['val_loss']) / x64_summary['val_loss'] * 100):.1f}% increase)")

print(f"\nx64 test_loss: {x64_summary['test_loss']:.6f}")
print(f"x80 test_loss: {x80_summary['test_loss']:.6f}")
print(f"Difference: {x80_summary['test_loss'] - x64_summary['test_loss']:.6f} ({((x80_summary['test_loss'] - x64_summary['test_loss']) / x64_summary['test_loss'] * 100):.1f}% increase)")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("""
1. Re-run x80 with TARGET_COLS_MANUAL set to x64's target list to remove the confounder
2. If degradation persists, investigate the extreme high-pressure/sulfur cases
3. Consider longer training (more epochs) or adjusted learning rate schedule for larger datasets
4. The absolute difference is small (~1.1e-4), so this may be within training variance
""")

