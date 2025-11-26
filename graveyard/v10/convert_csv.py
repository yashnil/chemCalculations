#!/usr/bin/env python3
"""
Convert raw all_gas.csv (with temperature/pressure/comp_*) 
to v10 format (with T_K/P_bar/abund_*_dex)

This script reads the raw CSV and creates a properly formatted version
that the v10 training code expects.
"""

import pandas as pd
import numpy as np
import os

# Use the Fastchemlp CSV (16k rows - good for initial training)
INPUT_CSV = '/Users/yashnilmohanty/Desktop/chemCalculations/Fastchemlp/all_gas.csv'
OUTPUT_CSV = '/Users/yashnilmohanty/Desktop/chemCalculations/v10/all_gas_v10_format.csv'

print("=" * 60)
print("CSV CONVERSION FOR V10")
print("=" * 60)
print(f"Input:  {INPUT_CSV}")
print(f"Output: {OUTPUT_CSV}\n")

# Check input exists
if not os.path.exists(INPUT_CSV):
    print(f"❌ Error: Input file not found!")
    print(f"   {INPUT_CSV}")
    exit(1)

# Load
print("Loading CSV...")
df = pd.read_csv(INPUT_CSV)
print(f"✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols\n")

# Show current column names
print("Current columns:")
print(f"  Temperature: {'temperature' if 'temperature' in df.columns else '❌ MISSING'}")
print(f"  Pressure:    {'pressure' if 'pressure' in df.columns else '❌ MISSING'}")
print(f"  Elements:    comp_H, comp_O, comp_C, comp_N, comp_S\n")

# Rename T and P
df.rename(columns={
    'temperature': 'T_K',
    'pressure': 'P_bar'
}, inplace=True)
print("Step 1: Renamed columns")
print("  ✓ temperature → T_K")
print("  ✓ pressure → P_bar\n")

# Create abund_*_dex columns
# Formula: abund_X_dex = 12 + log10(N_X / N_H)
print("Step 2: Creating abundance columns in dex scale")
elements = ['H', 'O', 'C', 'N', 'S']

for elem in elements:
    abund_col = f'abund_{elem}_dex'
    
    if elem == 'H':
        # Hydrogen is the reference: always 12.0
        df[abund_col] = 12.0
    else:
        comp_col = f'comp_{elem}'
        # Compute ratio and convert to dex scale
        ratio = df[comp_col] / (df['comp_H'] + 1e-100)  # Avoid division by zero
        df[abund_col] = 12.0 + np.log10(np.clip(ratio, 1e-100, None))
    
    print(f"  ✓ {abund_col:15s}: [{df[abund_col].min():7.2f}, {df[abund_col].max():7.2f}]")

# Save
print(f"\nSaving converted CSV...")
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print("✅ CONVERSION COMPLETE!")
print("=" * 60)
print(f"Output file: {OUTPUT_CSV}")
print(f"Rows:        {df.shape[0]:,}")
print(f"Columns:     {df.shape[1]}")
print("\nNew columns added:")
print("  • T_K (renamed from temperature)")
print("  • P_bar (renamed from pressure)")
print("  • abund_H_dex, abund_O_dex, abund_C_dex, abund_N_dex, abund_S_dex")
print("\n✓ Ready for training!")
print("=" * 60)

