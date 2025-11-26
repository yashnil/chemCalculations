#!/usr/bin/env python3
"""
Remove low-temperature samples that cause the vertical stripe artifact.

The stripe at 0.01-0.02 abundance happens with low-temperature, high-entropy
mixtures where many species are active simultaneously. 

Solution: Filter out the coldest 20% of samples (T-bin 0), as done in v8.
"""

import pandas as pd
import numpy as np

INPUT_CSV = '/Users/yashnilmohanty/Desktop/chemCalculations/v10/all_gas_v10_format.csv'
OUTPUT_CSV = '/Users/yashnilmohanty/Desktop/chemCalculations/v10/all_gas_v10_no_stripe.csv'

print("Loading CSV...")
df = pd.read_csv(INPUT_CSV)
print(f"Original: {len(df):,} rows")
print(f"Temperature range: {df['T_K'].min():.1f} - {df['T_K'].max():.1f} K\n")

# Drop low-T bin (same as v8's proven fix)
print("Creating temperature bins...")
df['T_bin'] = pd.qcut(df['T_K'], 5, labels=False, duplicates='drop')

for i in range(5):
    bin_df = df[df['T_bin'] == i]
    if len(bin_df) > 0:
        print(f"  Bin {i}: T = {bin_df['T_K'].min():.0f}-{bin_df['T_K'].max():.0f} K, n={len(bin_df):,}")

print("\nDropping T-bin 0 (coldest 20%)...")
df_filtered = df[df['T_bin'] != 0].copy()
df_filtered.drop(columns=['T_bin'], inplace=True)

print(f"After filtering: {len(df_filtered):,} rows")
print(f"Temperature range: {df_filtered['T_K'].min():.1f} - {df_filtered['T_K'].max():.1f} K")
print(f"Dropped: {len(df) - len(df_filtered):,} rows ({100*(len(df)-len(df_filtered))/len(df):.1f}%)\n")

# Save
df_filtered.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved: {OUTPUT_CSV}")
print(f"\n📝 Next steps:")
print(f"1. Update run_mlp.py line 33:")
print(f"   CSV_PATH = '{OUTPUT_CSV}'")
print(f"2. Re-run training: python run_mlp.py")
print(f"3. Re-run plot: python plot.py")
print(f"\nThis should eliminate the vertical stripe!")

