#!/usr/bin/env python3
"""
Analyze species statistics to determine static ordering.
Computes mean, min, max, std for all species and orders by mean abundance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_species_statistics(csv_path: str, output_csv: str = "plots/species_statistics.csv"):
    """Analyze all species and create statistics table."""
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Get input columns
    input_cols = ['T_K', 'P_bar', 'abund_C_dex', 'abund_H_dex', 'abund_N_dex', 
                   'abund_O_dex', 'abund_S_dex']
    if 'fZ' in df.columns:
        input_cols.append('fZ')
    if 'fZ_dex' in df.columns:
        input_cols.append('fZ_dex')
    
    # Get all species columns (exclude input columns and metadata)
    never_target = {'flag', 'flag_msg', 'mean_molecular_weight', 'total_element_density'}
    species_cols = [c for c in df.columns 
                    if c not in input_cols 
                    and not c.startswith('abund_')
                    and c not in never_target
                    and c not in ['comp_']]
    
    # Filter to numeric columns only
    species_cols = [c for c in species_cols if c in df.select_dtypes(include=[np.number]).columns]
    
    print(f"Analyzing {len(species_cols)} species...")
    
    # Compute statistics
    stats = []
    for col in species_cols:
        vals = df[col].to_numpy(dtype=float, copy=False)
        vals = np.where(np.isfinite(vals), np.clip(vals, 0.0, None), np.nan)
        
        if np.isnan(vals).all():
            continue
        
        non_zero_count = np.sum(vals > 1e-30)
        stats.append({
            'species': col,
            'mean': float(np.nanmean(vals)),
            'min': float(np.nanmin(vals)),
            'max': float(np.nanmax(vals)),
            'std': float(np.nanstd(vals)),
            'median': float(np.nanmedian(vals)),
            'p95': float(np.nanpercentile(vals, 95)),
            'non_zero_fraction': float(non_zero_count / len(vals)),
            'non_zero_count': int(non_zero_count)
        })
    
    # Sort by mean abundance
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('mean', ascending=False).reset_index(drop=True)
    
    # Add cumulative coverage
    total_mass = stats_df['mean'].sum()
    stats_df['cumulative_coverage'] = stats_df['mean'].cumsum() / total_mass * 100
    
    # Save
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Saved to {output_csv}")
    print(f"\n{'='*100}")
    print(f"TOP 40 SPECIES BY MEAN ABUNDANCE")
    print(f"{'='*100}")
    print(f"\n{'Rank':<6} {'Species':<15} {'Mean':<15} {'Min':<15} {'Max':<15} {'Std':<15} {'Median':<15} {'Non-Zero%':<12} {'Coverage%':<12}")
    print("-"*100)
    
    for idx, row in stats_df.head(40).iterrows():
        print(f"{idx+1:<6} {row['species']:<15} {row['mean']:<15.6e} {row['min']:<15.6e} "
              f"{row['max']:<15.6e} {row['std']:<15.6e} {row['median']:<15.6e} "
              f"{row['non_zero_fraction']*100:<12.2f} {row['cumulative_coverage']:<12.2f}")
    
    print(f"\n{'='*100}")
    print(f"COVERAGE ANALYSIS")
    print(f"{'='*100}")
    print(f"\nCumulative mass coverage:")
    for n in [20, 24, 32, 36, 40]:
        if n <= len(stats_df):
            coverage = stats_df.iloc[n-1]['cumulative_coverage']
            mean_val = stats_df.iloc[n-1]['mean']
            species_name = stats_df.iloc[n-1]['species']
            print(f"  Top {n:2d} species: {coverage:6.2f}% coverage (last: {species_name}, mean={mean_val:.6e})")
    
    print(f"\nRecommendations:")
    print(f"  - Top 20: {stats_df.iloc[19]['cumulative_coverage']:.2f}% coverage")
    if len(stats_df) >= 24:
        print(f"  - Top 24: {stats_df.iloc[23]['cumulative_coverage']:.2f}% coverage")
    if len(stats_df) >= 32:
        print(f"  - Top 32: {stats_df.iloc[31]['cumulative_coverage']:.2f}% coverage")
    if len(stats_df) >= 36:
        print(f"  - Top 36: {stats_df.iloc[35]['cumulative_coverage']:.2f}% coverage")
    
    return stats_df

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/datasets/all_gas_fastchem_x160.csv"
    analyze_species_statistics(csv_path)

