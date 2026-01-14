#!/usr/bin/env python3
"""
Generate static species list based on analysis.
Creates JSON files for different species counts (24, 32, 36).
"""

import pandas as pd
import json
from pathlib import Path
import sys

def generate_static_list(stats_csv: str = "plots/species_statistics.csv", 
                        n_species: int = 32,
                        include_electron: bool = True,
                        output_name: str = None):
    """Generate static species list."""
    stats_df = pd.read_csv(stats_csv)
    
    if len(stats_df) < n_species:
        print(f"⚠️  Warning: Only {len(stats_df)} species available, requested {n_species}")
        n_species = len(stats_df)
    
    # Get top N species
    top_species = stats_df.head(n_species)['species'].tolist()
    
    # Ensure e- is first if present
    if include_electron and 'e-' in stats_df['species'].values:
        if 'e-' in top_species:
            top_species.remove('e-')
        top_species.insert(0, 'e-')
    
    # Get coverage
    coverage = float(stats_df.iloc[n_species-1]['cumulative_coverage']) if n_species <= len(stats_df) else None
    
    # Save as JSON
    output = {
        'species': top_species,
        'n_species': len(top_species),
        'coverage': coverage,
        'description': f'Static species list ordered by mean abundance (top {n_species})'
    }
    
    if output_name is None:
        output_name = f"static_species_list_{n_species}.json"
    
    output_path = Path("configs") / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Generated static species list ({len(top_species)} species)")
    print(f"   Coverage: {coverage:.2f}%")
    print(f"   Saved to: {output_path}")
    print(f"\nSpecies list:")
    for i, sp in enumerate(top_species, 1):
        print(f"  {i:2d}. {sp}")
    
    return top_species, output_path

if __name__ == "__main__":
    stats_csv = sys.argv[1] if len(sys.argv) > 1 else "plots/species_statistics.csv"
    
    # Generate lists for 24, 32, 36 species
    print("="*80)
    print("GENERATING STATIC SPECIES LISTS")
    print("="*80)
    print()
    
    for n in [24, 32, 36]:
        print(f"\n{'='*80}")
        print(f"Generating list with {n} species...")
        print(f"{'='*80}")
        generate_static_list(stats_csv, n_species=n, output_name=f"static_species_list_{n}.json")
    
    print(f"\n{'='*80}")
    print("✅ All static species lists generated!")
    print(f"{'='*80}")

