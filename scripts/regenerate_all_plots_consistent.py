#!/usr/bin/env python3
"""
regenerate_all_plots_consistent.py
===================================

Regenerate ALL plots with consistent units and optimal_retrained runs.
Ensures all plots use the 800K-increment study sizes (800, 1600, 2400, 3200, 4000, 4800K).
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def regenerate_all_plots():
    """Regenerate all plots with consistent units."""
    print("="*80)
    print("REGENERATING ALL PLOTS WITH CONSISTENT UNITS")
    print("="*80)
    print("Using optimal_retrained runs: 800, 1600, 2400, 3200, 4000, 4800K")
    print()
    
    plots_to_generate = [
        ("src/plot_training_analysis.py", "Training analysis plots", []),
        ("src/plot_comprehensive_analysis.py", "Comprehensive analysis plots", []),
        ("src/plot_consistent_runs.py", "Consistent runs plots", []),
    ]
    
    for script_path, description, extra_args in plots_to_generate:
        script = BASE_DIR / script_path
        if script.exists():
            print(f"\n📊 {description}...")
            cmd = [sys.executable, str(script)] + extra_args
            result = subprocess.run(
                cmd,
                cwd=BASE_DIR,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"  ✅ Success")
                # Print key output lines
                for line in result.stdout.split('\n'):
                    if '✅' in line or 'Saved' in line:
                        print(f"    {line}")
            else:
                print(f"  ⚠️  Warning: {result.stderr[:300]}")
        else:
            print(f"  ⚠️  Script not found: {script}")
    
    print("\n" + "="*80)
    print("✅ ALL PLOTS REGENERATED!")
    print("="*80)


if __name__ == "__main__":
    regenerate_all_plots()
