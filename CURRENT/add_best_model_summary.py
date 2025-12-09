#!/usr/bin/env python3
"""
add_best_model_summary.py
==========================

Add best model summary to the bottom of comparison_metrics.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def main():
    comp_path = BASE_DIR / "comparison_metrics.csv"
    if not comp_path.exists():
        print("❌ comparison_metrics.csv not found!")
        return
    
    df = pd.read_csv(comp_path)
    
    # Find best model by test_loss (excluding NaN values)
    valid_df = df[df["test_loss"].notna()]
    if len(valid_df) == 0:
        print("❌ No valid test_loss values found!")
        return
    
    best_idx = valid_df["test_loss"].idxmin()
    best_model = valid_df.loc[best_idx]
    
    # Create summary row
    summary_row = pd.DataFrame([{
        "dataset": "=== BEST MODEL ===",
        "total_samples": best_model["total_samples"],
        "val_loss": best_model["val_loss"],
        "test_loss": best_model["test_loss"],
        "log_mae": best_model["log_mae"],
        "log_r2": best_model["log_r2"],
        "linear_mae": best_model["linear_mae"],
        "linear_mse": best_model["linear_mse"],
    }])
    
    # Append summary row
    df_with_summary = pd.concat([df, summary_row], ignore_index=True)
    
    # Save updated CSV
    df_with_summary.to_csv(comp_path, index=False)
    
    print("="*80)
    print("ADDED BEST MODEL SUMMARY")
    print("="*80)
    print(f"Best Model: {best_model['dataset']}")
    print(f"  Test Loss: {best_model['test_loss']:.6f}")
    print(f"  Log MAE: {best_model['log_mae']:.6f}" if pd.notna(best_model['log_mae']) else "  Log MAE: N/A")
    print(f"  Log R²: {best_model['log_r2']:.4f}" if pd.notna(best_model['log_r2']) else "  Log R²: N/A")
    print(f"  Dataset Size: {best_model['total_samples']:,} samples")
    print()
    print(f"✅ Updated {comp_path}")


if __name__ == "__main__":
    main()

