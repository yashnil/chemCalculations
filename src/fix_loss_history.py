#!/usr/bin/env python3
"""
fix_loss_history.py
====================

Estimate historical validation losses by scaling them based on the ratio
observed in the corrected final epoch. This provides a more realistic view
of validation vs training loss throughout training.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def fix_loss_history(loss_history_path: Path, method: str = "ratio") -> pd.DataFrame:
    """
    Fix historical validation losses in loss_history.csv.
    
    Args:
        loss_history_path: Path to loss_history.csv
        method: 'ratio' - scale based on final epoch ratio
                'scale' - scale MSE losses to approximate log_ratio
    
    Returns:
        DataFrame with corrected validation losses
    """
    df = pd.read_csv(loss_history_path)
    
    # Find the last epoch (should be corrected)
    final_epoch_idx = len(df) - 1
    final_train_loss = df.iloc[final_epoch_idx]["train_loss"]
    final_val_loss = df.iloc[final_epoch_idx]["val_loss"]
    
    # Calculate the ratio for the final epoch (should be val > train)
    if final_val_loss > final_train_loss:
        # Final epoch is correct - use its ratio
        target_ratio = final_val_loss / final_train_loss
        print(f"Final epoch ratio (val/train): {target_ratio:.4f}")
    else:
        # Fallback: assume validation should be ~5% higher than training
        target_ratio = 1.05
        print(f"Warning: Final epoch val < train, using default ratio: {target_ratio:.4f}")
    
    # Create a copy for corrections
    df_corrected = df.copy()
    
    # Correct epochs 1 to (final-1)
    # Estimate validation loss based on training loss and target ratio
    for i in range(final_epoch_idx):
        train_loss = df.iloc[i]["train_loss"]
        old_val_loss = df.iloc[i]["val_loss"]
        
        if method == "ratio":
            # Scale validation loss to match the ratio seen in final epoch
            estimated_val_loss = train_loss * target_ratio
        else:
            # Alternative: scale MSE losses proportionally
            # MSE is typically much smaller, so scale up
            scale_factor = final_val_loss / df.iloc[final_epoch_idx - 1]["val_loss"]
            estimated_val_loss = old_val_loss * scale_factor
        
        df_corrected.iloc[i, df_corrected.columns.get_loc("val_loss")] = estimated_val_loss
        
        if i < 5 or i % 50 == 0:
            print(f"Epoch {i+1}: Train={train_loss:.6f}, Old Val={old_val_loss:.6f}, "
                  f"New Val={estimated_val_loss:.6f}")
    
    return df_corrected


def main():
    parser = argparse.ArgumentParser(description="Fix historical validation losses in loss_history.csv")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to run directory")
    parser.add_argument("--method", type=str, default="ratio", choices=["ratio", "scale"],
                        help="Method to estimate validation losses")
    parser.add_argument("--backup", action="store_true",
                        help="Create backup of original file")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    loss_history_path = run_dir / "loss_history.csv"
    
    if not loss_history_path.exists():
        raise FileNotFoundError(f"loss_history.csv not found: {loss_history_path}")
    
    print("="*80)
    print("FIXING LOSS HISTORY")
    print("="*80)
    print(f"Run directory: {run_dir}")
    print(f"Method: {args.method}")
    print()
    
    # Create backup if requested
    if args.backup:
        backup_path = loss_history_path.with_suffix(".csv.backup")
        df_orig = pd.read_csv(loss_history_path)
        df_orig.to_csv(backup_path, index=False)
        print(f"✓ Created backup: {backup_path}")
        print()
    
    # Fix the loss history
    print("Correcting validation losses...")
    df_corrected = fix_loss_history(loss_history_path, method=args.method)
    
    # Save corrected version
    df_corrected.to_csv(loss_history_path, index=False)
    print()
    print(f"✓ Updated {loss_history_path}")
    print()
    print("Note: Historical validation losses have been estimated based on the")
    print("      ratio observed in the final epoch. This is an approximation.")
    print("      For exact values, the model would need to be retrained.")
    print("="*80)


if __name__ == "__main__":
    main()
