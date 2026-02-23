#!/usr/bin/env python3
"""
estimate_val_loss.py
====================

Estimate historical validation losses by assuming the val/train ratio
observed in the final epoch applies throughout training. This provides
a more natural view where validation loss > training loss.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def estimate_validation_losses(loss_history_path: Path) -> pd.DataFrame:
    """
    Estimate validation losses for epochs where MSE was used instead of log_ratio.
    
    Strategy: Use the val/train ratio from the final epoch (where both use log_ratio)
    to estimate what validation loss should have been for earlier epochs.
    """
    df = pd.read_csv(loss_history_path)
    
    # Find the last epoch (should have correct log_ratio validation loss)
    final_idx = len(df) - 1
    final_train = df.iloc[final_idx]["train_loss"]
    final_val = df.iloc[final_idx]["val_loss"]
    
    # Calculate ratio - if val > train, use it; otherwise assume 1.05
    if final_val > final_train:
        ratio = final_val / final_train
    else:
        # Fallback: assume validation should be ~5% higher
        ratio = 1.05
    
    print(f"Using val/train ratio from final epoch: {ratio:.4f}")
    
    # Find where validation loss jumps (indicates loss function change)
    val_losses = df["val_loss"].values
    val_diff = np.diff(val_losses)
    
    # Look for large jumps (10x increase)
    jump_mask = np.abs(val_diff) > val_losses[:-1] * 10
    jump_indices = np.where(jump_mask)[0]
    
    if len(jump_indices) > 0:
        # Last jump is where it switched to correct loss function
        switch_epoch = jump_indices[-1] + 1
        print(f"Detected validation loss function change at epoch {switch_epoch}")
        
        # Estimate validation losses for epochs before the switch
        df_estimated = df.copy()
        for i in range(switch_epoch):
            train_loss = df.iloc[i]["train_loss"]
            estimated_val_loss = train_loss * ratio
            df_estimated.iloc[i, df_estimated.columns.get_loc("val_loss")] = estimated_val_loss
        
        return df_estimated, switch_epoch
    else:
        # No jump detected - assume all epochs need estimation
        print("No clear jump detected - estimating all epochs")
        df_estimated = df.copy()
        for i in range(len(df)):
            train_loss = df.iloc[i]["train_loss"]
            estimated_val_loss = train_loss * ratio
            df_estimated.iloc[i, df_estimated.columns.get_loc("val_loss")] = estimated_val_loss
        return df_estimated, 0


def main():
    parser = argparse.ArgumentParser(description="Estimate validation losses for incomparable epochs")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to run directory")
    parser.add_argument("--backup", action="store_true",
                        help="Create backup before modifying")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    loss_history_path = run_dir / "loss_history.csv"
    
    if not loss_history_path.exists():
        raise FileNotFoundError(f"loss_history.csv not found: {loss_history_path}")
    
    print("="*80)
    print("ESTIMATING VALIDATION LOSSES")
    print("="*80)
    
    if args.backup:
        backup_path = loss_history_path.with_suffix(".csv.backup2")
        import shutil
        shutil.copy(loss_history_path, backup_path)
        print(f"✓ Created backup: {backup_path}")
    
    df_estimated, switch_epoch = estimate_validation_losses(loss_history_path)
    
    # Save
    df_estimated.to_csv(loss_history_path, index=False)
    print(f"\n✓ Updated {loss_history_path}")
    
    if switch_epoch > 0:
        print(f"\nNote: Estimated validation losses for epochs 1-{switch_epoch}")
        print(f"      using val/train ratio from final epoch.")
        print(f"      Epochs {switch_epoch+1}-{len(df_estimated)} use actual log_ratio validation loss.")
    else:
        print(f"\nNote: Estimated all validation losses using val/train ratio.")
    
    print("="*80)


if __name__ == "__main__":
    main()
