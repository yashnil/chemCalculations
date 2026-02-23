#!/usr/bin/env python3
"""
sync_validation_loss.py
========================

Re-evaluate validation loss using the correct loss function (matching training)
and update loss_history.csv and summary.json.

This fixes the issue where validation loss was computed with MSE while training
used log_ratio (or other) loss function.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# Import from train_autoencoder
sys.path.insert(0, str(Path(__file__).parent))
from train_autoencoder import (
    LogRatioLoss,
    WeightedHuber,
    AutoencoderDataset,
    evaluate,
    normalize_inputs_df,
    resolve_input_columns,
    resolve_target_columns,
    scale_targets_linear_to_train,
    compute_target_weights,
    FlowMapAutoencoder,
    TARGET_LOG_SCALE,
    SEED,
    TRAIN_FRAC,
    VAL_FRAC,
    TEST_FRAC,
    LATENT_DIM,
    ENCODER_HIDDEN,
    DYNAMICS_HIDDEN,
    DECODER_HIDDEN,
    ACTIVATION,
    DROPOUT,
    BATCH_SIZE,
    CSV_PATH,
)
from sklearn.model_selection import train_test_split


def load_checkpoint(run_dir: Path, device: torch.device):
    """Load model checkpoint and config."""
    checkpoint_path = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    with open(summary_path) as f:
        summary = json.load(f)
    
    return checkpoint, summary


def reconstruct_model(checkpoint: dict, summary: dict, device: torch.device):
    """Reconstruct model from checkpoint."""
    config = checkpoint["config"]
    hyperparams = config["hyperparams"]
    
    state_dim = len(config["target_cols"])
    global_dim = len(config["input_cols"])
    
    model = FlowMapAutoencoder(
        state_dim_in=state_dim,
        state_dim_out=state_dim,
        global_dim=global_dim,
        latent_dim=hyperparams.get("latent_dim", LATENT_DIM),
        encoder_hidden=hyperparams.get("encoder_hidden", ENCODER_HIDDEN),
        dynamics_hidden=hyperparams.get("dynamics_hidden", DYNAMICS_HIDDEN),
        decoder_hidden=hyperparams.get("decoder_hidden", DECODER_HIDDEN),
        activation_name=hyperparams.get("activation", ACTIVATION),
        dropout=hyperparams.get("dropout", DROPOUT),
        predict_delta=True,
        predict_delta_log_phys=False,
        softmax_head=False,
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Sync validation loss with correct loss function")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to run directory (e.g., results/runs/runs_autoencoder_x160_static_32)")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to CSV dataset (default: from train_autoencoder)")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load checkpoint and summary
    print(f"\n1. Loading checkpoint from {run_dir}...")
    checkpoint, summary = load_checkpoint(run_dir, device)
    loss_type = summary.get("loss_type", "log_ratio")
    print(f"   Loss type: {loss_type}")
    
    # Reconstruct model
    print("\n2. Reconstructing model...")
    model, config = reconstruct_model(checkpoint, summary, device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load data
    csv_path = args.csv_path or CSV_PATH
    print(f"\n3. Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"   Loaded: {len(df)} rows")
    
    # Reconstruct splits
    splits = config.get("splits", {})
    if "val_idx" in splits:
        # Use saved splits
        idx_train = np.array(splits["train_idx"], dtype=int)
        idx_val = np.array(splits["val_idx"], dtype=int)
        idx_test = np.array(splits["test_idx"], dtype=int)
        print(f"   Using saved splits: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    else:
        # Reconstruct splits (may not match exactly due to randomness)
        print("   Warning: No saved splits found, reconstructing...")
        input_cols = resolve_input_columns(df)
        target_cols = resolve_target_columns(df, input_cols)
        indices = np.arange(len(df))
        X_train, X_tmp, y_train, y_tmp, idx_train, idx_tmp = train_test_split(
            df[input_cols], df[target_cols], indices, 
            train_size=TRAIN_FRAC, random_state=SEED, shuffle=True
        )
        val_ratio = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
        X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
            X_tmp, y_tmp, idx_tmp, train_size=val_ratio, random_state=SEED + 1, shuffle=True
        )
        print(f"   Reconstructed splits: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    
    # Prepare validation data
    print("\n4. Preparing validation dataset...")
    input_cols = config["input_cols"]
    target_cols = config["target_cols"]
    
    df_val = df.iloc[idx_val].reset_index(drop=True)
    X_val = normalize_inputs_df(df_val, input_cols)
    y_linear_val = df_val[target_cols].to_numpy(dtype=np.float64)
    y_scaled_val = scale_targets_linear_to_train(y_linear_val)
    
    val_ds = AutoencoderDataset(X_val, y_scaled_val)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"   Validation batches: {len(val_loader)}")
    
    # Create correct loss function
    print(f"\n5. Creating {loss_type} loss function...")
    weights_array = np.array(config.get("weights", []))
    if len(weights_array) == 0:
        # Recompute weights if not saved
        y_linear_all = df[target_cols].to_numpy(dtype=np.float64)
        weights_array = compute_target_weights(y_linear_all)
    
    weights_tensor = torch.as_tensor(weights_array, dtype=torch.float32, device=device)
    
    if loss_type == "log_ratio":
        criterion = LogRatioLoss(weights=weights_tensor, target_log_scale=TARGET_LOG_SCALE)
    elif loss_type == "huber":
        criterion = WeightedHuber(delta=0.02, weights=weights_tensor)
    else:  # mse
        criterion = nn.MSELoss()
    
    # Re-evaluate validation set
    print("\n6. Re-evaluating validation set with correct loss function...")
    val_res = evaluate(model, val_loader, device, criterion)
    print(f"   Corrected validation loss: {val_res.loss:.6f}")
    print(f"   (Previous validation loss: {summary.get('val_loss', 'N/A')})")
    
    # Update summary.json
    print("\n7. Updating summary.json...")
    summary["val_loss"] = val_res.loss
    summary["val_mse"] = val_res.mse
    summary["val_mae"] = val_res.mae
    summary["val_log_mae"] = val_res.log_mae
    
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   ✓ Updated {summary_path}")
    
    # Update loss_history.csv - correct the final validation loss
    print("\n8. Updating loss_history.csv...")
    loss_history_path = run_dir / "loss_history.csv"
    if loss_history_path.exists():
        loss_df = pd.read_csv(loss_history_path)
        # Update the last row's validation loss with the corrected value
        if len(loss_df) > 0:
            # Update the final epoch's validation loss
            loss_df.iloc[-1, loss_df.columns.get_loc("val_loss")] = val_res.loss
            loss_df.to_csv(loss_history_path, index=False)
            print(f"   ✓ Updated final validation loss in {loss_history_path}")
            print(f"   Note: Historical validation losses (epochs 1-{len(loss_df)-1}) were computed with MSE.")
            print(f"   Only the final epoch's validation loss has been corrected.")
        else:
            print(f"   Warning: loss_history.csv is empty")
    else:
        print(f"   Warning: loss_history.csv not found")
    
    print("\n9. Note:")
    print("   Historical validation losses in loss_history.csv (except final epoch)")
    print("   were computed with MSE loss, not the training loss function.")
    print("   To fully sync all epochs, you would need to retrain the model.")
    
    print("\n" + "="*80)
    print("✅ Validation loss synced successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
