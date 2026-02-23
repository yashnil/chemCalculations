#!/usr/bin/env python3
"""
compute_test_metrics.py
========================

Compute test set log_mae and log_r2 for consistent architecture runs.
This script loads each model and evaluates on the test set.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "results" / "runs"

CONSISTENT_RUNS = [
    ("x160_static_32_consistent", 160000),
    ("x240_static_32_consistent", 240000),
    ("x480_static_32_consistent", 480000),
    ("x640_static_32_consistent", 640000),
]

CLIP = 1e-10


def compute_test_metrics(run_tag: str):
    """Compute test set log_mae and log_r2 for a model."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    best_model_path = run_dir / "best_model.py"
    summary_path = run_dir / "summary.json"
    
    if not best_model_path.exists() or not summary_path.exists():
        print(f"  ⚠️  Missing files for {run_tag}")
        return None, None
    
    # Load summary to get splits and dataset path
    summary = json.load(open(summary_path))
    splits = summary.get("splits", {})
    test_idx = np.array(splits.get("test_idx", []), dtype=int)
    
    if len(test_idx) == 0:
        print(f"  ⚠️  No test indices for {run_tag}")
        return None, None
    
    # Get dataset path
    size = run_tag.split('_')[0].replace('x', '')
    csv_path = BASE_DIR / "data" / "datasets" / f"all_gas_fastchem_x{size}.csv"
    
    if not csv_path.exists():
        print(f"  ⚠️  Dataset not found: {csv_path}")
        return None, None
    
    print(f"  Loading model and data for {run_tag}...")
    
    try:
        # Load model module
        import importlib.util
        spec = importlib.util.spec_from_file_location("best_model", best_model_path)
        if spec is None or spec.loader is None:
            print(f"  ⚠️  Could not load module")
            return None, None
        best_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(best_mod)
        
        # Load data
        df = pd.read_csv(csv_path)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        
        # Get target columns
        target_cols = summary.get("target_cols", [])
        if not target_cols:
            print(f"  ⚠️  No target columns in summary")
            return None, None
        
        # Normalize inputs
        X_test = best_mod.normalize_inputs(df_test)
        
        # Load model and run inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = best_mod.load_model(device=device)
        model.eval()
        
        with torch.no_grad():
            pred_scaled = best_mod.forward_autoencoder(model, X_test)
            pred_scaled = pred_scaled.cpu().numpy()
        
        # Denormalize predictions
        y_pred = best_mod.denormalize_targets(pred_scaled)
        y_true = df_test[target_cols].to_numpy(dtype=np.float64, copy=True)
        
        # Clip to avoid numerical issues
        y_true = np.clip(y_true, 0, None)
        y_pred = np.clip(y_pred, 0, None)
        
        # Compute log-space metrics
        y_true_log = np.log10(y_true + CLIP)
        y_pred_log = np.log10(y_pred + CLIP)
        
        log_mae = mean_absolute_error(y_true_log, y_pred_log)
        log_r2 = r2_score(y_true_log, y_pred_log, multioutput="variance_weighted")
        
        print(f"    Test log_mae: {log_mae:.6f}")
        print(f"    Test log_r2:  {log_r2:.6f}")
        
        return log_mae, log_r2
        
    except Exception as e:
        print(f"  ⚠️  Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def update_summary_json(run_tag: str, log_mae: float, log_r2: float):
    """Update summary.json with test metrics."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    summary_path = run_dir / "summary.json"
    
    if not summary_path.exists():
        return False
    
    summary = json.load(open(summary_path))
    summary["test_log_mae"] = log_mae
    summary["test_log_r2"] = log_r2
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return True


def main():
    print("="*80)
    print("COMPUTING TEST SET METRICS FOR CONSISTENT RUNS")
    print("="*80)
    
    results = {}
    
    for run_tag, dataset_size in CONSISTENT_RUNS:
        print(f"\n{run_tag}:")
        log_mae, log_r2 = compute_test_metrics(run_tag)
        
        if log_mae is not None and log_r2 is not None:
            results[run_tag] = {"log_mae": log_mae, "log_r2": log_r2}
            # Update summary.json
            if update_summary_json(run_tag, log_mae, log_r2):
                print(f"  ✅ Updated summary.json")
        else:
            print(f"  ⚠️  Failed to compute metrics")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for run_tag, metrics in results.items():
        print(f"{run_tag}: log_mae={metrics['log_mae']:.6f}, log_r2={metrics['log_r2']:.6f}")
    
    print("\nNext step: Update comparison_metrics.csv")
    print("Run: python scripts/extract_and_update_metrics.py")


if __name__ == "__main__":
    main()
