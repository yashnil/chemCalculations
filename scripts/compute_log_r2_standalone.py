#!/usr/bin/env python3
"""
compute_log_r2_standalone.py
=============================

Compute log_r2 for existing optimal_retrained models.
Run this OUTSIDE the sandbox (directly in terminal) to avoid torch permission issues.

Usage:
    python scripts/compute_log_r2_standalone.py
"""

import json
import sys
from pathlib import Path

# Set up Python path BEFORE importing torch or other modules that might import autoencoder_model
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

RUNS_DIR = BASE_DIR / "results" / "runs"

OPTIMAL_RETRAINED_RUNS = [
    ("x160_optimal_retrained", 160000),
    ("x320_optimal_retrained", 320000),
    ("x480_optimal_retrained", 480000),
    ("x640_optimal_retrained", 640000),
    ("x800_optimal_retrained", 800000),
]


def compute_log_r2(run_tag: str):
    """Compute log_r2 for a model by re-evaluating on test set."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    best_model_path = run_dir / "best_model.py"
    summary_path = run_dir / "summary.json"
    
    if not best_model_path.exists() or not summary_path.exists():
        print(f"  ⚠️  Missing files for {run_tag}")
        return None, None
    
    # Load summary
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
    
    print(f"  Computing log_r2 for {run_tag}...")
    
    try:
        # Add src/ to Python path so autoencoder_model can be imported
        import sys
        src_dir = BASE_DIR / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        
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
        
        # Run inference
        all_y_log = []
        all_pred_log = []
        
        with torch.no_grad():
            # Process in batches
            batch_size = 512
            for i in range(0, len(X_test), batch_size):
                X_batch = X_test[i:i+batch_size]
                X_tensor = torch.FloatTensor(X_batch).to(device)
                
                # Get predictions
                pred_scaled = best_mod.forward_autoencoder(model, X_tensor)
                pred_scaled = pred_scaled.cpu()
                
                # Convert to linear space
                y_true_lin = df_test[target_cols].iloc[i:i+batch_size].values
                y_pred_lin = best_mod.denormalize_targets(pred_scaled.numpy())
                
                # Clip to avoid numerical issues
                y_true_lin = np.clip(y_true_lin, 1e-30, None)
                y_pred_lin = np.clip(y_pred_lin, 1e-30, None)
                
                # Convert to log space
                y_true_log = np.log10(y_true_lin)
                y_pred_log = np.log10(y_pred_lin)
                
                all_y_log.append(y_true_log)
                all_pred_log.append(y_pred_log)
        
        # Compute log R²
        y_log_all = np.concatenate(all_y_log, axis=0)
        pred_log_all = np.concatenate(all_pred_log, axis=0)
        
        # Flatten for R² computation (treat all species together)
        log_r2 = float(r2_score(y_log_all.flatten(), pred_log_all.flatten()))
        
        # Also compute log_mae for verification
        log_mae = float(np.mean(np.abs(y_log_all.flatten() - pred_log_all.flatten())))
        
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
    print("COMPUTING LOG R² FOR OPTIMAL_RETRAINED MODELS")
    print("="*80)
    print("This script computes test set log_mae and log_r2 for all models")
    print()
    
    results = {}
    
    for run_tag, dataset_size in OPTIMAL_RETRAINED_RUNS:
        print(f"{run_tag}:")
        log_mae, log_r2 = compute_log_r2(run_tag)
        
        if log_mae is not None and log_r2 is not None:
            results[run_tag] = {"log_mae": log_mae, "log_r2": log_r2}
            # Update summary.json
            if update_summary_json(run_tag, log_mae, log_r2):
                print(f"  ✅ Updated summary.json")
        else:
            print(f"  ⚠️  Failed to compute metrics")
        print()
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    for run_tag, metrics in results.items():
        print(f"{run_tag}: log_mae={metrics['log_mae']:.6f}, log_r2={metrics['log_r2']:.6f}")
    
    print("\n" + "="*80)
    print("Next step: Update comparison_metrics.csv and regenerate plots")
    print("Run: python scripts/update_plots_for_optimal_retrained.py")
    print("="*80)


if __name__ == "__main__":
    main()
