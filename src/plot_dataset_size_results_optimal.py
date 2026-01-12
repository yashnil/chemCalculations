#!/usr/bin/env python3
"""
plot_dataset_size_results_optimal.py
=====================================

Generate plot from completed dataset size test results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def collect_results(datasets: list[str]) -> list[dict]:
    """Collect results from completed training runs."""
    results = []
    
    for dataset in datasets:
        run_dir = BASE_DIR / f"runs_autoencoder_optimal_{dataset}"
        summary_path = run_dir / "summary.json"
        
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            
            # Try to get log_mae and log_r2 from diagnostics
            log_mae = None
            log_r2 = None
            diag_path = run_dir / "diagnostics" / "global_metrics.txt"
            if diag_path.exists():
                for line in diag_path.read_text().splitlines():
                    if "Log MAE" in line and ":" in line:
                        try:
                            log_mae = float(line.split(":")[1].strip().split()[0].replace(",", ""))
                        except:
                            pass
                    if "Log R" in line and ":" in line:
                        try:
                            log_r2 = float(line.split(":")[1].strip().split()[0].replace(",", ""))
                        except:
                            pass
            
            total_samples = (
                summary.get("train_samples", 0) + 
                summary.get("val_samples", 0) + 
                summary.get("test_samples", 0)
            )
            
            results.append({
                "dataset": dataset,
                "total_samples": total_samples,
                "test_loss": summary.get("test_loss"),
                "val_loss": summary.get("val_loss"),
                "test_mae_linear": summary.get("test_mae_linear"),
                "log_mae": log_mae,
                "log_r2": log_r2,
            })
            print(f"✅ {dataset}: test_loss={summary.get('test_loss'):.6f}, log_mae={log_mae}")
        else:
            print(f"⏳ {dataset}: Still training or not started")
    
    return results


def plot_results(results: list[dict], output_path: Path):
    """Plot dataset size vs performance metrics."""
    if not results:
        print("❌ No results to plot!")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values("total_samples")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Test Loss vs Dataset Size
    axes[0, 0].plot(df["total_samples"] / 1000, df["test_loss"], 
                   marker="o", linewidth=2, markersize=8, color="steelblue")
    axes[0, 0].set_xlabel("Dataset Size (×1000 samples)", fontsize=12)
    axes[0, 0].set_ylabel("Test Loss (Normalized)", fontsize=12)
    axes[0, 0].set_title("Test Loss vs Dataset Size (Optimal Hyperparameters)", fontsize=14, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Log MAE vs Dataset Size
    if "log_mae" in df.columns and df["log_mae"].notna().any():
        valid = df["log_mae"].notna()
        axes[0, 1].plot(df[valid]["total_samples"] / 1000, df[valid]["log_mae"], 
                        marker="s", linewidth=2, markersize=8, color="coral")
        axes[0, 1].set_xlabel("Dataset Size (×1000 samples)", fontsize=12)
        axes[0, 1].set_ylabel("Log MAE", fontsize=12)
        axes[0, 1].set_title("Log MAE vs Dataset Size", fontsize=14, fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Log R² vs Dataset Size
    if "log_r2" in df.columns and df["log_r2"].notna().any():
        valid = df["log_r2"].notna()
        axes[1, 0].plot(df[valid]["total_samples"] / 1000, df[valid]["log_r2"], 
                       marker="^", linewidth=2, markersize=8, color="green")
        axes[1, 0].set_xlabel("Dataset Size (×1000 samples)", fontsize=12)
        axes[1, 0].set_ylabel("Log R²", fontsize=12)
        axes[1, 0].set_title("Log R² vs Dataset Size", fontsize=14, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0.95, 1.0])
    
    # Plot 4: Validation Loss vs Dataset Size
    axes[1, 1].plot(df["total_samples"] / 1000, df["val_loss"], 
                   marker="d", linewidth=2, markersize=8, color="purple")
    axes[1, 1].set_xlabel("Dataset Size (×1000 samples)", fontsize=12)
    axes[1, 1].set_ylabel("Validation Loss (Normalized)", fontsize=12)
    axes[1, 1].set_title("Validation Loss vs Dataset Size", fontsize=14, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Plot saved to {output_path}")
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    if len(df) > 0:
        best = df.loc[df["test_loss"].idxmin()]
        print(f"\n🏆 Best model: dataset={best['dataset']}, test_loss={best['test_loss']:.6f}")


def main():
    datasets = ["x32", "x48", "x64", "x80", "x96", "x112", "x128", "x144", "x160", "x176"]
    output_path = BASE_DIR / "dataset_size_study_optimal.png"
    
    print("="*80)
    print("DATASET SIZE STUDY - PLOT GENERATION")
    print("="*80)
    print(f"Checking for results: {datasets}")
    print("="*80)
    
    results = collect_results(datasets)
    
    if results:
        plot_results(results, output_path)
    else:
        print("\n❌ No completed results found. Please wait for training to complete.")


if __name__ == "__main__":
    main()



