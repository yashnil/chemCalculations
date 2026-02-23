#!/usr/bin/env python3
"""
check_progress.py
=================

Quick script to check progress of 3360K, 3680K, 4000K pipeline.
"""

import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "datasets"
RUNS_DIR = BASE_DIR / "results" / "runs"
JOBS_DIR = BASE_DIR / "results" / "fastchem_jobs"

DATASET_SIZES = {
    "x3360": 3360000,
    "x3680": 3680000,
    "x4000": 4000000,
}


def check_dataset(tag: str):
    """Check dataset status."""
    csv_path = DATA_DIR / f"all_gas_fastchem_{tag}.csv"
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        try:
            with open(csv_path, 'r') as f:
                lines = sum(1 for _ in f)
            return f"✓ EXISTS ({size_mb:.1f} MB, {lines:,} lines)"
        except:
            return f"✓ EXISTS ({size_mb:.1f} MB)"
    return "✗ MISSING"


def check_fastchem_jobs(tag: str):
    """Check FastChem job status."""
    jobs_root = JOBS_DIR / f"fastchem_jobs_{tag}"
    if not jobs_root.exists():
        return "✗ NOT PREPARED"
    
    job_dirs = sorted([d for d in jobs_root.iterdir() if d.is_dir() and d.name.startswith("job_")])
    if not job_dirs:
        return "✗ NO JOBS"
    
    total_jobs = len(job_dirs)
    complete_jobs = 0
    for job_dir in job_dirs:
        result_file = job_dir / "results" / "gas_species.csv"
        if result_file.exists() and result_file.stat().st_size > 0:
            complete_jobs += 1
    
    pct = (complete_jobs / total_jobs * 100) if total_jobs > 0 else 0
    return f"📊 {complete_jobs}/{total_jobs} jobs ({pct:.1f}%)"


def check_model(tag: str, size: int):
    """Check model training status."""
    run_dir = RUNS_DIR / f"runs_autoencoder_x{size}_optimal_retrained"
    
    if not run_dir.exists():
        return "✗ NOT STARTED"
    
    summary_path = run_dir / "summary.json"
    loss_history_path = run_dir / "loss_history.csv"
    
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            test_loss = summary.get("test_loss", "N/A")
            val_loss = summary.get("val_loss", "N/A")
            return f"✓ COMPLETE (test_loss={test_loss:.6f}, val_loss={val_loss:.6f})"
        except:
            return "✓ COMPLETE (summary exists)"
    
    if loss_history_path.exists():
        try:
            df = pd.read_csv(loss_history_path)
            epochs = len(df)
            if epochs > 0:
                latest_val = df.iloc[-1].get("val_loss", "N/A")
                latest_train = df.iloc[-1].get("train_loss", "N/A")
                return f"🔄 TRAINING ({epochs}/200 epochs, val_loss={latest_val:.6f})"
        except:
            return "🔄 TRAINING (loss_history exists)"
    
    return "⚠️  INCOMPLETE (directory exists but no progress files)"


def main():
    print("="*80)
    print("PROGRESS CHECK: 3360K, 3680K, 4000K PIPELINE")
    print("="*80)
    print()
    
    for tag, total_samples in DATASET_SIZES.items():
        size = int(tag.replace("x", ""))
        print(f"{tag} ({total_samples:,} samples):")
        print(f"  Dataset:     {check_dataset(tag)}")
        print(f"  FastChem:     {check_fastchem_jobs(tag)}")
        print(f"  Model:        {check_model(tag, size)}")
        print()
    
    print("="*80)
    print("QUICK STATUS SUMMARY")
    print("="*80)
    
    # Count completions
    datasets_done = sum(1 for tag in DATASET_SIZES.keys() 
                       if (DATA_DIR / f"all_gas_fastchem_{tag}.csv").exists())
    models_done = sum(1 for tag, size in [(t, int(t.replace("x", ""))) for t in DATASET_SIZES.keys()]
                     if (RUNS_DIR / f"runs_autoencoder_x{size}_optimal_retrained" / "summary.json").exists())
    
    print(f"Datasets complete: {datasets_done}/{len(DATASET_SIZES)}")
    print(f"Models complete:   {models_done}/{len(DATASET_SIZES)}")
    print()
    
    # Show currently training models
    training = []
    for tag, size in [(t, int(t.replace("x", ""))) for t in DATASET_SIZES.keys()]:
        run_dir = RUNS_DIR / f"runs_autoencoder_x{size}_optimal_retrained"
        loss_history = run_dir / "loss_history.csv"
        summary = run_dir / "summary.json"
        
        if loss_history.exists() and not summary.exists():
            try:
                df = pd.read_csv(loss_history)
                epochs = len(df)
                if epochs > 0:
                    latest_val = df.iloc[-1].get("val_loss", "N/A")
                    training.append((tag, epochs, latest_val))
            except:
                pass
    
    if training:
        print("Currently training:")
        for tag, epochs, val_loss in training:
            print(f"  {tag}: Epoch {epochs}/200, val_loss={val_loss:.6f}")
    else:
        print("No models currently training")
    
    print()


if __name__ == "__main__":
    main()
