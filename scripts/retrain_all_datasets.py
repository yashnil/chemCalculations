#!/usr/bin/env python3
"""
Retrain all datasets (x32, x48, x64, x80, x96, x112, x128, x144, x160, x176) with locked target species.

This script assumes datasets already exist and just retrains the models.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

DATASETS = ["x32", "x48", "x64", "x80", "x96", "x112", "x128", "x144", "x160", "x176"]


def run_cmd(cmd: list[str], env: dict | None = None) -> None:
    """Run a command and raise on failure."""
    pretty = " ".join(cmd)
    print(f"\n[cmd] {pretty}")
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit={result.returncode}): {pretty}")


def clean_runs_dir() -> None:
    """Remove stale runs_autoencoder directory."""
    run_dir = BASE_DIR / "runs_autoencoder"
    if run_dir.exists():
        print(f"[cleanup] removing stale {run_dir}")
        shutil.rmtree(run_dir)


def archive_runs(tag: str) -> None:
    """Archive runs_autoencoder to runs_autoencoder_{tag}."""
    run_dir = BASE_DIR / "runs_autoencoder"
    if not run_dir.exists():
        raise RuntimeError(f"Expected runs_autoencoder directory not found after training for {tag}")

    archive_dir = BASE_DIR / f"runs_autoencoder_{tag}"
    if archive_dir.exists():
        print(f"[archive] removing previous archive {archive_dir}")
        shutil.rmtree(archive_dir)

    shutil.move(str(run_dir), str(archive_dir))
    print(f"[archive] archived results → {archive_dir}")


def train_dataset(tag: str) -> None:
    """Train a single dataset."""
    csv_path = BASE_DIR / "datasets" / f"all_gas_fastchem_{tag}.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    print("\n" + "=" * 80)
    print(f"Training {tag} dataset")
    print("=" * 80)
    print(f"CSV: {csv_path}")

    # Clean up old run directory
    clean_runs_dir()

    # Train
    train_env = os.environ.copy()
    train_env["CSV_PATH"] = str(csv_path)
    run_cmd([PYTHON, str(BASE_DIR / "train_autoencoder.py")], env=train_env)

    # Diagnostics
    diagnostics_env = os.environ.copy()
    diagnostics_env["CSV_PATH"] = str(csv_path)
    diagnostics_env["BEST_MODULE"] = str(BASE_DIR / "runs_autoencoder" / "best_model.py")
    diagnostics_env["OUT_DIR"] = str(BASE_DIR / "runs_autoencoder" / "diagnostics")
    run_cmd([PYTHON, str(BASE_DIR / "diagnostics.py")], env=diagnostics_env)

    # Plot
    plot_env = diagnostics_env.copy()
    plot_env["OUT_PNG"] = str(BASE_DIR / "runs_autoencoder" / "pred_vs_true_test.png")
    run_cmd([PYTHON, str(BASE_DIR / "plot.py")], env=plot_env)

    # Archive
    archive_runs(tag)


def main() -> None:
    """Retrain all datasets."""
    print("=" * 80)
    print("Retraining all datasets with locked targets")
    print("=" * 80)

    for tag in DATASETS:
        train_dataset(tag)

    print("\n" + "=" * 80)
    print("All datasets retrained successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run: python make_comparison_metrics.py")
    print("2. Run: python plot_resolution_study.py")
    print()


if __name__ == "__main__":
    main()

