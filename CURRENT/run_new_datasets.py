#!/usr/bin/env python3
"""
Automate the full 32K / 48K dataset pipeline:

1. Prepare FastChem job shards (empirical resampling from the reference CSV)
2. Run FastChem across all shards
3. Merge shard outputs into a single CSV
4. Apply the 20% low-temperature filter and downsample to the target size
5. Train the FlowMap Autoencoder, run diagnostics, and generate plots
6. Archive each run under `runs_autoencoder_<tag>`

Usage:
    python run_new_datasets.py

Prerequisites:
    * `pyfastchem` must be installable/importable (run `pip install .` in your FastChem source)
    * Environment variables `FASTCHEM_LOGK` and `FASTCHEM_COND` must point to valid FastChem tables
    * Execute from `/Users/yashnilmohanty/Desktop/chemCalculations/UPDATED_VERS`
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASETS: List[Dict[str, int | str]] = [
    {
        "tag": "x32",
        "total_samples": 40_000,
        "retain": 32_000,
        "seed": 2032,
    },
    {
        "tag": "x48",
        "total_samples": 60_000,
        "retain": 48_000,
        "seed": 2048,
    },
    {
        "tag": "x64",
        "total_samples": 80_000,
        "retain": 64_000,
        "seed": 2056,
    },
    {
        "tag": "x80",
        "total_samples": 100_000,
        "retain": 80_000,
        "seed": 2080,
    },
]

REFERENCE_CSV = Path("/Users/yashnilmohanty/Desktop/chemCalculations/NEW_VERS/all_gas_v10_no_stripe_clean.csv")
DATASETS_DIR = BASE_DIR / "datasets"
PYTHON = sys.executable


def require_pyfastchem() -> None:
    try:
        import fastchem  # noqa: F401
    except ImportError:
        try:
            import pyfastchem  # noqa: F401
        except ImportError as exc:  # pragma: no cover - direct user feedback
            raise RuntimeError(
                "pyfastchem is not importable. Run `pip install pyfastchem` (or build the bindings from "
                "source) and re-run this script."
            ) from exc


def require_env_vars() -> None:
    missing = [var for var in ("FASTCHEM_LOGK", "FASTCHEM_COND") if not os.environ.get(var)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing environment variable(s): {joined}")


def run_cmd(cmd: List[str], env: dict | None = None) -> None:
    pretty = " ".join(cmd)
    print(f"\n[cmd] {pretty}")
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit={result.returncode}): {pretty}")


def filter_and_downsample(raw_csv: Path, filtered_csv: Path, retain: int, seed: int) -> None:
    print(f"[filter] loading {raw_csv}")
    df = pd.read_csv(raw_csv)
    if "T_K" not in df.columns:
        raise RuntimeError(f"'T_K' column missing from {raw_csv}")

    cutoff = df["T_K"].quantile(0.2)
    filtered = df[df["T_K"] >= cutoff].reset_index(drop=True)
    print(f"[filter] retained {len(filtered)} rows after 20% low-temperature filter (cutoff={cutoff:.3f} K)")

    if len(filtered) > retain:
        filtered = filtered.sample(n=retain, random_state=seed).reset_index(drop=True)
        print(f"[filter] downsampled to {retain} rows with seed {seed}")
    else:
        print(f"[filter] available rows ({len(filtered)}) <= target retain count ({retain}); skipping downsample")

    filtered.to_csv(filtered_csv, index=False)
    print(f"[filter] wrote {len(filtered)} rows → {filtered_csv}")


def ensure_clean_dataset(csv_path: Path) -> Path:
    print(f"[clean] scanning {csv_path} for NaN/Inf rows")
    df = pd.read_csv(csv_path)
    sanitized = df.replace([np.inf, -np.inf], np.nan)
    bad_mask = sanitized.isna().any(axis=1)
    dropped = int(bad_mask.sum())
    if dropped:
        cleaned = sanitized.loc[~bad_mask].reset_index(drop=True)
        cleaned_path = csv_path.with_name(csv_path.stem + "_clean.csv")
        cleaned.to_csv(cleaned_path, index=False)
        print(f"[clean] dropped {dropped} row(s); wrote cleaned dataset → {cleaned_path}")
        return cleaned_path
    print("[clean] no problematic rows detected")
    return csv_path


def archive_runs(tag: str) -> None:
    run_dir = BASE_DIR / "runs_autoencoder"
    if not run_dir.exists():
        raise RuntimeError(f"Expected runs_autoencoder directory not found after training for {tag}")

    archive_dir = BASE_DIR / f"runs_autoencoder_{tag}"
    if archive_dir.exists():
        print(f"[archive] removing previous archive {archive_dir}")
        shutil.rmtree(archive_dir)

    shutil.move(str(run_dir), str(archive_dir))
    print(f"[archive] archived results → {archive_dir}")


def clean_runs_dir() -> None:
    run_dir = BASE_DIR / "runs_autoencoder"
    if run_dir.exists():
        print(f"[cleanup] removing stale {run_dir}")
        shutil.rmtree(run_dir)


def process_dataset(cfg: Dict[str, int | str]) -> None:
    tag = cfg["tag"]
    total_samples = int(cfg["total_samples"])
    retain = int(cfg["retain"])
    seed = int(cfg["seed"])

    print(f"\n===== Processing dataset {tag} ({retain} rows target) =====")

    jobs_root = BASE_DIR / f"fastchem_jobs_{tag}"
    raw_csv = DATASETS_DIR / f"all_gas_fastchem_{tag}_raw.csv"
    final_csv = DATASETS_DIR / f"all_gas_fastchem_{tag}.csv"
    elements_path = os.environ.get("FASTCHEM_ELEM")
    if elements_path:
        elements_path = str(Path(elements_path).expanduser().resolve())
    else:
        inferred = (
            Path(os.environ["FASTCHEM_LOGK"])
            .expanduser()
            .resolve()
            .parent.parent
            / "element_abundances"
            / "asplund_2009.dat"
        )
        if not inferred.exists():
            raise RuntimeError(
                "Could not determine the FastChem element abundance file. "
                "Set FASTCHEM_ELEM to the appropriate path."
            )
        elements_path = str(inferred)

    # Step 1: prepare jobs
    prepare_cmd = [
        PYTHON,
        str(BASE_DIR / "data_generation" / "prepare_fastchem_jobs.py"),
        "--total-samples",
        str(total_samples),
        "--shard-size",
        "2000",
        "--output-root",
        str(jobs_root),
        "--reference-csv",
        str(REFERENCE_CSV),
        "--seed",
        str(seed),
    ]
    run_cmd(prepare_cmd)

    # Step 2: run FastChem
    run_all_cmd = [
        PYTHON,
        str(BASE_DIR / "data_generation" / "run_fastchem_all.py"),
        "--jobs-root",
        str(jobs_root),
        "--logk",
        os.environ["FASTCHEM_LOGK"],
        "--logk-cond",
        os.environ["FASTCHEM_COND"],
        "--chunksize",
        "128",
        "--condensates",
        "false",
    ]
    if elements_path:
        run_all_cmd += ["--element-abundances", elements_path]
    run_cmd(run_all_cmd)

    # Step 3: merge outputs
    merge_cmd = [
        PYTHON,
        str(BASE_DIR / "data_generation" / "merge_fastchem_outputs.py"),
        "--jobs-root",
        str(jobs_root),
        "--reference-csv",
        str(REFERENCE_CSV),
        "--output-csv",
        str(raw_csv),
    ]
    run_cmd(merge_cmd)

    # Step 4: apply filter & downsample
    filter_and_downsample(raw_csv, final_csv, retain=retain, seed=seed)
    final_csv = ensure_clean_dataset(final_csv)

    # Step 5: train autoencoder
    clean_runs_dir()
    train_env = os.environ.copy()
    train_env["CSV_PATH"] = str(final_csv)
    train_cmd = [
        PYTHON,
        str(BASE_DIR / "train_autoencoder.py"),
    ]
    run_cmd(train_cmd, env=train_env)

    # Step 6: diagnostics
    diagnostics_env = os.environ.copy()
    diagnostics_env["CSV_PATH"] = str(final_csv)
    diagnostics_env["BEST_MODULE"] = str(BASE_DIR / "runs_autoencoder" / "best_model.py")
    diagnostics_env["OUT_DIR"] = str(BASE_DIR / "runs_autoencoder" / "diagnostics")
    diagnostics_cmd = [
        PYTHON,
        str(BASE_DIR / "diagnostics.py"),
    ]
    run_cmd(diagnostics_cmd, env=diagnostics_env)

    # Step 7: plot
    plot_env = diagnostics_env.copy()
    plot_env["OUT_PNG"] = str(BASE_DIR / "runs_autoencoder" / "pred_vs_true_test.png")
    plot_cmd = [
        PYTHON,
        str(BASE_DIR / "plot.py"),
    ]
    run_cmd(plot_cmd, env=plot_env)

    # Step 8: archive run directory
    archive_runs(tag)


def main() -> None:
    require_pyfastchem()
    require_env_vars()
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in DATASETS:
        process_dataset(cfg)

    print("\nAll datasets processed successfully.")


if __name__ == "__main__":
    main()

