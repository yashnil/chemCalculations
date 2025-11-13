#!/usr/bin/env python3
"""
merge_fastchem_outputs.py
=========================

Merge shard results (gas_species.csv) into a single CSV suitable for training.
Columns are reordered to match the reference CSV, and any missing species are
filled with zeros.

Example:
    python merge_fastchem_outputs.py \
        --jobs-root fastchem_jobs_x32 \
        --reference-csv ../NEW_VERS/all_gas_v10_no_stripe_clean.csv \
        --output-csv ../NEW_VERS/all_gas_v10_x32.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-root",
        required=True,
        help="Root directory containing job_* shard folders with FastChem results",
    )
    parser.add_argument(
        "--reference-csv",
        required=True,
        help="Reference CSV whose columns define ordering and metadata to carry over",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Destination CSV for the merged results",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Drop duplicate rows based on all columns (default: keep all)",
    )
    return parser


def discover_result_files(jobs_root: Path) -> List[Path]:
    files = []
    for job_dir in sorted(p for p in jobs_root.iterdir() if p.is_dir() and p.name.startswith("job_")):
        gas_path = job_dir / "results" / "gas_species.csv"
        if not gas_path.exists():
            raise FileNotFoundError(f"Missing gas_species.csv under {job_dir}")
        files.append(gas_path)
    if not files:
        raise RuntimeError(f"No result files found under {jobs_root}")
    return files


def main(args: argparse.Namespace) -> None:
    jobs_root = Path(args.jobs_root).resolve()
    reference_path = Path(args.reference_csv).resolve()
    output_path = Path(args.output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ref_df = pd.read_csv(reference_path)
    input_cols = [c for c in ref_df.columns if c.startswith("abund_") and c.endswith("_dex")]
    metadata_cols = ["T_K", "P_bar"] + input_cols
    species_cols = [c for c in ref_df.columns if c not in metadata_cols]

    print(f"[merge] jobs root : {jobs_root}")
    print(f"[merge] reference : {reference_path}")
    print(f"[merge] output    : {output_path}")
    print(f"[merge] metadata columns: {metadata_cols}")
    print(f"[merge] species columns : {len(species_cols)}")

    result_files = discover_result_files(jobs_root)
    shards = []
    for file in result_files:
        df = pd.read_csv(file)
        if "index" in df.columns:
            df = df.drop(columns=["index"])
        shards.append(df)
    gas_df = pd.concat(shards, axis=0, ignore_index=True)
    print(f"[merge] Loaded {len(gas_df)} prediction rows from {len(result_files)} shard(s)")

    # Align species columns to match reference. Fill missing species with zeros.
    for col in species_cols:
        if col not in gas_df.columns:
            gas_df[col] = 0.0
    extra_cols = [c for c in gas_df.columns if c not in species_cols]
    if extra_cols:
        gas_df = gas_df.drop(columns=extra_cols)
    gas_df = gas_df[species_cols]

    # Reattach metadata by matching index order. We assume shards were generated in the
    # same order as conditions.csv; therefore we simply reuse the metadata from the job folders.
    # Gather metadata by concatenating all conditions again.
    meta_shards = []
    for job_dir in sorted(p for p in jobs_root.iterdir() if p.is_dir() and p.name.startswith("job_")):
        cond_path = job_dir / "conditions.csv"
        meta = pd.read_csv(cond_path)
        meta_shards.append(meta[["T_K", "P_bar"] + input_cols])
    meta_df = pd.concat(meta_shards, axis=0, ignore_index=True)

    if len(meta_df) != len(gas_df):
        raise RuntimeError(
            f"Row mismatch between conditions ({len(meta_df)}) and gas results ({len(gas_df)})"
        )

    combined = pd.concat([meta_df, gas_df], axis=1)
    combined = combined[metadata_cols + species_cols]

    # Drop rows with NaNs/Infs before any further processing
    sanitized = combined.replace([np.inf, -np.inf], np.nan)
    bad_mask = sanitized.isna().any(axis=1)
    if bad_mask.any():
        dropped = int(bad_mask.sum())
        print(f"[merge] Warning: dropping {dropped} row(s) containing NaN/Inf values")
        combined = sanitized.loc[~bad_mask].reset_index(drop=True)
    else:
        combined = sanitized

    if args.drop_duplicates:
        before = len(combined)
        combined = combined.drop_duplicates()
        print(f"[merge] Dropped {before - len(combined)} duplicate rows")

    combined.to_csv(output_path, index=False)
    print(f"[merge] ✓ Saved merged CSV with {len(combined)} rows → {output_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())


