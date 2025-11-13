#!/usr/bin/env python3
"""
prepare_fastchem_jobs.py
========================

Create shard folders with sampled FastChem conditions.
By default we resample (with replacement) from an existing cleaned CSV and
apply small jitters in temperature / log-pressure / elemental abundances.

Each shard directory will contain:
    - conditions.csv      -> rows of (T_K, P_bar, abund_*_dex)
    - metadata.json       -> provenance and jitter/seed information

Usage (example):
    python prepare_fastchem_jobs.py \
        --total-samples 40000 \
        --shard-size 2000 \
        --output-root fastchem_jobs_x32 \
        --reference-csv ../NEW_VERS/all_gas_v10_no_stripe_clean.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-csv",
        required=True,
        help="CSV to resample from (e.g. NEW_VERS/all_gas_v10_no_stripe_clean.csv)",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where shard folders will be created",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        required=True,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2000,
        help="Number of rows per shard folder (default: 2000)",
    )
    parser.add_argument(
        "--strategy",
        choices=("empirical", "uniform"),
        default="empirical",
        help="Sampling strategy for T/P/composition (default: empirical resampling with jitter)",
    )
    parser.add_argument(
        "--temperature-range",
        type=float,
        nargs=2,
        metavar=("T_MIN", "T_MAX"),
        help="Override temperature range for uniform sampling (K)",
    )
    parser.add_argument(
        "--log-pressure-range",
        type=float,
        nargs=2,
        metavar=("LOGP_MIN", "LOGP_MAX"),
        help="Override log10 pressure range for uniform sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed",
    )
    parser.add_argument(
        "--temp-jitter",
        type=float,
        default=25.0,
        help="Std dev (K) for temperature jitter when using empirical sampling",
    )
    parser.add_argument(
        "--logp-jitter",
        type=float,
        default=0.1,
        help="Std dev in log10(P) for pressure jitter (empirical sampling)",
    )
    parser.add_argument(
        "--dex-jitter",
        type=float,
        default=0.02,
        help="Std dev for abundance epsilon jitter (empirical sampling)",
    )
    return parser


def empirical_resample(
    df: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
    temp_jitter: float,
    logp_jitter: float,
    dex_jitter: float,
) -> pd.DataFrame:
    idx = rng.integers(0, len(df), size=n_samples)
    sampled = df.iloc[idx].copy(deep=True).reset_index(drop=True)

    # Temperature jitter (ensure positive)
    sampled["T_K"] = np.clip(
        sampled["T_K"].to_numpy() + rng.normal(0.0, temp_jitter, size=n_samples),
        10.0,
        None,
    )

    # Log pressure jitter
    logp = np.log10(np.clip(sampled["P_bar"].to_numpy(), 1e-30, None))
    logp = logp + rng.normal(0.0, logp_jitter, size=n_samples)
    sampled["P_bar"] = 10.0 ** logp

    # Jitter elemental abundances (epsilon dex columns)
    dex_cols = [c for c in sampled.columns if c.startswith("abund_") and c.endswith("_dex")]
    if dex_cols:
        sampled[dex_cols] = sampled[dex_cols] + rng.normal(
            0.0, dex_jitter, size=(n_samples, len(dex_cols))
        )

    return sampled


def uniform_sample(
    df: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
    temp_range: tuple[float, float] | None,
    logp_range: tuple[float, float] | None,
) -> pd.DataFrame:
    if temp_range is None:
        temp_range = (float(df["T_K"].min()), float(df["T_K"].max()))
    if logp_range is None:
        logp_range = (
            float(np.log10(df["P_bar"].min())),
            float(np.log10(df["P_bar"].max())),
        )

    out = pd.DataFrame(index=np.arange(n_samples))
    out["T_K"] = rng.uniform(temp_range[0], temp_range[1], size=n_samples)
    out["P_bar"] = 10.0 ** rng.uniform(logp_range[0], logp_range[1], size=n_samples)

    # Elemental abundances: sample each epsilon independently using uniform over observed range.
    dex_cols = [c for c in df.columns if c.startswith("abund_") and c.endswith("_dex")]
    for col in dex_cols:
        out[col] = rng.uniform(float(df[col].min()), float(df[col].max()), size=n_samples)

    return out


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    reference_path = Path(args.reference_csv).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(reference_path)
    if not {"T_K", "P_bar"}.issubset(df.columns):
        raise ValueError("Reference CSV must contain 'T_K' and 'P_bar' columns.")

    if args.strategy == "empirical":
        sampled = empirical_resample(
            df,
            args.total_samples,
            rng,
            args.temp_jitter,
            args.logp_jitter,
            args.dex_jitter,
        )
    else:
        sampled = uniform_sample(
            df,
            args.total_samples,
            rng,
            args.temperature_range,
            args.log_pressure_range,
        )

    shard_size = args.shard_size
    n_shards = math.ceil(len(sampled) / shard_size)

    # Keep only the columns FastChem needs: temperature, pressure, and elemental abundances.
    required_cols = ["T_K", "P_bar"]
    dex_cols = [c for c in sampled.columns if c.startswith("abund_") and c.endswith("_dex")]
    missing_required = [c for c in required_cols if c not in sampled.columns]
    if missing_required:
        raise ValueError(f"Sampled data missing required columns: {missing_required}")
    keep_cols = required_cols + dex_cols
    if not dex_cols:
        raise ValueError("Reference CSV must include at least one 'abund_*_dex' column.")
    sampled = sampled[keep_cols]

    print(f"[prepare] reference: {reference_path}")
    print(f"[prepare] total samples: {len(sampled)} (shard size={shard_size}, shards={n_shards})")
    print(f"[prepare] output root: {output_root}")

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min((shard_idx + 1) * shard_size, len(sampled))
        shard_df = sampled.iloc[start:end].reset_index(drop=True)

        shard_dir = output_root / f"job_{shard_idx:04d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        conditions_path = shard_dir / "conditions.csv"
        metadata_path = shard_dir / "metadata.json"

        shard_df.to_csv(conditions_path, index=False)

        metadata = {
            "reference_csv": str(reference_path),
            "strategy": args.strategy,
            "seed": args.seed,
            "rows": len(shard_df),
            "start_index": int(start),
            "end_index": int(end),
        }
        if args.strategy == "empirical":
            metadata.update(
                {
                    "temp_jitter": args.temp_jitter,
                    "logp_jitter": args.logp_jitter,
                    "dex_jitter": args.dex_jitter,
                }
            )
        else:
            metadata.update(
                {
                    "temperature_range": args.temperature_range,
                    "log_pressure_range": args.log_pressure_range,
                }
            )

        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"[prepare] wrote {len(shard_df):4d} rows → {conditions_path}")

    print(f"Prepared {n_shards} shard(s) totalling {len(sampled)} samples.")


if __name__ == "__main__":
    main(build_parser().parse_args())


