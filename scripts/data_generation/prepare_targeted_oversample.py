#!/usr/bin/env python3
"""
prepare_targeted_oversample.py
==============================

Generate FastChem conditions in regions where the ML emulator shows highest error
(independent validation): low-pressure hot Jupiter regime and high C/O ratio.

Regions:
  1. Low-P (hot Jupiter): log10(P) in [-6, -4] bar, T in [800, 2200] K, solar composition
  2. High C/O: C/O in [1.5, 2.5], T and P across training range, solar metallicity

Output: Same shard format as prepare_fastchem_jobs.py (conditions.csv, metadata.json)
        for use with run_fastchem_batch.py and merge_fastchem_outputs.py.

Usage:
    python prepare_targeted_oversample.py \
        --output-root fastchem_jobs_targeted \
        --n-low-p 50000 \
        --n-high-co 50000 \
        --reference-csv data/datasets/all_gas_fastchem_x800.csv \
        --shard-size 2000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Solar composition (dex scale)
SOLAR = {"H": 12.00, "O": 8.69, "C": 8.43, "N": 7.83, "S": 7.12}


def sample_low_p(
    n_samples: int,
    rng: np.random.Generator,
    logp_range: tuple[float, float] = (-6.0, -4.0),
    temp_range: tuple[float, float] = (800.0, 2200.0),
) -> pd.DataFrame:
    """Sample conditions in low-pressure hot Jupiter regime (10^-6 to 10^-4 bar)."""
    T_K = rng.uniform(temp_range[0], temp_range[1], size=n_samples)
    log_p = rng.uniform(logp_range[0], logp_range[1], size=n_samples)
    P_bar = 10.0 ** log_p

    rows = []
    for i in range(n_samples):
        row = {
            "T_K": T_K[i],
            "P_bar": P_bar[i],
            "abund_H_dex": SOLAR["H"],
            "abund_O_dex": SOLAR["O"],
            "abund_C_dex": SOLAR["C"],
            "abund_N_dex": SOLAR["N"],
            "abund_S_dex": SOLAR["S"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def sample_high_co(
    n_samples: int,
    rng: np.random.Generator,
    co_range: tuple[float, float] = (1.5, 2.5),
    temp_range: tuple[float, float] = (1000.0, 2500.0),
    logp_range: tuple[float, float] = (-2.0, 1.0),
) -> pd.DataFrame:
    """Sample conditions with high C/O ratio (carbon-rich). C/O = 10^(C_dex - O_dex)."""
    T_K = rng.uniform(temp_range[0], temp_range[1], size=n_samples)
    log_p = rng.uniform(logp_range[0], logp_range[1], size=n_samples)
    P_bar = 10.0 ** log_p

    # C/O from 1.5 to 2.5: C_dex = O_dex + log10(C/O)
    co_ratios = rng.uniform(co_range[0], co_range[1], size=n_samples)
    O_dex = SOLAR["O"]
    C_dex = O_dex + np.log10(co_ratios)

    rows = []
    for i in range(n_samples):
        row = {
            "T_K": T_K[i],
            "P_bar": P_bar[i],
            "abund_H_dex": SOLAR["H"],
            "abund_O_dex": O_dex,
            "abund_C_dex": C_dex[i],
            "abund_N_dex": SOLAR["N"],
            "abund_S_dex": SOLAR["S"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where shard folders will be created",
    )
    parser.add_argument(
        "--reference-csv",
        required=True,
        help="Reference CSV for column schema (must match merge_fastchem_outputs)",
    )
    parser.add_argument(
        "--n-low-p",
        type=int,
        default=50000,
        help="Number of samples in low-P (hot Jupiter) regime (default: 50000)",
    )
    parser.add_argument(
        "--n-high-co",
        type=int,
        default=50000,
        help="Number of samples in high C/O regime (default: 50000)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2000,
        help="Number of rows per shard folder (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed (default: 2026, distinct from main pipeline)",
    )
    return parser


def main(args: argparse.ArgumentParser) -> None:
    rng = np.random.default_rng(args.seed)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reference_path = Path(args.reference_csv).resolve()

    ref_df = pd.read_csv(reference_path)
    dex_cols = [c for c in ref_df.columns if c.startswith("abund_") and c.endswith("_dex")]
    if not dex_cols:
        raise ValueError("Reference CSV must include abund_*_dex columns.")

    # Generate targeted samples
    df_low_p = sample_low_p(args.n_low_p, rng)
    df_high_co = sample_high_co(args.n_high_co, rng)

    # Ensure all dex columns exist (reference may have more elements)
    for col in dex_cols:
        if col not in df_low_p.columns:
            df_low_p[col] = ref_df[col].iloc[0]  # use first value as default
        if col not in df_high_co.columns:
            df_high_co[col] = ref_df[col].iloc[0]

    df_low_p = df_low_p[["T_K", "P_bar"] + dex_cols]
    df_high_co = df_high_co[["T_K", "P_bar"] + dex_cols]

    sampled = pd.concat([df_low_p, df_high_co], axis=0, ignore_index=True)
    # Shuffle so low-P and high-C/O are interleaved
    sampled = sampled.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    total = len(sampled)
    shard_size = args.shard_size
    n_shards = math.ceil(total / shard_size)

    print(f"[targeted] output root: {output_root}")
    print(f"[targeted] low-P samples: {args.n_low_p}, high C/O samples: {args.n_high_co}")
    print(f"[targeted] total: {total} (shard size={shard_size}, shards={n_shards})")

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min((shard_idx + 1) * shard_size, total)
        shard_df = sampled.iloc[start:end].reset_index(drop=True)

        shard_dir = output_root / f"job_{shard_idx:04d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        conditions_path = shard_dir / "conditions.csv"
        metadata_path = shard_dir / "metadata.json"

        shard_df.to_csv(conditions_path, index=False)

        metadata = {
            "strategy": "targeted_oversample",
            "regions": ["low_p_hot_jupiter", "high_co_carbon_rich"],
            "n_low_p": args.n_low_p,
            "n_high_co": args.n_high_co,
            "seed": args.seed,
            "rows": len(shard_df),
            "start_index": int(start),
            "end_index": int(end),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"[targeted] wrote {len(shard_df):4d} rows → {conditions_path}")

    print(f"Prepared {n_shards} shard(s) totalling {total} targeted samples.")


if __name__ == "__main__":
    main(build_parser().parse_args())
