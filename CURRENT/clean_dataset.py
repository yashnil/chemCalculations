#!/usr/bin/env python3
"""
Remove rows containing NaN/Inf values from a CSV produced by the FastChem pipeline.

Usage:
    python clean_dataset.py --input datasets/all_gas_fastchem_x48.csv
    python clean_dataset.py --input datasets/all_gas_fastchem_x48.csv --output cleaned.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the CSV to clean")
    parser.add_argument(
        "--output",
        help="Destination for the cleaned CSV (defaults to <input> with _clean suffix)",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist")

    df = pd.read_csv(input_path)
    sanitized = df.replace([np.inf, -np.inf], np.nan)
    bad_mask = sanitized.isna().any(axis=1)
    dropped = int(bad_mask.sum())

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_name(input_path.stem + "_clean.csv")

    if dropped:
        cleaned = sanitized.loc[~bad_mask].reset_index(drop=True)
        cleaned.to_csv(output_path, index=False)
        print(f"[clean_dataset] dropped {dropped} row(s); wrote {len(cleaned)} rows → {output_path}")
    else:
        df.to_csv(output_path, index=False)
        print(f"[clean_dataset] no NaN/Inf rows found; copied dataset → {output_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())

