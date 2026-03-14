#!/usr/bin/env python3
"""
run_fastchem_batch.py
=====================

Execute FastChem for a single shard directory produced by prepare_fastchem_jobs.py.
This script relies on the FastChem Python bindings (https://github.com/exoclime/FastChem).

Output: Species number densities in cm⁻³ (FastChem default; output_data.number_densities).

Example:
    python run_fastchem_batch.py \
        --job-dir fastchem_jobs_x32/job_0000 \
        --output-dir fastchem_jobs_x32/job_0000/results \
        --logk $FASTCHEM_LOGK \
        --logk-cond $FASTCHEM_COND \
        --chunksize 128 \
        --condensates false
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyfastchem
from tqdm import tqdm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, help="Shard directory containing conditions.csv")
    parser.add_argument(
        "--output-dir",
        help="Directory to store results (defaults to <job-dir>/results)",
    )
    parser.add_argument(
        "--logk",
        default=None,
        help="Path to FastChem logK.dat (falls back to $FASTCHEM_LOGK)",
    )
    parser.add_argument(
        "--logk-cond",
        default=None,
        help="Path to FastChem logK_condensates.dat (falls back to $FASTCHEM_COND)",
    )
    parser.add_argument(
        "--element-abundances",
        default=None,
        help="Path to FastChem element abundance table (falls back to $FASTCHEM_ELEM or inferred from --logk)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=128,
        help="Batch size when calling FastChem (default: 128)",
    )
    parser.add_argument(
        "--condensates",
        choices=("true", "false"),
        default="false",
        help="Enable condensate calculations (default: false)",
    )
    parser.add_argument(
        "--pressure-unit",
        default="bar",
        help="Pressure unit to pass to FastChem (default: bar)",
    )
    parser.add_argument(
        "--abundance-unit",
        default="number_fraction",
        help="Elemental abundance unit (default: number_fraction = linear ratio to H)",
    )
    return parser


def resolve_path(value: str | None, env_var: str) -> Path | None:
    if value:
        return Path(value).expanduser().resolve()
    import os

    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return None


def infer_element_path(logk_path: Path) -> Path | None:
    # Attempt to infer the element abundance file from the logK path.
    candidates = [
        logk_path.parent.parent / "element_abundances" / "asplund_2009.dat",
        logk_path.parent.parent / "element_abundances" / "solar.abundances",  # alternate naming
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_fastchem(
    logk_path: Path,
    logk_cond_path: Path | None,
    elements_path: Path,
    use_condensates: bool,
):
    try:
        from fastchem import FastChem  # type: ignore
    except ImportError:
        try:
            from pyfastchem import FastChem  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Could not import FastChem Python bindings. Make sure they are installed "
                "(pip install pyfastchem or build from source) and available on PYTHONPATH."
            ) from exc

    # Prefer positional constructors as exposed by pyfastchem.
    if logk_cond_path is not None:
        engine = FastChem(str(elements_path), str(logk_path), str(logk_cond_path), int(use_condensates))
    else:
        engine = FastChem(str(elements_path), str(logk_path), int(use_condensates))
    return engine


def build_element_matrix(
    df: pd.DataFrame,
    element_symbols: List[str],
    base_vector: np.ndarray,
) -> np.ndarray:
    """
    Construct an (n_rows × n_elements) matrix of elemental number fractions.
    Input columns are provided as epsilon values (12 + log10(N_X / N_H)).
    """
    n_rows = len(df)
    n_elements = len(element_symbols)
    if base_vector.shape[0] != n_elements:
        raise ValueError("Base element abundance vector length does not match FastChem element list.")
    abundances = np.tile(base_vector.reshape(1, -1), (n_rows, 1)).astype(np.float64, copy=True)

    # Track which dataset columns we actually use for diagnostics.
    present_symbols: List[str] = []

    for idx, symbol in enumerate(element_symbols):
        col_name = f"abund_{symbol}_dex"
        if col_name in df.columns:
            eps = df[col_name].to_numpy(dtype=np.float64, copy=False)
            abundances[:, idx] = 10.0 ** (eps - 12.0)
            present_symbols.append(symbol)

    if "H" not in present_symbols:
        raise ValueError("conditions.csv must include 'abund_H_dex' so hydrogen can be normalised.")

    # Ensure hydrogen remains positive to avoid numerical issues.
    h_idx = element_symbols.index("H")
    abundances[:, h_idx] = np.clip(abundances[:, h_idx], 1e-30, None)

    missing = [f"abund_{s}_dex" for s in element_symbols if f"abund_{s}_dex" not in df.columns]
    if missing:
        print(f"[warn] Missing abundance columns for {len(missing)} element(s); defaulting to zero: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    return abundances


def main(args: argparse.Namespace) -> None:
    job_dir = Path(args.job_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else job_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions_path = job_dir / "conditions.csv"
    if not conditions_path.exists():
        raise FileNotFoundError(f"{conditions_path} does not exist")

    logk_path = resolve_path(args.logk, "FASTCHEM_LOGK")
    if logk_path is None:
        raise ValueError("Provide --logk or set FASTCHEM_LOGK")

    logk_cond_path = resolve_path(args.logk_cond, "FASTCHEM_COND")
    use_condensates = args.condensates.lower() == "true"
    if use_condensates and logk_cond_path is None:
        raise ValueError("Condensates requested but logK condensate path missing (use --logk-cond)")

    elements_path = resolve_path(args.element_abundances, "FASTCHEM_ELEM")
    if elements_path is None:
        inferred = infer_element_path(logk_path)
        if inferred is None:
            raise ValueError(
                "Could not infer element abundance file. Provide --element-abundances or set FASTCHEM_ELEM."
            )
        elements_path = inferred

    df = pd.read_csv(conditions_path)
    if not {"T_K", "P_bar"}.issubset(df.columns):
        raise ValueError("conditions.csv must have T_K and P_bar columns")

    print(f"[run] job dir     : {job_dir}")
    print(f"[run] output dir  : {output_dir}")
    print(f"[run] logK        : {logk_path}")
    if logk_cond_path:
        print(f"[run] logK_cond   : {logk_cond_path}")
    print(f"[run] elements    : {elements_path}")
    print(f"[run] condensates : {use_condensates}")
    print(f"[run] rows        : {len(df)}")

    template_engine = load_fastchem(logk_path, logk_cond_path, elements_path, use_condensates)
    element_symbols = [template_engine.getElementSymbol(i) for i in range(template_engine.getElementNumber())]
    base_vector = np.array(template_engine.getElementAbundances(), dtype=np.float64, copy=True)
    species_names = [template_engine.getGasSpeciesSymbol(i) for i in range(template_engine.getGasSpeciesNumber())]
    cond_species_names = [template_engine.getCondSpeciesSymbol(i) for i in range(template_engine.getCondSpeciesNumber())]
    del template_engine
    column_arrays: dict[str, np.ndarray] = {}
    for symbol in element_symbols:
        col_name = f"abund_{symbol}_dex"
        if col_name in df.columns:
            column_arrays[symbol] = df[col_name].to_numpy(dtype=np.float64, copy=False)


    temperatures = df["T_K"].to_numpy(dtype=np.float64, copy=False)
    pressures = df["P_bar"].to_numpy(dtype=np.float64, copy=False)

    chunksize = args.chunksize
    n_chunks = math.ceil(len(df) / chunksize)

    gas_outputs: List[np.ndarray] = []
    cond_outputs: List[np.ndarray] = []
    fail_indices: List[int] = []

    for chunk_idx in tqdm(range(n_chunks), desc=f"FastChem {job_dir.name}"):
        start = chunk_idx * chunksize
        end = min((chunk_idx + 1) * chunksize, len(df))

        for row_idx in range(start, end):
            engine = load_fastchem(logk_path, logk_cond_path, elements_path, use_condensates)
            vec = base_vector.copy()
            for idx, symbol in enumerate(element_symbols):
                col_vals = column_arrays.get(symbol)
                if col_vals is not None:
                    vec[idx] = 10.0 ** (col_vals[row_idx] - 12.0)
            engine.setElementAbundances(vec.tolist())

            input_data = pyfastchem.FastChemInput()
            output_data = pyfastchem.FastChemOutput()
            input_data.temperature = np.array([temperatures[row_idx]], dtype=np.float64)
            input_data.pressure = np.array([pressures[row_idx]], dtype=np.float64)

            flag = engine.calcDensities(input_data, output_data)
            if flag != pyfastchem.FASTCHEM_SUCCESS:
                fail_indices.append(row_idx)
                gas_vec = np.full(len(species_names), np.nan, dtype=np.float64)
                cond_vec = np.full(len(cond_species_names), np.nan, dtype=np.float64)
            else:
                gas_vec = np.array(output_data.number_densities[0], dtype=np.float64, copy=True)
                if cond_species_names and hasattr(output_data, "cond_number_densities"):
                    cond_vec = np.array(output_data.cond_number_densities[0], dtype=np.float64, copy=True)
                else:
                    cond_vec = np.empty(0, dtype=np.float64)

            gas_outputs.append(gas_vec)
            if cond_species_names and cond_vec.size:
                cond_outputs.append(cond_vec)

            del engine

    gas_array = np.vstack(gas_outputs) if gas_outputs else np.empty((0, len(species_names)))
    gas_df = pd.DataFrame(gas_array, columns=species_names)
    gas_out = output_dir / "gas_species.csv"
    gas_df.to_csv(gas_out, index=False)
    print(f"[run] wrote {len(gas_df)} rows → {gas_out}")

    metadata = {
        "job_dir": str(job_dir),
        "rows": len(df),
        "chunksize": chunksize,
        "logk_path": str(logk_path),
        "logk_cond_path": str(logk_cond_path) if logk_cond_path else None,
        "elements_path": str(elements_path),
        "condensates": use_condensates,
        "pressure_unit": args.pressure_unit,
        "species_columns": species_names,
    }

    if cond_species_names:
        cond_array = np.vstack(cond_outputs) if cond_outputs else np.empty((0, len(cond_species_names)))
        cond_df = pd.DataFrame(cond_array, columns=cond_species_names)
        cond_path = output_dir / "condensates_species.csv"
        cond_df.to_csv(cond_path, index=False)
        print(f"[run] wrote {len(cond_df)} rows → {cond_path}")
        metadata["cond_species_columns"] = cond_species_names

    if fail_indices:
        metadata["failed_indices"] = fail_indices
        print(f"[warn] FastChem did not converge for {len(fail_indices)} row(s); NaNs written to output.")

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[run] ✓ Saved gas-phase abundances → {gas_out}")


if __name__ == "__main__":
    main(build_parser().parse_args())


