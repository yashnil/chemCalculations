#!/usr/bin/env python3
"""
run_fastchem_all.py
===================

Convenience wrapper that iterates over all shard folders produced by
prepare_fastchem_jobs.py and invokes run_fastchem_batch.py on each one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-root",
        required=True,
        help="Root directory containing job_* shard folders",
    )
    parser.add_argument(
        "--logk",
        help="Path to FastChem logK.dat (or rely on FASTCHEM_LOGK)",
    )
    parser.add_argument(
        "--logk-cond",
        help="Path to FastChem logK_condensates.dat (or rely on FASTCHEM_COND)",
    )
    parser.add_argument(
        "--element-abundances",
        help="Path to FastChem element abundance table (or rely on FASTCHEM_ELEM / inference)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=128,
        help="Chunksize passed to run_fastchem_batch.py (default: 128)",
    )
    parser.add_argument(
        "--condensates",
        choices=("true", "false"),
        default="false",
        help="Enable condensate calculations (default: false)",
    )
    parser.add_argument(
        "--batch-script",
        default=str(Path(__file__).resolve().parent / "run_fastchem_batch.py"),
        help="Path to the batch runner script (default: run_fastchem_batch.py)",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    jobs_root = Path(args.jobs_root).resolve()
    if not jobs_root.exists():
        raise FileNotFoundError(f"{jobs_root} does not exist")

    job_dirs = sorted([p for p in jobs_root.iterdir() if p.is_dir() and p.name.startswith("job_")])
    if not job_dirs:
        raise RuntimeError(f"No shard folders found under {jobs_root}")

    print(f"[run_all] Found {len(job_dirs)} shard(s) under {jobs_root}")

    for job_dir in job_dirs:
        output_dir = job_dir / "results"
        cmd = [
            "python",
            args.batch_script,
            "--job-dir",
            str(job_dir),
            "--output-dir",
            str(output_dir),
            "--chunksize",
            str(args.chunksize),
            "--condensates",
            args.condensates,
        ]
        if args.logk:
            cmd += ["--logk", args.logk]
        if args.logk_cond:
            cmd += ["--logk-cond", args.logk_cond]
        if args.element_abundances:
            cmd += ["--element-abundances", args.element_abundances]

        print(f"[run_all] Running FastChem for {job_dir.name} …")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"run_fastchem_batch.py failed for {job_dir} (exit={result.returncode})")

    print("[run_all] ✓ Completed all shards")


if __name__ == "__main__":
    main(build_parser().parse_args())


