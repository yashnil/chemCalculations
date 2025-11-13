#!/usr/bin/env python3
"""
Create a comparative diagnostics chart across dataset sizes.

Usage:
    python plot_resolution_study.py \
        --metrics-csv comparison_metrics.csv \
        --output resolution_study.png

The input CSV is expected to match the format emitted by the helper script:

dataset,total_samples,val_loss,test_loss,log_mae,log_r2,linear_mae,linear_mse
base,12412,7.54e-04,9.23e-04,1.44e-01,0.981,1.74e-02,8.77e-03
x32,31997,4.79e-04,6.30e-04,1.88e-01,0.996,3.75e+20,inf
...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot resolution study metrics vs dataset size.")
    parser.add_argument(
        "--metrics-csv",
        default="comparison_metrics.csv",
        help="Path to the CSV produced by the comparison metrics helper.",
    )
    parser.add_argument(
        "--output",
        default="resolution_study.png",
        help="Destination path for the generated figure.",
    )
    return parser


def add_series(ax, x, y, label, color=None):
    return ax.plot(x, y, marker="o", label=label, color=color)


def annotate_points(ax, x, y, tags):
    for xi, yi, tag in zip(x, y, tags):
        ax.annotate(tag, (xi, yi), textcoords="offset points", xytext=(5, -10), fontsize=8)


def main(args: argparse.Namespace) -> None:
    csv_path = Path(args.metrics_csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "total_samples" not in df.columns:
        raise ValueError("metrics CSV must include a 'total_samples' column.")

    # Sort by dataset size to keep the lines monotonic.
    df = df.sort_values("total_samples").reset_index(drop=True)

    samples = df["total_samples"]
    tags = df["dataset"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=150, sharex=True)
    axes = axes.ravel()

    metrics = [
        ("val_loss", "Validation Loss"),
        ("test_loss", "Test Loss"),
        ("log_mae", "Log-space MAE"),
        ("log_r2", "Log-space R²"),
    ]

    for ax, (column, title) in zip(axes, metrics):
        if column not in df.columns:
            ax.set_visible(False)
            continue
        y = df[column]
        add_series(ax, samples, y, title)
        annotate_points(ax, samples, y, tags)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(loc="best")

    axes[2].set_xlabel("Total samples (post-filter)")
    axes[3].set_xlabel("Total samples (post-filter)")

    fig.tight_layout()
    out_path = Path(args.output).expanduser().resolve()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[plot] saved resolution study chart → {out_path}")

    fig.tight_layout()
    out_path = Path(args.output).expanduser().resolve()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[plot] saved resolution study chart → {out_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())

