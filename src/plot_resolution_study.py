#!/usr/bin/env python3
"""
Create separate comparative diagnostics charts across dataset sizes.
Now generates 4 separate figures instead of one 4-panel figure for easier quantification.

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
    # Default to the CSV in plots directory
    default_csv = Path(__file__).parent.parent / "plots" / "comparison_metrics.csv"
    parser.add_argument(
        "--metrics-csv",
        default=str(default_csv),
        help="Path to the CSV produced by the comparison metrics helper.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "plots" / "resolution_study.png"),
        help="Base path for generated figures (will create resolution_study_{metric}.png files).",
    )
    return parser


def annotate_points(ax, x, y, tags):
    """Annotate points with dataset tags."""
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

    # Create 4 separate figures instead of one 4-panel figure
    metrics = [
        ("val_loss", "Validation Loss", "validation_loss"),
        ("test_loss", "Test Loss", "test_loss"),
        ("log_mae", "Log-space MAE", "log_mae"),
        ("log_r2", "Log-space R²", "log_r2"),
    ]

    base_output_path = Path(args.output).expanduser().resolve()
    output_dir = base_output_path.parent

    for column, title, filename_suffix in metrics:
        if column not in df.columns:
            print(f"⚠️  Column '{column}' not found in CSV, skipping...")
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=150)
        
        # Filter out NaN values - use original index
        valid_mask = df[column].notna()
        samples_valid = samples[valid_mask].values
        y_valid = df[column][valid_mask].values
        tags_valid = tags[valid_mask].values
        
        ax.set_xscale("log")
        
        # Use log-log scale for loss and R² to see improvements better
        if column in ["val_loss", "test_loss", "log_mae"]:
            ax.set_yscale("log")
            ax.plot(samples_valid, y_valid, marker="o", linewidth=2.5, markersize=10, 
                   label=title, color="steelblue", zorder=3)
            annotate_points(ax, samples_valid, y_valid, tags_valid)
        elif column == "log_r2":
            # For R², use log scale on (1 - R²) to see improvements
            # R² goes from 0.98 to 0.999, so plot (1 - R²) on log scale
            y_transformed = 1.0 - y_valid
            ax.plot(samples_valid, y_transformed, marker="o", linewidth=2.5, markersize=10,
                   label=title, color="green", zorder=3)
            annotate_points(ax, samples_valid, y_transformed, tags_valid)
            ax.set_yscale("log")
            ax.set_ylabel("1 - Log R² (log scale)", fontsize=14)
        else:
            ax.plot(samples_valid, y_valid, marker="o", linewidth=2.5, markersize=10, label=title)
            annotate_points(ax, samples_valid, y_valid, tags_valid)
        
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Total samples (post-filter)", fontsize=13)
        if column != "log_r2":
            ax.set_ylabel(title, fontsize=13)
        ax.grid(True, which="both", linestyle="--", alpha=0.3, zorder=1)
        ax.legend(loc="best", fontsize=11, framealpha=0.9)
        
        # Save individual figure
        out_path = output_dir / f"resolution_study_{filename_suffix}.png"
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"[plot] saved {title} → {out_path}")
        plt.close(fig)
    
    print(f"\n✅ Generated 4 separate resolution study figures in {output_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
