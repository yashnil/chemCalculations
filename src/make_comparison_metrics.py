#!/usr/bin/env python3
"""
Aggregate metrics from archived runs into UPDATED_VERS/comparison_metrics.csv
so resolution_study.png can be regenerated with consistent target species.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

BASE = Path(__file__).resolve().parent


def parse_global_metrics(txt_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not txt_path.exists():
        return metrics
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, val = [x.strip() for x in line.split(":", 1)]
        # map keys to column names we want
        if key.startswith("Log MAE"):
            metrics["log_mae"] = float(val.replace(",", " ").split()[0])
        elif key.startswith("Log R"):
            # handles "Log R²" or "Log R^2"
            val_num = val.replace(",", " ").split()[0]
            metrics["log_r2"] = float(val_num)
    return metrics


def collect_one(tag: str, run_dir: Path, total_samples: Optional[int]) -> Optional[Dict[str, object]]:
    summary_path = run_dir / "summary.json"
    diag_metrics = run_dir / "diagnostics" / "global_metrics.txt"
    if not summary_path.exists():
        return None
    s = json.loads(summary_path.read_text())
    gm = parse_global_metrics(diag_metrics)
    row: Dict[str, object] = {
        "dataset": tag,
        "total_samples": int(total_samples) if total_samples is not None else (s.get("train_samples", 0) + s.get("val_samples", 0) + s.get("test_samples", 0)),
        "val_loss": float(s.get("val_loss")),
        "test_loss": float(s.get("test_loss")),
        "log_mae": float(gm.get("log_mae", "nan")),
        "log_r2": float(gm.get("log_r2", "nan")),
        "linear_mae": float(s.get("test_mae_linear")),
        "linear_mse": float(s.get("test_mse_linear", s.get("test_loss_linear", float("nan")))),
    }
    return row


def main() -> None:
    # Map run archives to expected sample counts (post-filter counts from datasets)
    runs = [
        ("base", BASE / "runs_autoencoder_base", None),  # original base
        ("x32", BASE / "runs_autoencoder_x32", 31997),
        ("x48", BASE / "runs_autoencoder_x48", 47997),
        ("x64", BASE / "runs_autoencoder_x64", 63985),
        ("x80", BASE / "runs_autoencoder_x80", 79992),
        ("x96", BASE / "runs_autoencoder_x96", 96000),
        ("x112", BASE / "runs_autoencoder_x112", 112000),
        ("x128", BASE / "runs_autoencoder_x128", 128000),
        ("x144", BASE / "runs_autoencoder_x144", 144000),
        ("x160", BASE / "runs_autoencoder_x160", 160000),
        ("x176", BASE / "runs_autoencoder_x176", 176000),
    ]

    rows = []
    for tag, run_dir, samples in runs:
        row = collect_one(tag, run_dir, samples)
        if row:
            rows.append(row)

    if not rows:
        print("No runs found to aggregate.")
        return

    out_csv = BASE / "comparison_metrics.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "total_samples",
                "val_loss",
                "test_loss",
                "log_mae",
                "log_r2",
                "linear_mae",
                "linear_mse",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[metrics] wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()


