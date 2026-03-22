"""
Mean fractional absolute error (MFAE) over all (sample, species) pairs.

MFAE = mean(|pred - true| / true) for all pairs where true >= threshold.
This matches "average of all the dots" in parity/scatter plots (same threshold as diagnostics).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
RUNS_DIR = BASE_DIR / "results" / "runs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_THRESHOLD = 1e-10  # same as diagnostics.AAFE_THRESHOLD
# Cap each pair's fractional error before averaging — raw mean is dominated by rare
# (sample, species) pairs where true is tiny but pred is wrong by orders of magnitude.
WINSOR_CAP = 2.0  # count at most 200% relative error per pair toward the mean


def load_best_model_module(run_tag: str):
    """Load best_model.py from runs_autoencoder_{run_tag}."""
    run_dir = RUNS_DIR / f"runs_autoencoder_{run_tag}"
    best_path = run_dir / "best_model.py"
    if not best_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"best_model_{run_tag}", best_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dataset_csv_for_run(run_tag: str) -> Path | None:
    """Infer CSV path from run tag (e.g. x4800_improved -> all_gas_fastchem_x4800.csv)."""
    size = run_tag.split("_")[0].replace("x", "")
    csv_path = BASE_DIR / "data" / "datasets" / f"all_gas_fastchem_x{size}.csv"
    return csv_path if csv_path.exists() else None


def compute_mfae_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    winsor_cap: float | None = WINSOR_CAP,
) -> tuple[float, float]:
    """
    Returns (mfae_winsorized_or_mean, mfae_median) for arrays shaped (n_samples, n_species).
    """
    yt = y_true.ravel()
    yp = y_pred.ravel()
    mask = (yt > threshold) & (yp > threshold)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    denom = np.maximum(yt[mask], 1e-300)
    frac = np.abs(yp[mask] - yt[mask]) / denom
    frac = frac.astype(np.float64)
    med = float(np.median(frac))
    if winsor_cap is not None:
        frac_m = np.minimum(frac, winsor_cap)
    else:
        frac_m = frac
    return float(np.mean(frac_m)), med


def compute_mfae_for_run(
    run_tag: str,
    threshold: float = DEFAULT_THRESHOLD,
    winsor_cap: float | None = WINSOR_CAP,
) -> float | None:
    """
    Average fractional absolute error over scatter-plot points (both true and pred > threshold).

    For each (sample, species): frac = |pred - true| / true. The raw arithmetic mean is dominated
    by extreme outliers; by default we use a winsorized mean (each frac capped at ``winsor_cap``)
    so the value matches an intuitive "average error". Set ``winsor_cap=None`` for the raw mean.

    Returns None if model or data is missing.
    """
    mod = load_best_model_module(run_tag)
    if mod is None:
        return None
    csv_path = dataset_csv_for_run(run_tag)
    if csv_path is None:
        return None

    df = pd.read_csv(csv_path)
    splits = getattr(mod, "SPLITS", {})
    test_idx = np.array(splits.get("test_idx", []), dtype=int)
    if len(test_idx) == 0:
        return None

    df_test = df.iloc[test_idx].reset_index(drop=True)
    target_cols = mod.TARGET_COLS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mod.load_model(device=device)
    model.eval()
    X_test = mod.normalize_inputs(df_test)
    with torch.no_grad():
        pred_scaled = mod.forward_autoencoder(model, X_test)
        pred_scaled = pred_scaled.cpu().numpy()
    y_pred = mod.denormalize_targets(pred_scaled)
    y_true = df_test[target_cols].to_numpy(dtype=np.float64, copy=True)
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)

    yt = y_true.ravel()
    yp = y_pred.ravel()
    # Same mask as parity/scatter log plots: both true and pred must exceed threshold
    # (otherwise |pred-true|/true blows up when pred ≪ true)
    mask = (yt > threshold) & (yp > threshold)
    if mask.sum() == 0:
        return None
    denom = np.maximum(yt[mask], 1e-300)
    mfae, _med = compute_mfae_from_arrays(y_true, y_pred, threshold, winsor_cap)
    return mfae
