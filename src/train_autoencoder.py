#!/usr/bin/env python3
"""
train_autoencoder.py
====================

Minimal training harness that fits the FlowMapAutoencoder on the FastChem CSV.
The pipeline mirrors the old `run_mlp.py` data handling (automatic feature /
target discovery, log-space target scaling) but swaps in the autoencoder
architecture supplied by the user.

Usage:
    python train_autoencoder.py
    python train_autoencoder.py --loss-type mse --run-dir runs_autoencoder_x160_mse

Outputs are written to `runs_autoencoder/` (checkpoints, loss curves, metrics).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.model_selection import train_test_split
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required. Please install scikit-learn.") from exc

from autoencoder_model import FlowMapAutoencoder


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_PATH = os.environ.get(
    "CSV_PATH",
    "/Users/yashnilmohanty/Desktop/chemCalculations/NEW_VERS/all_gas_v10_no_stripe_clean.csv",
)
OUT_DIR = Path("runs_autoencoder")

SEED = 42
TRAIN_FRAC = 0.85
VAL_FRAC = 0.10
TEST_FRAC = 0.05
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-8

# optimisation
EPOCHS = 50
BATCH_SIZE = 512
LR = 5e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0

# autoencoder architecture
LATENT_DIM = 96
ENCODER_HIDDEN = [512, 512, 512]
DYNAMICS_HIDDEN = [512, 512, 512]
DECODER_HIDDEN = [512, 512, 512]
ACTIVATION = "silu"
DROPOUT = 0.0

TARGET_TOPK_SPECIES = 20

# normalisation constants (aligned with previous baseline)
TEMP_DIVISOR = 4_000.0
INPUT_LOG_SCALE = 10.0
ABUND_EPSILON_OFFSET = 12.0
ABUND_DEX_SCALE = 10.0
TARGET_ZERO_FLOOR = 1e-30
TARGET_LOG_SCALE = 30.0
LOG_EPS = 1e-30

INCLUDE_FZ_AS_FEATURE = True
INPUT_COLS_MANUAL: Optional[List[str]] = None
TARGET_COLS_MANUAL: Optional[List[str]] = None
NEVER_TARGET_COLS = {"T_K", "P_bar", "fZ", "fZ_dex", "flag", "flag_msg", "mean_molecular_weight", "total_element_density"}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("autoencoder")


def fmt(x) -> str:
    if isinstance(x, torch.Tensor):
        x = x.item()
    if isinstance(x, (int, np.integer)) and abs(x) < 1e6:
        return str(int(x))
    try:
        return f"{float(x):.3e}"
    except Exception:
        return str(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Normalisation utilities
# ---------------------------------------------------------------------------

def safe_log10(arr: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(arr, LOG_EPS, None))


def is_abund_col(name: str) -> bool:
    return name.startswith("abund_") and name.endswith("_dex")


def normalize_inputs_df(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    X = df[columns].to_numpy(dtype=np.float64)
    name_to_idx = {c: i for i, c in enumerate(columns)}
    if "T_K" in name_to_idx:
        X[:, name_to_idx["T_K"]] = X[:, name_to_idx["T_K"]] / TEMP_DIVISOR
    if "P_bar" in name_to_idx:
        X[:, name_to_idx["P_bar"]] = safe_log10(X[:, name_to_idx["P_bar"]]) / INPUT_LOG_SCALE
    if INCLUDE_FZ_AS_FEATURE:
        if "fZ_dex" in name_to_idx:
            X[:, name_to_idx["fZ_dex"]] = X[:, name_to_idx["fZ_dex"]] / INPUT_LOG_SCALE
        if "fZ" in name_to_idx:
            X[:, name_to_idx["fZ"]] = safe_log10(X[:, name_to_idx["fZ"]]) / INPUT_LOG_SCALE
    for c, idx in name_to_idx.items():
        if is_abund_col(c):
            X[:, idx] = (X[:, idx] - ABUND_EPSILON_OFFSET) / ABUND_DEX_SCALE
    return X


def scale_targets_linear_to_train(y_linear: np.ndarray) -> np.ndarray:
    y = y_linear.copy()
    y[y < TARGET_ZERO_FLOOR] = 0.0
    return safe_log10(y) / TARGET_LOG_SCALE


def scale_targets_train_to_linear(y_scaled: np.ndarray) -> np.ndarray:
    y_log = y_scaled * TARGET_LOG_SCALE
    return np.clip(np.power(10.0, y_log), 0.0, None)


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

def resolve_input_columns(df: pd.DataFrame) -> List[str]:
    if INPUT_COLS_MANUAL:
        missing = [c for c in INPUT_COLS_MANUAL if c not in df.columns]
        if missing:
            raise ValueError(f"Manual input columns missing: {missing}")
        return list(INPUT_COLS_MANUAL)

    cols: List[str] = []
    if "T_K" not in df.columns or "P_bar" not in df.columns:
        raise ValueError("Expected columns 'T_K' and 'P_bar' not found.")
    cols.extend(["T_K", "P_bar"])

    if INCLUDE_FZ_AS_FEATURE:
        if "fZ_dex" in df.columns:
            cols.append("fZ_dex")
        elif "fZ" in df.columns:
            cols.append("fZ")

    abund_cols = sorted([c for c in df.columns if is_abund_col(c)])
    cols.extend(abund_cols)
    log.info("Input columns (%d): %s", len(cols), cols[:10] + (["..."] if len(cols) > 10 else []))
    return cols


def species_candidates(df: pd.DataFrame, input_cols: Sequence[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set(input_cols) | set(NEVER_TARGET_COLS)
    return [c for c in numeric_cols if c not in exclude and not is_abund_col(c) and not c.startswith("comp_")]


def topk_species_linear_mean(df: pd.DataFrame, candidates: Sequence[str], k: int) -> List[str]:
    if not candidates or k <= 0:
        return []
    stats = []
    for col in candidates:
        vals = df[col].to_numpy(dtype=float, copy=False)
        vals = np.where(np.isfinite(vals), np.clip(vals, 0.0, None), np.nan)
        stats.append((float(np.nanmean(vals)), col))
    stats.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in stats[:k]]


def resolve_target_columns(df: pd.DataFrame, input_cols: Sequence[str]) -> List[str]:
    if TARGET_COLS_MANUAL:
        missing = [c for c in TARGET_COLS_MANUAL if c not in df.columns]
        if missing:
            raise ValueError(f"Manual target columns missing: {missing}")
        return list(TARGET_COLS_MANUAL)

    candidates = species_candidates(df, input_cols)
    top = topk_species_linear_mean(df, candidates, TARGET_TOPK_SPECIES)
    if "e-" in df.columns and "e-" in candidates and "e-" not in top:
        top.insert(0, "e-")
    log.info("Target columns (%d): %s", len(top), top[:10] + (["..."] if len(top) > 10 else []))
    return top


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class AutoencoderDataset(Dataset):
    def __init__(self, g: np.ndarray, y: np.ndarray):
        self.g = torch.as_tensor(g, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.g.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.g[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class WeightedHuber(nn.Module):
    def __init__(self, delta: float = 0.02, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.delta = delta
        self.register_buffer("w", weights if weights is not None else None)

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        err = pred - true
        abs_err = err.abs()
        huber = torch.where(
            abs_err <= self.delta,
            0.5 * (err ** 2) / self.delta,
            abs_err - 0.5 * self.delta,
        )
        if self.w is not None:
            huber = huber * self.w.view(1, -1)
        return huber.mean()


def compute_target_weights(y_linear: np.ndarray, present_floor: float = 1e-8) -> np.ndarray:
    freq = (y_linear > present_floor).mean(axis=0)
    w = 1.0 / np.sqrt(np.clip(freq, 1e-6, None))
    return (w / w.mean()).astype(np.float32)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    loss: float
    mse: float
    mae: float
    log_mae: float = 0.0


def evaluate(model: FlowMapAutoencoder, loader: DataLoader, device: torch.device) -> EvalResult:
    model.eval()
    criterion = nn.MSELoss()
    losses, mses, maes, log_maes = [], [], [], []

    with torch.no_grad():
        for g, y in loader:
            g = g.to(device)
            y = y.to(device)

            y0 = torch.zeros_like(y)
            dt = torch.ones((g.shape[0], 1), device=device, dtype=g.dtype)
            pred = model(y0, dt, g)[:, 0, :]

            loss = criterion(pred, y)
            y_lin = scale_targets_train_to_linear_tensor(y)
            pred_lin = scale_targets_train_to_linear_tensor(pred)
            mse = criterion(pred_lin, y_lin)
            mae = torch.mean((pred_lin - y_lin).abs())
            
            # Compute log MAE
            y_log = torch.log10(torch.clamp(y_lin, min=1e-30))
            pred_log = torch.log10(torch.clamp(pred_lin, min=1e-30))
            log_mae = torch.mean((pred_log - y_log).abs())

            losses.append(loss.item())
            mses.append(mse.item())
            maes.append(mae.item())
            log_maes.append(log_mae.item())

    return EvalResult(
        loss=float(np.mean(losses)),
        mse=float(np.mean(mses)),
        mae=float(np.mean(maes)),
        log_mae=float(np.mean(log_maes)),
    )


def scale_targets_train_to_linear_tensor(y_scaled: torch.Tensor) -> torch.Tensor:
    return torch.clamp((10.0 ** (y_scaled * TARGET_LOG_SCALE)), min=0.0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def write_best_model_py(
    module_path: Path,
    checkpoint_path: Path,
    input_cols: Sequence[str],
    target_cols: Sequence[str],
    weights: Sequence[float],
    splits: dict,
) -> None:
    code = f"""#!/usr/bin/env python3
# Auto-generated by train_autoencoder.py at {time.strftime("%Y-%m-%d %H:%M:%S")}
# Reconstructs the FlowMapAutoencoder and exposes helper utilities for inference.

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from autoencoder_model import FlowMapAutoencoder

INPUT_COLS = {list(input_cols)!r}
TARGET_COLS = {list(target_cols)!r}
TARGET_WEIGHTS = {list(weights)!r}
SPLITS = {splits!r}

CHECKPOINT_PATH = {str(checkpoint_path)!r}

TEMP_DIVISOR = {TEMP_DIVISOR:.10g}
INPUT_LOG_SCALE = {INPUT_LOG_SCALE:.10g}
ABUND_EPSILON_OFFSET = {ABUND_EPSILON_OFFSET:.10g}
ABUND_DEX_SCALE = {ABUND_DEX_SCALE:.10g}
TARGET_ZERO_FLOOR = {TARGET_ZERO_FLOOR:.10g}
TARGET_LOG_SCALE = {TARGET_LOG_SCALE:.10g}
LOG_EPS = {LOG_EPS:.10g}

LATENT_DIM = {LATENT_DIM}
ENCODER_HIDDEN = {ENCODER_HIDDEN!r}
DYNAMICS_HIDDEN = {DYNAMICS_HIDDEN!r}
DECODER_HIDDEN = {DECODER_HIDDEN!r}
ACTIVATION = {ACTIVATION!r}
DROPOUT = {DROPOUT:.10g}


def safe_log10(arr: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(arr, LOG_EPS, None))


def _is_abund_col(name: str) -> bool:
    return name.startswith("abund_") and name.endswith("_dex")


def normalize_inputs(df) -> Tensor:
    X = df[INPUT_COLS].to_numpy(dtype=np.float64).copy()
    name_to_idx = {{c: i for i, c in enumerate(INPUT_COLS)}}
    if "T_K" in name_to_idx:
        X[:, name_to_idx["T_K"]] = X[:, name_to_idx["T_K"]] / TEMP_DIVISOR
    if "P_bar" in name_to_idx:
        X[:, name_to_idx["P_bar"]] = safe_log10(X[:, name_to_idx["P_bar"]]) / INPUT_LOG_SCALE
    if "fZ_dex" in name_to_idx:
        X[:, name_to_idx["fZ_dex"]] = X[:, name_to_idx["fZ_dex"]] / INPUT_LOG_SCALE
    if "fZ" in name_to_idx:
        X[:, name_to_idx["fZ"]] = safe_log10(X[:, name_to_idx["fZ"]]) / INPUT_LOG_SCALE
    for col, idx in name_to_idx.items():
        if _is_abund_col(col):
            X[:, idx] = (X[:, idx] - ABUND_EPSILON_OFFSET) / ABUND_DEX_SCALE
    return torch.as_tensor(X.astype(np.float32))


def denormalize_targets(y_scaled: np.ndarray) -> np.ndarray:
    return np.power(10.0, y_scaled * TARGET_LOG_SCALE)


def scale_targets_train_to_linear_torch(y_scaled: Tensor) -> Tensor:
    return torch.clamp(torch.pow(10.0, y_scaled * TARGET_LOG_SCALE), min=0.0)


def load_model(device: str | torch.device | None = None) -> FlowMapAutoencoder:
    device = torch.device(device) if device else torch.device("cpu")
    model = FlowMapAutoencoder(
        state_dim_in=len(TARGET_COLS),
        state_dim_out=len(TARGET_COLS),
        global_dim=len(INPUT_COLS),
        latent_dim=LATENT_DIM,
        encoder_hidden=ENCODER_HIDDEN,
        dynamics_hidden=DYNAMICS_HIDDEN,
        decoder_hidden=DECODER_HIDDEN,
        activation_name=ACTIVATION,
        dropout=DROPOUT,
        predict_delta=True,
        predict_delta_log_phys=False,
        softmax_head=False,
    ).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def forward_autoencoder(model: FlowMapAutoencoder, g: Tensor) -> Tensor:
    device = next(model.parameters()).device
    g = torch.as_tensor(g, dtype=torch.float32, device=device)
    y0 = torch.zeros((g.shape[0], len(TARGET_COLS)), dtype=g.dtype, device=device)
    dt = torch.ones((g.shape[0], 1), dtype=g.dtype, device=device)
    pred = model(y0, dt, g)
    return pred[:, 0, :]


if __name__ == "__main__":
    mdl = load_model()
    print("Loaded FlowMapAutoencoder with", sum(p.numel() for p in mdl.parameters()), "parameters")
"""
    module_path.write_text(code)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FlowMapAutoencoder with configurable loss function")
    parser.add_argument("--loss-type", type=str, default="huber", choices=["huber", "mse"],
                        help="Loss function: 'huber' (weighted) or 'mse' (plain MSE in normalized space)")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Output directory (default: runs_autoencoder_{dataset_tag})")
    args = parser.parse_args()
    
    # Update OUT_DIR if specified
    global OUT_DIR
    if args.run_dir:
        OUT_DIR = Path(args.run_dir)
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    log.info("Training with loss type: %s", args.loss_type)
    log.info("Output directory: %s", OUT_DIR)
    log.info("Loading CSV: %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    log.info("Loaded: %s rows × %s cols", fmt(len(df)), fmt(len(df.columns)))

    input_cols = resolve_input_columns(df)
    target_cols = resolve_target_columns(df, input_cols)

    X = normalize_inputs_df(df, input_cols)
    y_linear = df[target_cols].to_numpy(dtype=np.float64)
    y_scaled = scale_targets_linear_to_train(y_linear)

    weights = compute_target_weights(y_linear)
    log.info("Target weights (first 5): %s", [f"{w:.2f}" for w in weights[:5]])

    indices = np.arange(len(df))
    X_train, X_tmp, y_train, y_tmp, idx_train, idx_tmp = train_test_split(
        X, y_scaled, indices, train_size=TRAIN_FRAC, random_state=SEED, shuffle=True
    )
    val_ratio = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_tmp, y_tmp, idx_tmp, train_size=val_ratio, random_state=SEED + 1, shuffle=True
    )

    train_ds = AutoencoderDataset(X_train, y_train)
    val_ds = AutoencoderDataset(X_val, y_val)
    test_ds = AutoencoderDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    state_dim = len(target_cols)
    global_dim = X.shape[1]
    model = FlowMapAutoencoder(
        state_dim_in=state_dim,
        state_dim_out=state_dim,
        global_dim=global_dim,
        latent_dim=LATENT_DIM,
        encoder_hidden=ENCODER_HIDDEN,
        dynamics_hidden=DYNAMICS_HIDDEN,
        decoder_hidden=DECODER_HIDDEN,
        activation_name=ACTIVATION,
        dropout=DROPOUT,
        predict_delta=True,
        predict_delta_log_phys=False,
        softmax_head=False,
    )
    device = get_device()
    model.to(device)
    log.info(
        "Model ready (params: %s) — state_dim=%d global_dim=%d latent=%d",
        fmt(sum(p.numel() for p in model.parameters())),
        state_dim,
        global_dim,
        LATENT_DIM,
    )

    # Select loss function based on argument
    if args.loss_type == "mse":
        criterion = nn.MSELoss()
        log.info("Using MSE loss (plain) in normalized space")
    else:
        criterion = WeightedHuber(delta=0.02, weights=torch.as_tensor(weights, dtype=torch.float32, device=device))
        log.info("Using Weighted Huber loss (delta=0.02)")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    best_val = math.inf
    best_path = OUT_DIR / "best.pt"
    best_py = OUT_DIR / "best_model.py"
    
    # Track loss history for plotting
    loss_history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mse": [],
        "val_mae": [],
        "val_log_mae": []
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_losses = []
        start = time.time()
        for g, y in train_loader:
            g = g.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y0 = torch.zeros_like(y)
            dt = torch.ones((g.shape[0], 1), device=device, dtype=g.dtype)
            pred = model(y0, dt, g)[:, 0, :]
            loss = criterion(pred, y)
            loss.backward()
            if GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            ep_losses.append(loss.item())

        train_loss = float(np.mean(ep_losses))
        val_res = evaluate(model, val_loader, device)
        scheduler.step(val_res.loss)
        
        # Record history
        loss_history["epoch"].append(epoch)
        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_res.loss)
        loss_history["val_mse"].append(val_res.mse)
        loss_history["val_mae"].append(val_res.mae)
        loss_history["val_log_mae"].append(val_res.log_mae)
        
        log.info(
            "Epoch %03d | train=%.4f | val_loss=%.4f | val_MSE=%.4e | val_MAE=%.4e | val_LogMAE=%.4f | time=%.1fs",
            epoch,
            train_loss,
            val_res.loss,
            val_res.mse,
            val_res.mae,
            val_res.log_mae,
            time.time() - start,
        )

        if val_res.loss < best_val:
            best_val = val_res.loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "input_cols": input_cols,
                        "target_cols": target_cols,
                        "weights": weights.tolist(),
                        "splits": {
                            "train_idx": idx_train.tolist(),
                            "val_idx": idx_val.tolist(),
                            "test_idx": idx_test.tolist(),
                        },
                        "hyperparams": {
                            "latent_dim": LATENT_DIM,
                            "encoder_hidden": ENCODER_HIDDEN,
                            "dynamics_hidden": DYNAMICS_HIDDEN,
                            "decoder_hidden": DECODER_HIDDEN,
                            "activation": ACTIVATION,
                            "dropout": DROPOUT,
                        },
                    },
                },
                best_path,
            )
            log.info("  ↳ Saved new best checkpoint: %s", best_path)

    log.info("Training complete. Loading best checkpoint...")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_res = evaluate(model, test_loader, device)
    log.info(
        "Test metrics | loss=%.4f | MSE=%.4e | MAE=%.4e",
        test_res.loss,
        test_res.mse,
        test_res.mae,
    )
    
    # Save loss history to CSV
    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(OUT_DIR / "loss_history.csv", index=False)
    log.info("Loss history saved to %s", OUT_DIR / "loss_history.csv")

    splits = {
        "train_idx": idx_train.tolist(),
        "val_idx": idx_val.tolist(),
        "test_idx": idx_test.tolist(),
    }

    write_best_model_py(
        best_py,
        best_path,
        input_cols,
        target_cols,
        weights.tolist(),
        splits,
    )
    log.info("Exported inference helper → %s", best_py)

    summary = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "loss_type": args.loss_type,
        "val_loss": best_val,
        "test_loss": test_res.loss,
        "test_mse_linear": test_res.mse,
        "test_mae_linear": test_res.mae,
        "input_cols": input_cols,
        "target_cols": target_cols,
        "weights": weights.tolist(),
        "splits": splits,
        "best_model_py": str(best_py),
        "checkpoint_path": str(best_path),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary written to %s", OUT_DIR / "summary.json")


if __name__ == "__main__":
    main()

