#!/usr/bin/env python3
"""
train_autoencoder_improved.py
==============================

Improved training harness with four changes vs. baseline:
  1. AdamW optimizer (instead of Adam)
  2. Optional SimpleMLP architecture (in addition to FlowMap)
  3. Normalization (weights, target_cols) computed on TRAIN split only
  4. Correct loss naming (loss_linear not mse) and config-driven loss type

Otherwise identical to train_autoencoder.py. Use for baseline vs improved comparison.

Usage:
    python train_autoencoder_improved.py --config configs/x4800_improved.json \\
        --run-dir results/runs/runs_autoencoder_x4800_improved
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

from chemcalculations.autoencoder_model import FlowMapAutoencoder, SimpleMLP

from chemcalculations._paths import project_root as repo_root


# ---------------------------------------------------------------------------
# Configuration (mirrors train_autoencoder.py)
# ---------------------------------------------------------------------------

CSV_PATH = os.environ.get(
    "CSV_PATH",
    str(repo_root() / "data" / "datasets" / "all_gas_fastchem_x160.csv"),
)
OUT_DIR = Path("results/runs/runs_autoencoder")

SEED = 42
TRAIN_FRAC = 0.85
VAL_FRAC = 0.10
TEST_FRAC = 0.05
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-8

EPOCHS = 50
BATCH_SIZE = 512
LR = 5e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0

ARCHITECTURE_TYPE = "flowmap"
LATENT_DIM = 96
ENCODER_HIDDEN = [512, 512, 512]
DYNAMICS_HIDDEN = [512, 512, 512]
DECODER_HIDDEN = [512, 512, 512]
ACTIVATION = "silu"
DROPOUT = 0.0
MLP_HIDDEN = [1024, 1024, 1024, 1024, 1024, 1024]
MLP_DROPOUT = 0.05
OPTIMIZER = "adamw"
LOSS_TYPE = "huber"

TARGET_TOPK_SPECIES = 20
USE_STATIC_SPECIES_LIST = False
STATIC_SPECIES_LIST_PATH: Optional[str] = None

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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("autoencoder_improved")


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
    if USE_STATIC_SPECIES_LIST and STATIC_SPECIES_LIST_PATH:
        static_path = Path(STATIC_SPECIES_LIST_PATH)
        if not static_path.is_absolute():
            root = repo_root()
            if static_path.parent == Path("."):
                static_path = root / "configs" / static_path.name
            else:
                static_path = root / static_path
        if static_path.exists():
            try:
                with open(static_path, 'r') as f:
                    static_data = json.load(f)
                    static_species = static_data.get('species', [])
                available_species = [s for s in static_species if s in df.columns]
                missing_species = set(static_species) - set(available_species)
                if len(available_species) == len(static_species):
                    log.info("Using static species list (%d) from %s", len(available_species), static_path.name)
                    return available_species
                elif len(available_species) > 0:
                    log.warning("Using %d available from static list", len(available_species))
                    return available_species
            except Exception as e:
                log.warning("Static list failed: %s", e)
    candidates = species_candidates(df, input_cols)
    top = topk_species_linear_mean(df, candidates, TARGET_TOPK_SPECIES)
    if "e-" in df.columns and "e-" in candidates and "e-" not in top:
        top.insert(0, "e-")
    log.info("Target columns (%d) [dynamic]: %s", len(top), top[:10] + (["..."] if len(top) > 10 else []))
    return top


class AutoencoderDataset(Dataset):
    def __init__(self, g: np.ndarray, y: np.ndarray):
        self.g = torch.as_tensor(g, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.g.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.g[idx], self.y[idx]


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


class LogRatioLoss(nn.Module):
    def __init__(self, weights: Optional[torch.Tensor] = None, target_log_scale: float = 30.0):
        super().__init__()
        self.register_buffer("w", weights if weights is not None else None)
        self.target_log_scale = target_log_scale

    def forward(self, pred_scaled: torch.Tensor, true_scaled: torch.Tensor) -> torch.Tensor:
        log_ratio = self.target_log_scale * torch.abs(pred_scaled - true_scaled)
        if self.w is not None:
            log_ratio = log_ratio * self.w.view(1, -1)
        return log_ratio.mean()


def compute_target_weights(y_linear: np.ndarray, present_floor: float = 1e-8) -> np.ndarray:
    freq = (y_linear > present_floor).mean(axis=0)
    w = 1.0 / np.sqrt(np.clip(freq, 1e-6, None))
    return (w / w.mean()).astype(np.float32)


@dataclass
class EvalResult:
    loss: float
    loss_linear: float
    mae: float
    log_mae: float = 0.0
    log_r2: float = 0.0


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: Optional[nn.Module] = None) -> EvalResult:
    model.eval()
    if criterion is None:
        criterion = nn.MSELoss()
    losses, losses_linear, maes, log_maes = [], [], [], []
    all_y_log, all_pred_log = [], []

    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                g, y, y_linear_batch = batch_data
                y_linear_batch = y_linear_batch.to(device) if y_linear_batch is not None else None
            else:
                g, y = batch_data
                y_linear_batch = None
            g = g.to(device)
            y = y.to(device)
            y0 = torch.zeros_like(y)
            dt = torch.ones((g.shape[0], 1), device=device, dtype=g.dtype)
            pred = model(y0, dt, g)[:, 0, :]
            loss = criterion(pred, y)
            if y_linear_batch is not None:
                y_lin = y_linear_batch
            else:
                y_lin = scale_targets_train_to_linear_tensor(y)
            pred_lin = scale_targets_train_to_linear_tensor(pred)
            loss_linear = criterion(pred_lin, y_lin)
            mae = torch.mean((pred_lin - y_lin).abs())
            y_log = torch.log10(torch.clamp(y_lin, min=1e-30))
            pred_log = torch.log10(torch.clamp(pred_lin, min=1e-30))
            log_mae = torch.mean((pred_log - y_log).abs())
            all_y_log.append(y_log.cpu().numpy())
            all_pred_log.append(pred_log.cpu().numpy())
            losses.append(loss.item())
            losses_linear.append(loss_linear.item())
            maes.append(mae.item())
            log_maes.append(log_mae.item())

    try:
        from sklearn.metrics import r2_score
        log_r2 = float(r2_score(np.concatenate(all_y_log).flatten(), np.concatenate(all_pred_log).flatten()))
    except Exception:
        log_r2 = 0.0

    return EvalResult(
        loss=float(np.mean(losses)),
        loss_linear=float(np.mean(losses_linear)),
        mae=float(np.mean(maes)),
        log_mae=float(np.mean(log_maes)),
        log_r2=log_r2,
    )


def scale_targets_train_to_linear_tensor(y_scaled: torch.Tensor) -> torch.Tensor:
    return torch.clamp((10.0 ** (y_scaled * TARGET_LOG_SCALE)), min=0.0)


def write_best_model_py(
    module_path: Path,
    checkpoint_path: Path,
    input_cols: Sequence[str],
    target_cols: Sequence[str],
    weights: Sequence[float],
    splits: dict,
    architecture_type: str = "flowmap",
) -> None:
    if architecture_type == "mlp":
        load_body = f"""
    model = SimpleMLP(
        global_dim=len(INPUT_COLS),
        state_dim=len(TARGET_COLS),
        hidden_dims={MLP_HIDDEN!r},
        activation_name={ACTIVATION!r},
        dropout_p={MLP_DROPOUT:.10g},
    ).to(device)"""
        model_str = "SimpleMLP"
    else:
        load_body = f"""
    model = FlowMapAutoencoder(
        state_dim_in=len(TARGET_COLS),
        state_dim_out=len(TARGET_COLS),
        global_dim=len(INPUT_COLS),
        latent_dim={LATENT_DIM},
        encoder_hidden={ENCODER_HIDDEN!r},
        dynamics_hidden={DYNAMICS_HIDDEN!r},
        decoder_hidden={DECODER_HIDDEN!r},
        activation_name={ACTIVATION!r},
        dropout={DROPOUT:.10g},
        predict_delta=True,
        predict_delta_log_phys=False,
        softmax_head=False,
    ).to(device)"""
        model_str = "FlowMapAutoencoder"

    code = f'''#!/usr/bin/env python3
# Auto-generated by train_autoencoder_improved.py
from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from chemcalculations.autoencoder_model import FlowMapAutoencoder, SimpleMLP

from chemcalculations._paths import project_root as repo_root

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

def safe_log10(arr): return np.log10(np.clip(arr, LOG_EPS, None))
def _is_abund_col(name): return name.startswith("abund_") and name.endswith("_dex")

def normalize_inputs(df):
    X = df[INPUT_COLS].to_numpy(dtype=np.float64).copy()
    name_to_idx = {{c: i for i, c in enumerate(INPUT_COLS)}}
    if "T_K" in name_to_idx: X[:, name_to_idx["T_K"]] /= TEMP_DIVISOR
    if "P_bar" in name_to_idx: X[:, name_to_idx["P_bar"]] = safe_log10(X[:, name_to_idx["P_bar"]]) / INPUT_LOG_SCALE
    if "fZ_dex" in name_to_idx: X[:, name_to_idx["fZ_dex"]] /= INPUT_LOG_SCALE
    if "fZ" in name_to_idx: X[:, name_to_idx["fZ"]] = safe_log10(X[:, name_to_idx["fZ"]]) / INPUT_LOG_SCALE
    for col, idx in name_to_idx.items():
        if _is_abund_col(col): X[:, idx] = (X[:, idx] - ABUND_EPSILON_OFFSET) / ABUND_DEX_SCALE
    return torch.as_tensor(X.astype(np.float32))

def denormalize_targets(y_scaled): return np.power(10.0, y_scaled * TARGET_LOG_SCALE)
def scale_targets_train_to_linear_torch(y_scaled): return torch.clamp(torch.pow(10.0, y_scaled * TARGET_LOG_SCALE), min=0.0)

def load_model(device=None):
    device = torch.device(device) if device else torch.device("cpu")
{load_body}
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def forward_autoencoder(model, g):
    device = next(model.parameters()).device
    g = torch.as_tensor(g, dtype=torch.float32, device=device)
    y0 = torch.zeros((g.shape[0], len(TARGET_COLS)), dtype=g.dtype, device=device)
    dt = torch.ones((g.shape[0], 1), dtype=g.dtype, device=device)
    return model(y0, dt, g)[:, 0, :]
'''
    module_path.write_text(code)


def load_config(config_path: Optional[str] = None) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        log.info("Loaded config: %s", config_path)
        return config
    return {}


def apply_config(config: dict) -> None:
    global CSV_PATH, OUT_DIR, SEED, TRAIN_FRAC, VAL_FRAC, TEST_FRAC
    global EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, GRAD_CLIP
    global LATENT_DIM, ENCODER_HIDDEN, DYNAMICS_HIDDEN, DECODER_HIDDEN, ACTIVATION, DROPOUT
    global ARCHITECTURE_TYPE, MLP_HIDDEN, MLP_DROPOUT, OPTIMIZER, LOSS_TYPE
    global TARGET_TOPK_SPECIES, TEMP_DIVISOR, INPUT_LOG_SCALE, ABUND_EPSILON_OFFSET
    global ABUND_DEX_SCALE, TARGET_ZERO_FLOOR, TARGET_LOG_SCALE, LOG_EPS, INCLUDE_FZ_AS_FEATURE
    global USE_STATIC_SPECIES_LIST, STATIC_SPECIES_LIST_PATH
    if not config:
        return
    if "data" in config:
        d = config["data"]
        if "csv_path" in d:
            CSV_PATH = os.environ.get("CSV_PATH", str(repo_root() / d["csv_path"]))
        if "train_frac" in d: TRAIN_FRAC = d["train_frac"]
        if "val_frac" in d: VAL_FRAC = d["val_frac"]
        if "test_frac" in d: TEST_FRAC = d["test_frac"]
        if "target_topk_species" in d: TARGET_TOPK_SPECIES = d["target_topk_species"]
        if "include_fz_as_feature" in d: INCLUDE_FZ_AS_FEATURE = d["include_fz_as_feature"]
        if "use_static_species_list" in d: USE_STATIC_SPECIES_LIST = d["use_static_species_list"]
        if "static_species_list_path" in d: STATIC_SPECIES_LIST_PATH = d["static_species_list_path"]
    if "optimization" in config:
        o = config["optimization"]
        if "optimizer" in o: OPTIMIZER = o["optimizer"].lower()
        if "epochs" in o: EPOCHS = o["epochs"]
        if "batch_size" in o: BATCH_SIZE = o["batch_size"]
        if "learning_rate" in o: LR = o["learning_rate"]
        if "weight_decay" in o: WEIGHT_DECAY = o["weight_decay"]
        if "grad_clip" in o: GRAD_CLIP = o["grad_clip"]
        if "seed" in o: SEED = o["seed"]
    if "architecture" in config:
        a = config["architecture"]
        if "type" in a: ARCHITECTURE_TYPE = a["type"].lower()
        if "latent_dim" in a: LATENT_DIM = a["latent_dim"]
        if "encoder_hidden" in a: ENCODER_HIDDEN = a["encoder_hidden"]
        if "dynamics_hidden" in a: DYNAMICS_HIDDEN = a["dynamics_hidden"]
        if "decoder_hidden" in a: DECODER_HIDDEN = a["decoder_hidden"]
        if "activation" in a: ACTIVATION = a["activation"]
        if "dropout" in a: DROPOUT = a["dropout"]
        if "mlp_hidden" in a: MLP_HIDDEN = a["mlp_hidden"]
        if "mlp_dropout" in a: MLP_DROPOUT = a["mlp_dropout"]
    if "normalization" in config:
        n = config["normalization"]
        if "temp_divisor" in n: TEMP_DIVISOR = n["temp_divisor"]
        if "input_log_scale" in n: INPUT_LOG_SCALE = n["input_log_scale"]
        if "abund_epsilon_offset" in n: ABUND_EPSILON_OFFSET = n["abund_epsilon_offset"]
        if "abund_dex_scale" in n: ABUND_DEX_SCALE = n["abund_dex_scale"]
        if "target_zero_floor" in n: TARGET_ZERO_FLOOR = n["target_zero_floor"]
        if "target_log_scale" in n: TARGET_LOG_SCALE = n["target_log_scale"]
        if "log_eps" in n: LOG_EPS = n["log_eps"]
    if "loss" in config and "type" in config["loss"]:
        LOSS_TYPE = config["loss"]["type"].lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train improved model (AdamW, train-only norm, etc.)")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--loss-type", type=str, default=None, help="Override loss type")
    parser.add_argument("--run-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_config(config)
    global OUT_DIR
    OUT_DIR = Path(args.run_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    loss_type = (args.loss_type or LOSS_TYPE).lower()
    log.info("IMPROVED trainer: AdamW, train-only norm, loss=%s", loss_type)
    log.info("Output: %s", OUT_DIR)
    log.info("Loading: %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    log.info("Loaded %s rows", fmt(len(df)))

    input_cols = resolve_input_columns(df)
    indices = np.arange(len(df))
    idx_train, idx_tmp = train_test_split(indices, train_size=TRAIN_FRAC, random_state=SEED, shuffle=True)
    val_ratio = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    idx_val, idx_test = train_test_split(idx_tmp, train_size=val_ratio, random_state=SEED + 1, shuffle=True)
    df_train = df.iloc[idx_train]

    target_cols = resolve_target_columns(df_train, input_cols)
    X = normalize_inputs_df(df, input_cols)
    y_linear = df[target_cols].to_numpy(dtype=np.float64)
    y_scaled = scale_targets_linear_to_train(y_linear)
    y_train_linear = y_linear[idx_train]
    weights = compute_target_weights(y_train_linear)
    log.info("Weights from train split only (first 5): %s", [f"{w:.2f}" for w in weights[:5]])

    X_train = X[idx_train]
    X_tmp = X[idx_tmp]
    y_train = y_scaled[idx_train]
    y_tmp = y_scaled[idx_tmp]
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
    if ARCHITECTURE_TYPE == "mlp":
        model = SimpleMLP(global_dim=global_dim, state_dim=state_dim, hidden_dims=MLP_HIDDEN,
                          activation_name=ACTIVATION, dropout_p=MLP_DROPOUT)
        log.info("SimpleMLP: %s hidden", MLP_HIDDEN)
    else:
        model = FlowMapAutoencoder(
            state_dim_in=state_dim, state_dim_out=state_dim, global_dim=global_dim,
            latent_dim=LATENT_DIM, encoder_hidden=ENCODER_HIDDEN, dynamics_hidden=DYNAMICS_HIDDEN,
            decoder_hidden=DECODER_HIDDEN, activation_name=ACTIVATION, dropout=DROPOUT,
            predict_delta=True, predict_delta_log_phys=False, softmax_head=False,
        )
    device = get_device()
    model.to(device)

    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "log_ratio":
        criterion = LogRatioLoss(weights=torch.as_tensor(weights, dtype=torch.float32, device=device), target_log_scale=TARGET_LOG_SCALE)
    else:
        criterion = WeightedHuber(delta=0.02, weights=torch.as_tensor(weights, dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    best_val = math.inf
    best_path = OUT_DIR / "best.pt"
    best_py = OUT_DIR / "best_model.py"
    loss_history = {"epoch": [], "train_loss": [], "val_loss": [], "val_loss_linear": [], "val_mae": [], "val_log_mae": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_losses = []
        start = time.time()
        for g, y in train_loader:
            g, y = g.to(device), y.to(device)
            optimizer.zero_grad()
            y0 = torch.zeros_like(y)
            dt = torch.ones((g.shape[0], 1), device=device, dtype=g.dtype)
            loss = criterion(model(y0, dt, g)[:, 0, :], y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            if GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            ep_losses.append(loss.item())
        train_loss = float(np.mean(ep_losses))
        val_res = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_res.loss)
        loss_history["epoch"].append(epoch)
        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_res.loss)
        loss_history["val_loss_linear"].append(val_res.loss_linear)
        loss_history["val_mae"].append(val_res.mae)
        loss_history["val_log_mae"].append(val_res.log_mae)
        log.info("Epoch %03d | train=%.4f | val=%.4f | val_lin=%.4e | LogMAE=%.4f | %.1fs",
                 epoch, train_loss, val_res.loss, val_res.loss_linear, val_res.log_mae, time.time() - start)
        if val_res.loss < best_val:
            best_val = val_res.loss
            torch.save({"model_state": model.state_dict(), "config": {"input_cols": input_cols, "target_cols": target_cols, "weights": weights.tolist(),
                        "splits": {"train_idx": idx_train.tolist(), "val_idx": idx_val.tolist(), "test_idx": idx_test.tolist()},
                        "architecture_type": ARCHITECTURE_TYPE, "hyperparams": {"latent_dim": LATENT_DIM, "encoder_hidden": ENCODER_HIDDEN,
                        "dynamics_hidden": DYNAMICS_HIDDEN, "decoder_hidden": DECODER_HIDDEN, "activation": ACTIVATION, "dropout": DROPOUT,
                        "mlp_hidden": MLP_HIDDEN, "mlp_dropout": MLP_DROPOUT}}}, best_path)
            log.info("  Saved best")

    log.info("Loading best...")
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    test_res = evaluate(model, test_loader, device, criterion)
    log.info("Test | loss=%.4f | loss_linear=%.4e | MAE=%.4e | Log MAE=%.4f | Log R²=%.6f",
             test_res.loss, test_res.loss_linear, test_res.mae, test_res.log_mae, test_res.log_r2)

    pd.DataFrame(loss_history).to_csv(OUT_DIR / "loss_history.csv", index=False)
    splits = {"train_idx": idx_train.tolist(), "val_idx": idx_val.tolist(), "test_idx": idx_test.tolist()}
    write_best_model_py(best_py, best_path, input_cols, target_cols, weights.tolist(), splits, architecture_type=ARCHITECTURE_TYPE)

    summary = {
        "train_samples": len(train_ds), "val_samples": len(val_ds), "test_samples": len(test_ds),
        "loss_type": loss_type, "val_loss": best_val, "test_loss": test_res.loss,
        "test_loss_linear": test_res.loss_linear, "test_mae_linear": test_res.mae,
        "test_log_mae": test_res.log_mae, "test_log_r2": test_res.log_r2,
        "input_cols": input_cols, "target_cols": target_cols, "weights": weights.tolist(),
        "splits": splits, "best_model_py": str(best_py), "checkpoint_path": str(best_path),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary: %s", OUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
