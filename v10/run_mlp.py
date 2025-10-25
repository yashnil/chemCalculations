#!/usr/bin/env python3
# run_mlp_grid.py — Plain MLP for the NEW FastChem grid CSV
# Accepts new structure (T_K, P_bar, fZ/fZ_dex, abund_<Elem>_dex, species columns), auto-detects features.
# Targets = Top-K most abundant species (+ always include electron 'e-' if present).
# Loss is computed in the full, scaled target space.
#
# IMPORTANT CHANGE:
#   - Any row that yields a non-finite in normalized inputs or scaled targets is DROPPED.
#   - We log an error summary with counts and sample indices instead of replacing values.
#   - TARGET_TOPK_SPECIES controls how many species are chosen as targets (excluding the optional 'e-' add-on).

import os
import time
import math
import logging
from typing import List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
except Exception as e:
    raise RuntimeError("scikit-learn is required for splitting. Please install scikit-learn.") from e


# =============================================================================
# PATHS / OUT
# =============================================================================
CSV_PATH: str = '/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv'
OUT_DIR:  str = "runs_mlp_v10"
# If present, this file should contain a 'species' column with names ranked by abundance.
# We'll take the first TARGET_TOPK_SPECIES from it (filtered to those present in the current CSV),
# and then fill from computed ranking if needed.
TOP20_FILE: str = os.path.join(OUT_DIR, "top20_species_by_abundance.csv")


# =============================================================================
# RUNTIME
# =============================================================================
SEED: int = 1337
DEVICE_FALLBACK: str = "cuda"   # preferred accelerator if available (no MPS)
NUM_WORKERS: int = 0


# =============================================================================
# DATA SPLIT (85/10/5)
# =============================================================================
TRAIN_FRAC: float = 0.85
VAL_FRAC: float   = 0.10
TEST_FRAC: float  = 0.05
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-8
USE_GROUP_SPLIT: bool = False  # grid rows are independent; set True only if you add a group column


# =============================================================================
# OPTIMIZATION
# =============================================================================
EPOCHS: int = 200
BATCH_SIZE: int = 512
LR: float = 5.0e-04
WEIGHT_DECAY: float = 1.0e-05
ETA_MIN: float = 1.0e-06  # cosine annealing min lr
GRAD_CLIP_NORM: Optional[float] = 5.0


# =============================================================================
# MODEL
# =============================================================================
HIDDEN: int = 512
DEPTH: int = 3
ACTIVATION: str = "leaky_relu"  # relu | gelu | tanh | sigmoid | leaky_relu
DROPOUT: float = 0.05


# =============================================================================
# NORMALIZATION (inputs/targets)
# =============================================================================
# Inputs:
TEMP_DIVISOR: float = 4.0e3     # T_K / 4000
INPUT_LOG_SCALE: float = 1.0e1  # divide log10-like inputs by 10
# Abundance (epsilon) columns are in dex already (12 + log10(N_el/N_H)):
ABUND_EPSILON_OFFSET: float = 12.0   # center epsilon by subtracting 12
ABUND_DEX_SCALE: float = 1.0e1       # then divide by 10

# Use fZ_dex (preferred) if present; else compute log10(fZ)
INCLUDE_FZ_AS_FEATURE: bool = True

# Targets:
TARGET_ZERO_FLOOR: float = 1.0e-30   # clamp below to 0, then log10(all)
TARGET_LOG_SCALE: float = 3.0e1
LOG_EPS: float = 1.0e-30             # safe guard for log10(x)


# =============================================================================
# TARGET SELECTION
# =============================================================================
# How many species to use as targets (by mean linear abundance).
# NOTE: If a column named 'e-' (electron number density) exists, it is always ADDED on top.
TARGET_TOPK_SPECIES: int = 20


# =============================================================================
# COLUMN LOGIC
# =============================================================================
# Manual override for inputs: set to a list of column names to force exact inputs, else None to auto-detect.
INPUT_COLS_MANUAL: Optional[List[str]] = None

# Manual override for targets: set to a list of species to force exact targets, else None to auto-select top-K (+ e-)
TARGET_COLS_MANUAL: Optional[List[str]] = None

# Hard exclusions (never be targets)
NEVER_TARGET_COLS: List[str] = [
    "T_K", "P_bar", "fZ", "fZ_dex", "flag", "flag_msg",
    "mean_molecular_weight", "total_element_density",
]

# IMPORTANT: we do NOT exclude abund_e-_dex from inputs; all abund_*_dex are allowed.


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mlp_grid")


# =============================================================================
# UTILS
# =============================================================================
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

def get_device(preferred: str = DEVICE_FALLBACK) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name in ("leaky_relu", "leakyrelu", "lrelu"):
        return nn.LeakyReLU(negative_slope=1.0e-02)
    raise ValueError(f"Unknown activation: {name}")

def safe_log10_np(arr: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(arr, LOG_EPS, None))

def _is_abund_col(c: str) -> bool:
    # e.g., abund_O_dex, abund_Si_dex, abund_e-_dex
    return c.startswith("abund_") and c.endswith("_dex")


# =============================================================================
# DATASET
# =============================================================================
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y_scaled_full: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y_scaled_full, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# =============================================================================
# FEATURE/TARGET RESOLUTION FOR THE NEW CSV
# =============================================================================
def resolve_input_columns(df: pd.DataFrame) -> List[str]:
    if INPUT_COLS_MANUAL is not None:
        cols = [c for c in INPUT_COLS_MANUAL if c in df.columns]
        missing = [c for c in INPUT_COLS_MANUAL if c not in df.columns]
        if missing:
            log.warning("Manual INPUT columns missing and skipped: %s", missing)
        if not cols:
            raise ValueError("No valid manual INPUT columns present.")
        log.info("Using manual INPUT columns (%d): %s", len(cols), cols[:10] + (["..."] if len(cols) > 10 else []))
        return cols

    cols: List[str] = []

    # Core: T and P
    if "T_K" not in df.columns or "P_bar" not in df.columns:
        raise ValueError("Expected columns 'T_K' and 'P_bar' not found in CSV.")
    cols += ["T_K", "P_bar"]

    # fZ or fZ_dex
    if INCLUDE_FZ_AS_FEATURE:
        if "fZ_dex" in df.columns:
            cols.append("fZ_dex")
        elif "fZ" in df.columns:
            cols.append("fZ")

    # Abundance epsilon columns — include ALL, including abund_e-_dex
    abund_cols = sorted([c for c in df.columns if _is_abund_col(c)])
    if not abund_cols:
        log.warning("No 'abund_*_dex' columns found; proceeding without elemental abundances.")
    cols += abund_cols

    log.info("Resolved INPUT columns (%d): %s", len(cols), cols[:10] + (["..."] if len(cols) > 10 else []))
    return cols


def _species_candidates(df: pd.DataFrame, input_cols: Sequence[str]) -> List[str]:
    """Numeric columns not in inputs/NEVER_TARGET_COLS and not abund_*_dex are species candidates."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set(input_cols) | set(NEVER_TARGET_COLS)
    return [c for c in numeric_cols if (c not in exclude and not _is_abund_col(c))]

def _topk_species_linear_mean(
    df: pd.DataFrame, candidates: List[str], k: int
) -> List[str]:
    """Compute mean in *linear* space, treating NaNs/±Inf as drop (via nanmean after clipping)."""
    if not candidates or k <= 0:
        return []
    means = []
    for c in candidates:
        vals = df[c].to_numpy(dtype=float, copy=False)
        vals = np.where(np.isfinite(vals), np.clip(vals, 0.0, None), np.nan)  # negatives->0; NaN/Inf->nan
        mean_val = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else 0.0
        means.append((mean_val, c))
    means.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in means[:k]]

def _read_top_from_file(path: str, candidates: List[str], k: int) -> List[str]:
    """Try to read a ranked species list from CSV (col 'species'), keep only candidates, take first k."""
    try:
        tdf = pd.read_csv(path)
        listed = [str(s) for s in tdf["species"].tolist()]
        valid = [s for s in listed if s in candidates]
        return valid[:k]
    except Exception:
        return []

def resolve_target_columns(df: pd.DataFrame, input_cols: Sequence[str]) -> List[str]:
    if TARGET_COLS_MANUAL is not None:
        req = []
        seen = set()
        for c in TARGET_COLS_MANUAL:
            if c in df.columns and c not in seen:
                req.append(c)
                seen.add(c)
        # filter
        num_cols = set(df.select_dtypes(include=[np.number]).columns)
        exclude = set(input_cols) | set(NEVER_TARGET_COLS)
        targets = [c for c in req if (c in num_cols and c not in exclude and not _is_abund_col(c))]
        if not targets:
            raise ValueError("No valid manual TARGET columns present after filtering.")
        log.info("Using manual TARGET columns (%d)", len(targets))
        return targets

    # Auto: top-K by mean abundance (linear) + always include electron 'e-' if present
    candidates = _species_candidates(df, input_cols)

    # Try to read from file; if not enough, fill from computed ranking
    top_from_file = _read_top_from_file(TOP20_FILE, candidates, TARGET_TOPK_SPECIES) if os.path.isfile(TOP20_FILE) else []
    need = max(0, TARGET_TOPK_SPECIES - len(top_from_file))
    computed_fill = _topk_species_linear_mean(df, [c for c in candidates if c not in top_from_file], need) if need > 0 else []
    top = top_from_file + computed_fill

    # Always include electron species 'e-' if present as a named column
    forced = []
    if "e-" in df.columns and "e-" in candidates:
        forced.append("e-")

    # Build final list preserving order and uniqueness
    seen = set()
    final_targets: List[str] = []
    for c in forced + top:
        if c not in seen:
            final_targets.append(c)
            seen.add(c)

    if not final_targets:
        raise ValueError("No valid auto-detected TARGET columns. Check CSV and exclusions.")

    log.info("Resolved TARGET columns (%d) [topK=%d%s]: %s",
             len(final_targets),
             TARGET_TOPK_SPECIES,
             " + e-" if ("e-" in forced) else "",
             final_targets[:10] + (["..."] if len(final_targets) > 10 else []))
    return final_targets


# =============================================================================
# NORMALIZATION + VALIDATION (NO SANITIZING)
# =============================================================================
def normalize_inputs_df(df: pd.DataFrame, input_cols: Sequence[str]) -> np.ndarray:
    """Return normalized inputs WITHOUT sanitizing; may contain NaN/Inf if source is bad."""
    X = df[input_cols].copy().to_numpy(dtype=np.float64)
    name_to_idx = {c: i for i, c in enumerate(input_cols)}
    # T_K
    if "T_K" in name_to_idx:
        X[:, name_to_idx["T_K"]] = X[:, name_to_idx["T_K"]] / TEMP_DIVISOR
    # P_bar
    if "P_bar" in name_to_idx:
        X[:, name_to_idx["P_bar"]] = safe_log10_np(X[:, name_to_idx["P_bar"]]) / INPUT_LOG_SCALE
    # fZ / fZ_dex
    if "fZ_dex" in name_to_idx:
        X[:, name_to_idx["fZ_dex"]] = X[:, name_to_idx["fZ_dex"]] / INPUT_LOG_SCALE
    if "fZ" in name_to_idx:
        X[:, name_to_idx["fZ"]] = safe_log10_np(X[:, name_to_idx["fZ"]]) / INPUT_LOG_SCALE
    # Abundances (epsilon dex) — includes abund_e-_dex
    for c, i in name_to_idx.items():
        if _is_abund_col(c):
            X[:, i] = (X[:, i] - ABUND_EPSILON_OFFSET) / ABUND_DEX_SCALE
    return X

def scale_targets_linear_to_train(y_linear: np.ndarray) -> np.ndarray:
    """Transform targets to training space WITHOUT sanitizing; may contain NaN/Inf if source is bad."""
    y = y_linear.copy()
    y[y < TARGET_ZERO_FLOOR] = 0.0
    y = safe_log10_np(y) / TARGET_LOG_SCALE
    return y

def _log_nonfinite_summary(tag: str, arr: np.ndarray, col_names: Sequence[str], max_to_show: int = 10) -> None:
    """Log per-column counts of non-finites and up to N sample row indices with any non-finite."""
    mask_row_bad = ~np.isfinite(arr).all(axis=1)
    n_bad = int(mask_row_bad.sum())
    if n_bad == 0:
        return
    bad_rows = np.where(mask_row_bad)[0]
    log.error("[%s] Non-finite rows detected: %d", tag, n_bad)
    if bad_rows.size:
        log.error("[%s] Example bad row indices: %s", tag, bad_rows[:max_to_show].tolist())
    # Per-column summary
    bad_per_col = (~np.isfinite(arr)).sum(axis=0)
    # Report only columns with non-zero bad counts
    offenders = [(col_names[i], int(cnt)) for i, cnt in enumerate(bad_per_col) if cnt > 0]
    offenders.sort(key=lambda t: t[1], reverse=True)
    offenders_show = offenders[:max_to_show]
    if offenders_show:
        log.error("[%s] Non-finite counts by column (top %d): %s", tag, max_to_show, offenders_show)

def _drop_bad_rows_and_report(X: np.ndarray, X_cols: Sequence[str],
                              y_scaled: np.ndarray, y_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return filtered (X, y, keep_idx). Logs errors and drops any row with non-finite in inputs or targets."""
    mask_inputs_ok = np.isfinite(X).all(axis=1)
    mask_targets_ok = np.isfinite(y_scaled).all(axis=1)

    # Log summaries before dropping
    _log_nonfinite_summary("INPUTS", X, X_cols)
    _log_nonfinite_summary("TARGETS", y_scaled, y_cols)

    keep_mask = mask_inputs_ok & mask_targets_ok
    dropped = int((~keep_mask).sum())
    if dropped > 0:
        bad_idx = np.where(~keep_mask)[0]
        log.error("Dropping %d row(s) due to non-finite values in inputs/targets. Examples: %s",
                  dropped, bad_idx[:10].tolist())
    keep_idx = np.where(keep_mask)[0]
    return X[keep_mask], y_scaled[keep_mask], keep_idx


# =============================================================================
# MODEL
# =============================================================================
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int, activation: str, dropout: float):
        super().__init__()
        act = get_activation(activation)
        layers: List[nn.Module] = []
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden), act, nn.Dropout(p=dropout)]
            last = hidden
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# TRAIN / EVAL
# =============================================================================
def train_one_epoch(model, optimizer, loader, device, loss_fn) -> float:
    model.train()
    vals = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        yp = model(xb)
        loss = loss_fn(yp, yb)
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("Encountered non-finite loss during training step — check filtered data.")
        loss.backward()
        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        vals.append(loss.detach().item())
    return float(np.mean(vals)) if vals else float("nan")

@torch.no_grad()
def evaluate(model, loader, device, loss_fn) -> float:
    model.eval()
    vals = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yp = model(xb)
        loss = loss_fn(yp, yb)
        vals.append(loss.detach().item())
    return float(np.mean(vals)) if vals else float("nan")


# =============================================================================
# EMIT SELF-CONTAINED INFERENCE MODULE
# =============================================================================
def write_best_model_py(
    module_path: str,
    checkpoint_path: str,
    input_cols: List[str],
    target_cols: List[str],
    hidden: int,
    depth: int,
    activation: str,
    dropout: float,
) -> None:
    code = f"""#!/usr/bin/env python3
# Auto-generated by run_mlp_grid.py at {time.strftime("%Y-%m-%d %H:%M:%S")}
# Reconstructs the plain MLP and loads weights. Forward returns FULL-SIZE SCALED TARGETS.

import numpy as np
import torch
from torch import nn

INPUT_COLS = {input_cols!r}
TARGET_COLS = {target_cols!r}

HIDDEN = {hidden}
DEPTH = {depth}
ACTIVATION = {activation!r}
DROPOUT = {dropout:.10g}
CHECKPOINT_PATH = {checkpoint_path!r}

# Normalization constants (must match training):
TEMP_DIVISOR = {TEMP_DIVISOR:.10g}
INPUT_LOG_SCALE = {INPUT_LOG_SCALE:.10g}
ABUND_EPSILON_OFFSET = {ABUND_EPSILON_OFFSET:.10g}
ABUND_DEX_SCALE = {ABUND_DEX_SCALE:.10g}
TARGET_ZERO_FLOOR = {TARGET_ZERO_FLOOR:.10g}
TARGET_LOG_SCALE = {TARGET_LOG_SCALE:.10g}
LOG_EPS = {LOG_EPS:.10g}

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu": return nn.ReLU()
    if name == "gelu": return nn.GELU()
    if name == "tanh": return nn.Tanh()
    if name == "sigmoid": return nn.Sigmoid()
    if name in ("leaky_relu","leakyrelu","lrelu"): return nn.LeakyReLU(negative_slope=1.0e-02)
    raise ValueError(f"Unknown activation: {{name}}")

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int, activation: str, dropout: float):
        super().__init__()
        act = get_activation(activation)
        layers = []
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden), act, nn.Dropout(p=dropout)]
            last = hidden
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def safe_log10_np(arr: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(arr, LOG_EPS, None))

def _is_abund_col(c: str) -> bool:
    return c.startswith("abund_") and c.endswith("_dex")

def normalize_inputs(df) -> torch.Tensor:
    X = df[INPUT_COLS].copy().to_numpy(dtype=np.float64)
    name_to_idx = {{c:i for i,c in enumerate(INPUT_COLS)}}
    if "T_K" in name_to_idx:
        X[:, name_to_idx["T_K"]] = X[:, name_to_idx["T_K"]] / TEMP_DIVISOR
    if "P_bar" in name_to_idx:
        X[:, name_to_idx["P_bar"]] = safe_log10_np(X[:, name_to_idx["P_bar"]]) / INPUT_LOG_SCALE
    if "fZ_dex" in name_to_idx:
        X[:, name_to_idx["fZ_dex"]] = X[:, name_to_idx["fZ_dex"]] / INPUT_LOG_SCALE
    if "fZ" in name_to_idx:
        X[:, name_to_idx["fZ"]] = safe_log10_np(X[:, name_to_idx["fZ"]]) / INPUT_LOG_SCALE
    for c,i in name_to_idx.items():
        if _is_abund_col(c):
            X[:, i] = (X[:, i] - ABUND_EPSILON_OFFSET) / ABUND_DEX_SCALE
    return torch.as_tensor(X.astype(np.float32))

def denormalize_targets(y_scaled: np.ndarray) -> np.ndarray:
    # inverse of: y_scaled = log10(max(y, LOG_EPS)) / TARGET_LOG_SCALE
    return np.power(10.0, y_scaled * TARGET_LOG_SCALE)

def load_model(device: str | None = None) -> nn.Module:
    device = torch.device(device) if device else torch.device("cpu")
    in_dim = len(INPUT_COLS); out_dim = len(TARGET_COLS)
    m = MLP(in_dim, out_dim, HIDDEN, DEPTH, ACTIVATION, DROPOUT).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m

if __name__ == "__main__":
    m = load_model()
    print("Loaded MLP with", sum(p.numel() for p in m.parameters()), "parameters")
"""
    with open(module_path, "w", encoding="utf-8") as f:
        f.write(code)


# =============================================================================
# SPLITTING
# =============================================================================
def three_way_split_indices(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    use_group_split: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    if use_group_split and groups is not None:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
        trainval_idx, test_idx = next(gss1.split(X, y, groups=groups))
        inner_val_frac = VAL_FRAC / (1.0 - TEST_FRAC)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=inner_val_frac, random_state=seed)
        train_rel, val_rel = next(gss2.split(X[trainval_idx], y[trainval_idx], groups=groups[trainval_idx]))
        train_idx = trainval_idx[train_rel]
        val_idx   = trainval_idx[val_rel]
    else:
        idx_all = np.arange(n)
        trainval_idx, test_idx = train_test_split(idx_all, test_size=TEST_FRAC, random_state=seed, shuffle=True)
        inner_val_frac = VAL_FRAC / (1.0 - TEST_FRAC)
        train_idx, val_idx = train_test_split(trainval_idx, test_size=inner_val_frac, random_state=seed, shuffle=True)
    return train_idx, val_idx, test_idx


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Summary
    log.info("Device: %s", device.type)
    log.info("SPLITS: train=%.1f%% | val=%.1f%% | test=%.1f%% | group_split=%s",
             TRAIN_FRAC*100, VAL_FRAC*100, TEST_FRAC*100, USE_GROUP_SPLIT)
    log.info("MODEL : depth=%d hidden=%d act=%s dropout=%.3f", DEPTH, HIDDEN, ACTIVATION, DROPOUT)
    log.info("NORM  : T/%.0f | log10(P)/%.0f | fZ*_div=%.0f | abund: (eps-%.1f)/%.0f | target log10/%.0f",
             TEMP_DIVISOR, INPUT_LOG_SCALE, INPUT_LOG_SCALE, ABUND_EPSILON_OFFSET, ABUND_DEX_SCALE, TARGET_LOG_SCALE)
    log.info("TARGETS: topK=%d (plus 'e-' if present)", TARGET_TOPK_SPECIES)

    # Load
    log.info("Loading CSV: %s", os.path.abspath(CSV_PATH))
    df = pd.read_csv(CSV_PATH)
    log.info("Loaded: %d rows × %d cols", df.shape[0], df.shape[1])

    # Resolve columns
    input_cols = resolve_input_columns(df)
    target_cols = resolve_target_columns(df, input_cols)

    # Prepare arrays (NO sanitizing)
    X_np_raw = normalize_inputs_df(df, input_cols)
    y_linear = df[target_cols].to_numpy(dtype=np.float64)
    y_scaled_raw = scale_targets_linear_to_train(y_linear)

    # Drop any row with non-finite values in inputs/targets
    X_np, y_scaled, keep_idx = _drop_bad_rows_and_report(X_np_raw, input_cols, y_scaled_raw, target_cols)
    if len(keep_idx) == 0:
        raise RuntimeError("All rows were dropped due to non-finite inputs/targets — aborting.")

    # Optional groups (off by default) — align with kept rows if present
    groups = None
    if USE_GROUP_SPLIT:
        if "group_index" in df.columns:
            groups = df["group_index"].to_numpy()[keep_idx]
        elif "point_index" in df.columns:
            groups = df["point_index"].to_numpy()[keep_idx]

    # Split
    train_idx, val_idx, test_idx = three_way_split_indices(
        X_np, y_scaled, groups, USE_GROUP_SPLIT and (groups is not None), SEED
    )
    log.info("Split sizes (after filtering): Train=%d | Val=%d | Test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Persist split indices (relative to the FILTERED dataset)
    split_path = os.path.join(OUT_DIR, "split_indices.npz")
    np.savez_compressed(split_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=SEED)
    log.info("Saved split indices: %s", split_path)

    # Tensors
    X_train = X_np.astype(np.float32)[train_idx]
    X_val   = X_np.astype(np.float32)[val_idx]
    X_test  = X_np.astype(np.float32)[test_idx]
    y_train = y_scaled.astype(np.float32)[train_idx]
    y_val   = y_scaled.astype(np.float32)[val_idx]
    y_test  = y_scaled.astype(np.float32)[test_idx]

    # Model
    in_dim  = X_train.shape[1]
    out_dim = y_train.shape[1]
    model = MLP(in_dim=in_dim, out_dim=out_dim, hidden=HIDDEN, depth=DEPTH,
                activation=ACTIVATION, dropout=DROPOUT).to(device)
    log.info("Model params: %s | in=%d out=%d", fmt(count_parameters(model)), in_dim, out_dim)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=ETA_MIN)
    loss_fn = nn.MSELoss()

    best_val = math.inf
    best_epoch = -1
    best_path = os.path.join(OUT_DIR, "best.pt")
    last_path = os.path.join(OUT_DIR, "last.pt")
    best_py   = os.path.join(OUT_DIR, "best_model.py")

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_mse = train_one_epoch(
            model, opt,
            DataLoader(TabDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
            device, loss_fn
        )
        val_mse = evaluate(
            model,
            DataLoader(TabDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
            device, loss_fn
        )
        sch.step()

        is_best = val_mse < best_val
        if is_best:
            best_val = val_mse
            best_epoch = epoch
            cfg = {
                "input_cols": input_cols,
                "target_cols": target_cols,
                "temp_divisor": TEMP_DIVISOR,
                "input_log_scale": INPUT_LOG_SCALE,
                "abund_epsilon_offset": ABUND_EPSILON_OFFSET,
                "abund_dex_scale": ABUND_DEX_SCALE,
                "target_log_scale": TARGET_LOG_SCALE,
                "log_eps": LOG_EPS,
                "target_zero_floor": TARGET_ZERO_FLOOR,
                "hidden": HIDDEN,
                "depth": DEPTH,
                "activation": ACTIVATION,
                "dropout": DROPOUT,
                "splits": {
                    "train_idx": train_idx.tolist(),
                    "val_idx": val_idx.tolist(),
                    "test_idx": test_idx.tolist(),
                },
                "out_dim": out_dim,
                "topk_species": TARGET_TOPK_SPECIES,
            }
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_mse": best_val, "config": cfg}, best_path)
            write_best_model_py(best_py, best_path, input_cols, target_cols, HIDDEN, DEPTH, ACTIVATION, DROPOUT)

        log.info("Epoch %03d/%d | train_mse=%s | val_mse=%s | best=%s",
                 epoch, EPOCHS, fmt(train_mse), fmt(val_mse), "Yes" if is_best else "No")

    torch.save({"model": model.state_dict(), "epoch": EPOCHS, "val_mse": best_val}, last_path)

    dur = time.time() - t0
    log.info("Done in %s s. Best val_mse=%s @ epoch %d", fmt(dur), fmt(best_val), best_epoch)
    log.info("Best checkpoint: %s", best_path)
    log.info("Saved module    : %s", best_py)

    # Final test (best)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    test_mse = evaluate(
        model,
        DataLoader(TabDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
        device, loss_fn
    )
    log.info("TEST MSE (best epoch %d): %s", best_epoch, fmt(test_mse))


if __name__ == "__main__":
    main()
