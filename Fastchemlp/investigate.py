#!/usr/bin/env python3
# plot_inputs_grid.py — ensure abund_e-_dex is always included among plotted inputs

import os
import math
import logging
from typing import List, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# GLOBALS
# =============================================================================
CSV_PATH: str = "/Users/imalsky/Desktop/FastChem-master/python/runs_mlp_all_gas/fastchem_grid_runs.csv"
OUT_DIR:  str = "runs_mlp_all_gas"

BINS: int = 120
FIGSIZE_SINGLE: Tuple[int, int] = (8, 5)
BAR_ALPHA: float = 0.9
BAR_EDGE: str = "none"
FONT_SIZE: int = 12

DROP_COLUMNS: Tuple[str, ...] = ("group_index", "point_index")

EXCLUDE_EXTRA: Tuple[str, ...] = (
    "id", "run_id", "seed", "split", "fold", "index", "sample_id",
    "time", "t", "target", "label",
    "T_K", "P_bar", "fZ", "fZ_dex",
    "flag", "flag_msg", "mean_molecular_weight", "total_element_density",
    "group_index", "point_index",
)

# Prefer these abundance cols first — NOW includes e-
PREFERRED_ABUND: Tuple[str, ...] = (
    "abund_e-_dex",
    "abund_H_dex", "abund_He_dex", "abund_C_dex", "abund_O_dex",
    "abund_N_dex", "abund_S_dex", "abund_Fe_dex", "abund_Mg_dex",
    "abund_Si_dex", "abund_Na_dex"
)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("plot_inputs_grid")

def _is_abund_col(c: str) -> bool:
    return c.startswith("abund_") and c.endswith("_dex")

def ensure_features(df: pd.DataFrame) -> List[str]:
    feats: List[str] = []
    if "T_K" in df.columns: feats.append("T_K")
    if "P_bar" in df.columns: feats.append("P_bar")
    if "fZ_dex" in df.columns: feats.append("fZ_dex")
    elif "fZ" in df.columns: feats.append("fZ")

    abund_cols = [c for c in df.columns if _is_abund_col(c)]
    if abund_cols:
        added = [c for c in PREFERRED_ABUND if c in df.columns]
        remaining = [c for c in sorted(abund_cols) if c not in added]
        cap = 16
        room = max(0, cap - len(feats) - len(added))
        feats += added + remaining[:room]

        # Guarantee abund_e-_dex is included
        if "abund_e-_dex" in df.columns and "abund_e-_dex" not in feats:
            feats.insert(min(len(feats), 3), "abund_e-_dex")  # drop it near the top row

    if not feats:
        log.error("No input features found to plot. Check CSV columns.")
    else:
        log.info("Features to plot (%d): %s", len(feats), feats[:10] + (["..."] if len(feats) > 10 else []))
    return feats

def positive(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values) & (values > 0.0)]

def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]

def log10_hist_counts(series_pos: np.ndarray, bins: int):
    lx = np.log10(series_pos)
    xmin, xmax = float(np.min(lx)), float(np.max(lx))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        xmax = xmin + 1e-6
    edges = np.linspace(xmin, xmax, bins + 1)
    counts, edges = np.histogram(lx, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    return centers, widths, counts

def linear_hist_counts(series_finite: np.ndarray, bins: int):
    xmin, xmax = float(np.min(series_finite)), float(np.max(series_finite))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        xmax = xmin + 1e-6
    edges = np.linspace(xmin, xmax, bins + 1)
    counts, edges = np.histogram(series_finite, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    return centers, widths, counts

def style_axes(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, pad=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.tick_params(axis='both', which='both', labelsize=FONT_SIZE-1)

def choose_grid(n: int) -> Tuple[int, int, Tuple[int, int]]:
    if n <= 3: ncols, nrows = n, 1
    else: ncols, nrows = min(3, n), math.ceil(n / min(3, n))
    figsize = (max(10, 5 * ncols), max(4, int(3.8 * nrows)))
    return nrows, ncols, figsize

def _looks_like_log10(arr_finite: np.ndarray) -> bool:
    if arr_finite.size == 0: return False
    neg_frac = np.mean(arr_finite <= 0.0)
    p5, p95 = np.percentile(arr_finite, 5), np.percentile(arr_finite, 95)
    return (neg_frac > 0.8) and (p95 <= 0.1) and (p5 >= -60.0)

def _convert_to_linear_allrows(vals: np.ndarray, is_log10: bool) -> np.ndarray:
    lin = np.zeros_like(vals, dtype=float)
    mask = np.isfinite(vals)
    if is_log10:
        lin[mask] = np.power(10.0, vals[mask], dtype=float)
        lin[mask] = np.clip(lin[mask], 0.0, None)
    else:
        lin[mask] = np.clip(vals[mask], 0.0, None)
    return lin

def find_species_columns(df: pd.DataFrame, input_cols: Sequence[str]) -> List[str]:
    exclude = set(input_cols) | set(EXCLUDE_EXTRA)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    species = [c for c in numeric_cols if (c not in exclude and not _is_abund_col(c))]
    if len(species) == 0:
        log.warning("No candidate species columns found.")
    else:
        log.info("Detected %d candidate species columns.", len(species))
    return species

def rank_species_by_mean_abundance(df: pd.DataFrame, species_cols: List[str]) -> pd.DataFrame:
    records = []
    n_rows = len(df)
    for col in species_cols:
        vals = df[col].to_numpy(dtype=float, copy=False)
        is_log = _looks_like_log10(vals[np.isfinite(vals)])
        lin = _convert_to_linear_allrows(vals, is_log)
        mean_abund = float(np.mean(lin)) if n_rows > 0 else 0.0
        nonzero_frac = float(np.count_nonzero(lin) / n_rows) if n_rows > 0 else 0.0
        records.append({
            "species": col,
            "mean_abundance": mean_abund,
            "scale": "log10" if is_log else "linear",
            "nonzero_fraction": nonzero_frac
        })
    res = pd.DataFrame.from_records(records)
    res.sort_values(by="mean_abundance", ascending=False, inplace=True, kind="mergesort")
    res.reset_index(drop=True, inplace=True)
    return res

def emit_top20(spec_rank_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    top20 = spec_rank_df.head(20).copy()
    log.info("Top 20 most abundant target species (by mean linear abundance):")
    for i, row in top20.iterrows():
        log.info("  %2d. %-30s mean=%-10.3e  nz_frac=%.3f  (%s)",
                 i+1, row["species"], row["mean_abundance"], row["nonzero_fraction"], row["scale"])
    csv_path = os.path.join(out_dir, "top20_species_by_abundance.csv")
    txt_path = os.path.join(out_dir, "top20_species_by_abundance.txt")
    top20.to_csv(csv_path, index=False)
    with open(txt_path, "w") as f:
        for i, row in top20.iterrows():
            f.write(f"{i+1:2d}. {row['species']:30s}  mean={row['mean_abundance']:.6e}  "
                    f"nz_frac={row['nonzero_fraction']:.3f}  ({row['scale']})\n")
    log.info("Saved Top-20 CSV: %s", os.path.abspath(csv_path))
    log.info("Saved Top-20 TXT : %s", os.path.abspath(txt_path))

def main():
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "axes.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "figure.dpi": 110,
    })

    os.makedirs(OUT_DIR, exist_ok=True)

    log.info("Loading CSV: %s", os.path.abspath(CSV_PATH))
    df = pd.read_csv(CSV_PATH)
    log.info("Loaded shape: %d rows × %d cols", df.shape[0], df.shape[1])

    if any(c in df.columns for c in DROP_COLUMNS):
        df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], inplace=True, errors="ignore")
        log.info("Dropped columns (if present): %s", DROP_COLUMNS)

    feats = ensure_features(df)
    if not feats:
        return

    n_plots = len(feats)
    nrows, ncols, figsize = choose_grid(n_plots)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i, feat in enumerate(feats):
        ax = axes[i]
        raw = df[feat].to_numpy(dtype=float)
        use_logx = (feat in ("T_K", "P_bar", "fZ")) and np.all(np.isfinite(raw))
        if feat == "fZ_dex" or _is_abund_col(feat):
            use_logx = False
        if use_logx:
            pos = positive(raw)
            if pos.size == 0:
                ax.text(0.5, 0.5, "No positive values", ha="center", va="center", transform=ax.transAxes)
                style_axes(ax, f"log10({feat})", "log10(count)", f"{feat}")
            else:
                centers, widths, counts = log10_hist_counts(pos, BINS)
                mask = counts > 0
                ax.bar(centers[mask], np.log10(counts[mask].astype(float)),
                       width=widths[mask]*0.9, align="center", alpha=BAR_ALPHA, edgecolor=BAR_EDGE)
                style_axes(ax, f"log10({feat})", "log10(count)", f"{feat} (N={pos.size} > 0)")
        else:
            fin = finite(raw)
            if fin.size == 0:
                ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
                style_axes(ax, f"{feat}", "log10(count)", f"{feat}")
            else:
                centers, widths, counts = linear_hist_counts(fin, BINS)
                mask = counts > 0
                ax.bar(centers[mask], np.log10(np.maximum(counts[mask].astype(float), 1.0)),
                       width=widths[mask]*0.9, align="center", alpha=BAR_ALPHA, edgecolor=BAR_EDGE)
                style_axes(ax, f"{feat}", "log10(count)", f"{feat} (N={fin.size} finite)")

    for j in range(n_plots, nrows * ncols):
        axes[j].axis("off")

    out_grid = os.path.join(OUT_DIR, "input_distributions_grid.png")
    fig.suptitle("Input Distributions (Grid) — log10(count) histograms", y=1.02, fontsize=FONT_SIZE + 2)
    fig.savefig(out_grid, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved combined figure: %s", os.path.abspath(out_grid))

    for feat in feats:
        raw = df[feat].to_numpy(dtype=float)
        use_logx = (feat in ("T_K", "P_bar", "fZ")) and np.all(np.isfinite(raw))
        if feat == "fZ_dex" or _is_abund_col(feat):
            use_logx = False

        if use_logx:
            pos = positive(raw)
            if pos.size == 0:
                log.warning("Feature '%s' has no positive values — skipping.", feat)
                continue
            centers, widths, counts = log10_hist_counts(pos, BINS)
            mask = counts > 0
            fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
            ax.bar(centers[mask], np.log10(counts[mask].astype(float)),
                   width=widths[mask]*0.9, align="center", alpha=BAR_ALPHA, edgecolor=BAR_EDGE)
            style_axes(ax, f"log10({feat})", "log10(count)", f"{feat} — WHOLE SET (N={pos.size} > 0)")
        else:
            fin = finite(raw)
            if fin.size == 0:
                log.warning("Feature '%s' has no finite values — skipping.", feat)
                continue
            centers, widths, counts = linear_hist_counts(fin, BINS)
            mask = counts > 0
            fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
            ax.bar(centers[mask], np.log10(np.maximum(counts[mask].astype(float), 1.0)),
                   width=widths[mask]*0.9, align="center", alpha=BAR_ALPHA, edgecolor=BAR_EDGE)
            style_axes(ax, f"{feat}", "log10(count)", f"{feat} — WHOLE SET (N={fin.size} finite)")

        out_png = os.path.join(OUT_DIR, f"input_hist_{feat.replace('/', '_')}.png")
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved: %s", os.path.abspath(out_png))

    species_cols = find_species_columns(df, feats)
    if species_cols:
        spec_rank = rank_species_by_mean_abundance(df, species_cols)
        emit_top20(spec_rank, OUT_DIR)
    else:
        log.warning("Skipped species ranking (no candidate species columns).")

    log.info("Done.")

if __name__ == "__main__":
    main()
