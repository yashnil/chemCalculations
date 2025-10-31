#!/usr/bin/env python3
# step-5 → diagnostics.py
# --------------------------------------------------------------
# Rich end-to-end diagnostics for the FastChem-surrogate project
# --------------------------------------------------------------

from matplotlib.colors import LinearSegmentedColormap, Normalize   # ← add Normalize
import os, time, json, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import importlib
import warnings
from scipy.stats import gaussian_kde
_HAS_DS = importlib.util.find_spec("datashader") is not None
if _HAS_DS:
    import datashader as ds
    from datashader.mpl_ext import dsshow
_HAS_DENS = importlib.util.find_spec("mpl_scatter_density") is not None
if _HAS_DENS:
    from mpl_scatter_density import ScatterDensityArtist          # noqa: F401
from matplotlib.colors import LinearSegmentedColormap

white_viridis = LinearSegmentedColormap.from_list(
    'white_viridis',
    [(0, '#ffffff'), (1e-20, '#440053'), (0.2, '#404388'),
     (0.4, '#2a788e'), (0.6, '#21a784'), (0.8, '#78d151'),
     (1, '#fde624')],
    N=256
)
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf 
from tensorflow import keras
from losses import _mae_log                           # helper for log-MAE
import model_heads
from matplotlib.cm import ScalarMappable
keras.config.enable_unsafe_deserialization()



# ╭──────────────────────────── paths ───────────────────────────╮
ARTE_DIR   = "artefacts"
CSV_PATH   = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
MODEL_PATH = os.path.join(ARTE_DIR, "final_model.keras")         # written by finalize.py
CARD_JSON  = os.path.join(ARTE_DIR, "final_report.json")         #  ”       ”      ”
SCALER_PKL = os.path.join(ARTE_DIR, "input_scaler.pkl")
OUT_DIR    = os.path.join(ARTE_DIR, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

EPS, CLIP = 1e-12, 1e-10      # small helpers
# ╰──────────────────────────────────────────────────────────────╯


# ╭──────────────── 1.  load data & artefacts ───────────────────╮
df = pd.read_csv(CSV_PATH)

# ── mirror the exact filtering used in baseline_checks/train ───
df["T_bin"] = pd.qcut(df["temperature"], 5, labels=False, duplicates="drop")
df = df[df["T_bin"] != 0].reset_index(drop=True).drop(columns="T_bin")

ELEMENT_COLS = [f"comp_{e}" for e in ("H", "O", "C", "N", "S")]
META_COLS    = {"temperature", "pressure", "group_index", "point_index"} | set(ELEMENT_COLS)
SPECIES      = [c for c in df.columns if c not in META_COLS]

# renormalise gas-phase rows (numerical guard)
df[SPECIES] = df[SPECIES].div(df[SPECIES].sum(axis=1), axis=0)

# inputs / scaler ---------------------------------------------------------
with open(CARD_JSON) as fh:
    card = json.load(fh)
INPUTS  = card.get("inputs",
                   ["temperature", "pressure",
                    "comp_H", "comp_O", "comp_C", "comp_N", "comp_S"])

SPECIES = card.get("outputs")            # may be None
if SPECIES is None:
    ELEMENT_COLS = [f"comp_{e}" for e in ("H","O","C","N","S")]
    META_COLS    = {"temperature", "pressure",
                    "group_index", "point_index"} | set(ELEMENT_COLS)
    SPECIES = [c for c in df.columns if c not in META_COLS]

scaler = joblib.load(SCALER_PKL)

X = df[INPUTS].copy()
X["pressure"] = np.log10(X["pressure"])
for el in INPUTS[2:]:
    X[el] = np.log10(X[el]) + 9.0
X = scaler.transform(X).astype("float32")

Y_true = df[SPECIES].values.astype("float32")

def _inject_tf(layer):
    """Recursively walk `layer` and put `tf` in every Lambda’s globals."""
    if isinstance(layer, keras.layers.Lambda):
        fn = getattr(layer, "function", None) or getattr(layer, "_function")
        if fn is not None:
            fn.__globals__.setdefault("tf", tf)

    # If the layer contains sub-layers (e.g. Sequential, Functional)
    if hasattr(layer, "layers"):
        for sub in layer.layers:
            _inject_tf(sub)

# --- load model -------------------------------------------------
model = keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False,
    custom_objects={"tf": tf}  # needed for deserialisation
)

_inject_tf(model)

# --- ensure every Lambda can see tf at run-time -----------------
for lyr in model.layers:
    if isinstance(lyr, keras.layers.Lambda):
        fn = getattr(lyr, "function", None) or getattr(lyr, "_function")
        fn.__globals__.setdefault("tf", tf)
# ----------------------------------------------------------------

t0 = time.time()
Y_pred = model.predict(X, batch_size=256, verbose=0)
print(f"Predicted {len(X):,} samples in {time.time()-t0:.1f} s")

# guarantee positivity + re-normalise (just in case)
Y_pred = np.maximum(Y_pred, 0.0)
Y_pred /= Y_pred.sum(axis=1, keepdims=True) + EPS
# ╰──────────────────────────────────────────────────────────────╯


# ╭──────────────── 2.  global metrics ──────────────────────────╮
mae_glob  = mean_absolute_error(Y_true, Y_pred)
r2_glob   = r2_score(Y_true, Y_pred, multioutput="variance_weighted")
bkl_glob  = np.mean(np.sum(Y_true * (np.log(Y_true+EPS) - np.log(Y_pred+EPS)), axis=1))

# extra: speed-up
speedup = None
if "speedup" in card:        # value was written by finalize.py
    speedup = card["speedup"]
    print(f"GLOBAL  MAE={mae_glob:.4e}   R²={r2_glob:.3f}   "
          f"B-KL={bkl_glob:.4e}   speed-up ×{speedup:,.1f}")
else:
    print(f"GLOBAL  MAE={mae_glob:.4e}   R²={r2_glob:.3f}   "
          f"B-KL={bkl_glob:.4e}")

# write it to the txt file too
with open(os.path.join(OUT_DIR, "global_metrics.txt"), "w") as fh:
    fh.write(f"MAE       {mae_glob:.6e}\n")
    fh.write(f"R2        {r2_glob:.6f}\n")
    fh.write(f"BAL_KL    {bkl_glob:.6e}\n")
    if speedup is not None:
        fh.write(f"SPEEDUP  {speedup:.1f}\n")


print(f"GLOBAL  MAE={mae_glob:.4e}   R²={r2_glob:.3f}   B-KL={bkl_glob:.4e}")

with open(os.path.join(OUT_DIR, "global_metrics.txt"), "w") as fh:
    fh.write(f"MAE       {mae_glob:.6e}\n")
    fh.write(f"R2        {r2_glob:.6f}\n")
    fh.write(f"BAL_KL    {bkl_glob:.6e}\n")
# ╰──────────────────────────────────────────────────────────────╯


# ╭──────────────── 3.  per-species table ───────────────────────╮
mae_sp = np.mean(np.abs(Y_true - Y_pred), axis=0)
r2_sp  = [r2_score(Y_true[:, i], Y_pred[:, i]) for i in range(len(SPECIES))]
tbl = pd.DataFrame({"species": SPECIES,
                    "MAE": mae_sp,
                    "R2":  r2_sp,
                    "max_abun": Y_true.max(axis=0)})
tbl.sort_values("MAE").to_csv(os.path.join(OUT_DIR, "per_species_errors.csv"),
                              index=False)
# ╰──────────────────────────────────────────────────────────────╯

# ╭──────────── 3.5  extreme-ratio CSV for top-10 species ───────╮
top10_species = tbl.sort_values("max_abun", ascending=False)["species"].head(10).values
rows = []

for sp in top10_species:
    idx   = SPECIES.index(sp)
    y_t   = Y_true[:, idx]
    y_p   = Y_pred[:, idx]
    ratio = np.maximum(y_p/(y_t + EPS), y_t/(y_p + EPS))  # max(pred/obs, obs/pred)
    mask  = ratio > 1e3

    if not np.any(mask):
        continue

    sub = df.loc[mask, ["temperature", "pressure", "group_index", "point_index",
                        "comp_H", "comp_O", "comp_C", "comp_N", "comp_S"]].copy()
    sub["species"]      = sp
    sub["observed"]     = y_t[mask]
    sub["predicted"]    = y_p[mask]
    sub["pred_over_obs"]= y_p[mask] / (y_t[mask] + EPS)
    sub["obs_over_pred"]= y_t[mask] / (y_p[mask] + EPS)
    sub["ratio_max"]    = ratio[mask]
    sub["log10P"]       = np.log10(sub["pressure"])
    rows.append(sub)

cols = ["temperature","pressure","log10P","group_index","point_index",
        "comp_H","comp_O","comp_C","comp_N","comp_S",
        "species","observed","predicted","pred_over_obs","obs_over_pred","ratio_max"]
outliers_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
out_csv = os.path.join(OUT_DIR, "top10_extreme_ratio_gt1e3.csv")
outliers_df.to_csv(out_csv, index=False)
print(f"Saved extreme-ratio CSV → {out_csv} (rows={len(outliers_df)})")
# ╰──────────────────────────────────────────────────────────────╯

# ╭────────── 3.6  shoulder band (1.3–1.5%) diagnostics ─────────╮
LO, HI = 0.013, 0.015
band10 = tbl.sort_values("max_abun", ascending=False)["species"].head(10).values

records = []
rows = []
for sp in band10:
    i = SPECIES.index(sp)
    y_t = Y_true[:, i]; y_p = Y_pred[:, i]

    in_band = (y_t >= LO) & (y_t <= HI)
    if not in_band.any():
        records.append({"species": sp, "n_in_band": 0,
                        "cov_±10%": np.nan, "med_rel_err%": np.nan})
        continue

    rel_err = np.abs(np.log10((y_p + EPS) / (y_t + EPS)))  # log-ratio error
    cov10   = (rel_err <= np.log10(1.1))

    rec = {
        "species": sp,
        "n_in_band": int(in_band.sum()),
        "cov_±10%": float(cov10[in_band].mean()*100),
        "med_rel_err%": float(np.median((np.abs(y_p - y_t)/(y_t+EPS))[in_band]) * 100),
    }
    records.append(rec)

    # dump the actual rows for this band
    sub = df.loc[in_band, ["temperature","pressure","group_index","point_index",
                           "comp_H","comp_O","comp_C","comp_N","comp_S"]].copy()
    sub["species"]   = sp
    sub["observed"]  = y_t[in_band]
    sub["predicted"] = y_p[in_band]
    sub["rel_err%"]  = np.abs(y_p[in_band] - y_t[in_band])/(y_t[in_band] + EPS)*100
    sub["log10P"]    = np.log10(sub["pressure"])
    rows.append(sub)

pd.DataFrame(records).to_csv(os.path.join(OUT_DIR, "shoulder_band_metrics_top10.csv"), index=False)
pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT_DIR, "shoulder_band_rows_top10.csv"), index=False)
print("Saved → shoulder_band_metrics_top10.csv & shoulder_band_rows_top10.csv")
# ╰──────────────────────────────────────────────────────────────╯


# ── 4.  top-10 parity plots  (Datashader) ──────────────────────
def make_parity_panels(add_band: bool, out_name: str):
    """Render top-10 species parity plots; optionally add ±10 % band."""
    top10  = tbl.sort_values("max_abun", ascending=False)["species"].head(10).values
    fig    = plt.figure(figsize=(15, 6))
    axs    = [fig.add_subplot(2, 5, i) for i in range(1, 11)]

    if _HAS_DS:
        shade_kwargs = dict(
            agg    = ds.count(),
            cmap   = white_viridis,
            vmin   = 0,
            vmax   = 35,
            norm   = "linear",
            aspect = "auto",
        )

    for ax, sp in zip(axs, top10):
        idx  = SPECIES.index(sp)
        mask = (Y_true[:, idx] > CLIP) & (Y_pred[:, idx] > CLIP)
        x, y = Y_true[mask, idx], Y_pred[mask, idx]

        # highlight crazy outliers (already added)
        ratio_xy = np.maximum(y/(x + EPS), x/(y + EPS))
        red = ratio_xy > 1e3
        if red.any():
            ax.scatter(x[red], y[red], s=18, color="red",
                    edgecolors="white", linewidths=0.4, alpha=0.95, zorder=3)

        # NEW: highlight 1.3–1.5% shoulder misses (>±10%) in orange
        band_mask = (x >= 0.013) & (x <= 0.015)
        miss10    = np.abs(np.log10((y + EPS)/(x + EPS))) > np.log10(1.1)
        orange    = band_mask & miss10
        if orange.any():
            ax.scatter(x[orange], y[orange], s=18, color="orange",
                    edgecolors="k", linewidths=0.3, alpha=0.95, zorder=3)


        # -------- density / scatter -----------
        if _HAS_DS:
            dsshow(pd.DataFrame({"x": x, "y": y}),
                ds.Point("x", "y"), ax=ax, **shade_kwargs)
        else:
            ax.scatter(x, y, s=4, alpha=0.35, edgecolors="none", color="#404388")

        # -------- highlight |pred/obs| > 1e3 in bright red ----------
        ratio_xy = np.maximum(y/(x + EPS), x/(y + EPS))  # x,y already masked > CLIP
        red = ratio_xy > 1e3
        if red.any():
            ax.scatter(x[red], y[red], s=18, color="red",
                    edgecolors="white", linewidths=0.4, alpha=0.95, zorder=3)

        # -------- reference lines -------------
        x_line = np.logspace(np.log10(CLIP), 0, 200)
        ax.plot(x_line, x_line, lw=1.2, color="k")
        if add_band:
            ax.fill_between(x_line, x_line*0.9, x_line*1.1,
                            color="grey", alpha=0.12, zorder=0)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(CLIP, 1);  ax.set_ylim(CLIP, 1)

        # -------- small metrics box -----------
        mae_i = np.mean(np.abs(np.log10(y+EPS) - np.log10(x+EPS)))
        r2_i  = r2_score(np.log10(x+EPS), np.log10(y+EPS))
        ax.text(0.04, 0.94, f"MAE={mae_i:.3f} dex\nR²={r2_i:.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.65))
        ax.set_title(sp, fontsize=9)


    # -------- shared colour-bar --------------
    if _HAS_DS:
        sm = ScalarMappable(cmap=white_viridis,
                            norm=Normalize(0, 35))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs, fraction=0.03, pad=0.02)
        cbar.set_label("# points / pixel")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=180)
    plt.close()


# call once *with* band and once *without*
make_parity_panels(add_band=True,  out_name="parity_top10.png")
make_parity_panels(add_band=False, out_name="parity_top10_noband.png")

def parity_kde(add_band: bool, fname: str, max_pts: int = 15_000):
    """
    Render top-10 abundance species parity panels with KDE-coloured points.
    If the sample is huge we'll randomly down-sample to `max_pts` to keep
    the kernel fit tractable.
    """
    top10 = tbl.sort_values("max_abun", ascending=False)["species"].head(10).values
    fig   = plt.figure(figsize=(15, 6))
    axs   = [fig.add_subplot(2, 5, i) for i in range(1, 11)]

    for ax, sp in zip(axs, top10):
        idx  = SPECIES.index(sp)
        mask = (Y_true[:, idx] > CLIP) & (Y_pred[:, idx] > CLIP)
        x, y = Y_true[mask, idx], Y_pred[mask, idx]

        # highlight crazy outliers (already added)
        ratio_xy = np.maximum(y/(x + EPS), x/(y + EPS))
        red = ratio_xy > 1e3
        if red.any():
            ax.scatter(x[red], y[red], s=18, color="red",
                    edgecolors="white", linewidths=0.4, alpha=0.95, zorder=3)

        # NEW: highlight 1.3–1.5% shoulder misses (>±10%) in orange
        band_mask = (x >= 0.013) & (x <= 0.015)
        miss10    = np.abs(np.log10((y + EPS)/(x + EPS))) > np.log10(1.1)
        orange    = band_mask & miss10
        if orange.any():
            ax.scatter(x[orange], y[orange], s=18, color="orange",
                    edgecolors="k", linewidths=0.3, alpha=0.95, zorder=3)


        # optional down-sample to speed up KDE on very large arrays
        if x.size > max_pts:
            sel = np.random.default_rng(0).choice(x.size, size=max_pts, replace=False)
            x, y = x[sel], y[sel]

        # ------------- KDE density colours -------------
        try:
            xy   = np.vstack([np.log10(x), np.log10(y)])
            z    = gaussian_kde(xy)(xy)
        except Exception as e:
            warnings.warn(f"KDE failed for {sp}: {e}; falling back to plain scatter")
            z = np.full_like(x, 0.0)

        ord  = z.argsort()
        x, y, z = x[ord], y[ord], z[ord]
        sc = ax.scatter(x, y, c=z, s=7, cmap=white_viridis, edgecolor="none")

        # -------- highlight |pred/obs| > 1e3 in bright red ----------
        ratio_xy = np.maximum(y/(x + EPS), x/(y + EPS))
        red = ratio_xy > 1e3
        if red.any():
            ax.scatter(x[red], y[red], s=18, color="red",
                       edgecolors="white", linewidths=0.4, alpha=0.95, zorder=3)


        # 1:1 line & optional ±10 % band
        x_line = np.logspace(np.log10(CLIP), 0, 200)
        ax.plot(x_line, x_line, lw=1.2, color="k")
        if add_band:
            ax.fill_between(x_line, x_line*0.9, x_line*1.1,
                            color="grey", alpha=0.12, zorder=0)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(CLIP, 1); ax.set_ylim(CLIP, 1)

        # small per-species stats box (log-space)
        mae_i = np.mean(np.abs(np.log10(y+EPS) - np.log10(x+EPS)))
        r2_i  = r2_score(np.log10(x+EPS), np.log10(y+EPS))
        ax.text(0.04, 0.94, f"MAE={mae_i:.3f} dex\nR²={r2_i:.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.65))

        ax.set_title(sp, fontsize=9)

    # shared colour-bar
    cbar = fig.colorbar(ScalarMappable(cmap=white_viridis),
                        ax=axs, fraction=0.03, pad=0.02)
    cbar.set_label("KDE density (arb. units)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=180)
    plt.close()


# produce both variants
parity_kde(add_band=True,  fname="parity_top10_kde.png")       # with grey ±10 %
parity_kde(add_band=False, fname="parity_top10_kde_noband.png")  # clean heat-map

# ╭──────────────── 5.  residual T–P map (worst MAE species) ────╮
worst_idx = int(tbl["MAE"].idxmax())
worst_sp  = tbl.loc[worst_idx, "species"]

residual = (np.log10(Y_pred[:, worst_idx] + EPS) -
            np.log10(Y_true[:, worst_idx] + EPS))            # dex

plt.figure(figsize=(6.4, 4.8))
hb = plt.hexbin(df["temperature"], np.log10(df["pressure"]),
                C=residual, gridsize=70, cmap="coolwarm",
                vmin=-1, vmax=1, mincnt=3, linewidths=0)
plt.colorbar(label=r"$\Delta\log_{10}$ (dex)")
plt.xlabel("Temperature  [K]")
plt.ylabel("log10  Pressure  [bar]")
plt.title(f"Residual map – worst MAE species: {worst_sp}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"residual_TP_{worst_sp}.png"), dpi=180)
plt.close()
# ╰──────────────────────────────────────────────────────────────╯


# ╭──────────────── 6.  MAE vs species bar (+ colour) ───────────╮
plt.figure(figsize=(12, 4))
order  = tbl["MAE"].argsort().values
normed = (np.log10(tbl["max_abun"].values[order]) - np.log10(CLIP))
normed = normed / normed.max()
colors = sns.color_palette("viridis", as_cmap=True)(normed)
plt.bar(np.arange(len(SPECIES)), tbl["MAE"].values[order], color=colors)
plt.xticks(np.arange(len(SPECIES)), tbl["species"].values[order],
           rotation=90, fontsize=6)
plt.ylabel("MAE"); plt.title("Per-species MAE (colour = log-max abundance)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "MAE_per_species.png"), dpi=180)
plt.close()
# ╰──────────────────────────────────────────────────────────────╯


# ╭──────────────── 7.  sample-wise %-error chart ───────────────╮
pct_err   = (np.abs(Y_pred - Y_true) / (Y_true + EPS)).mean(axis=1) * 100
plt.figure(figsize=(10, 3))
plt.plot(pct_err, '.', ms=2.5, alpha=0.55)
plt.xlabel("Sample index"); plt.ylabel("mean % error")
plt.title("Per-sample deviation  ⟨|pred-true|/true⟩ × 100")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_deviation.png"), dpi=180)
plt.close()
# ╰──────────────────────────────────────────────────────────────╯


# ╭──────────────── 8.  worst-100 table for inspection ──────────╮
worst100 = pct_err.argsort()[-100:]
df.iloc[worst100].assign(mean_pct_err=pct_err[worst100]) \
  .to_csv(os.path.join(OUT_DIR, "worst100_samples.csv"), index=False)
# ╰──────────────────────────────────────────────────────────────╯

print(f"\nDiagnostics written to →  {OUT_DIR}")
