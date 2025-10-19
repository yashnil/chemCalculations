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

# pooled histogram across ALL species (shows the global bump near ~1–2%)
LO, HI = 0.013, 0.015

# pooled histogram across ALL species (shows the global bump near ~1–2%)
try:
    _band = json.load(open("stripe_patch.json")).get("band", [0.013, 0.015])
    LO, HI = _band
except Exception:
    LO, HI = 0.013, 0.015

vals = Y_true.ravel()
plt.figure(figsize=(5.2, 3.6))
plt.hist(vals, bins=np.logspace(-10, 0, 120), alpha=0.95)
plt.axvspan(LO, HI, color="red", alpha=0.15, lw=0)
plt.xscale("log")
plt.xlabel("Observed abundance (all species pooled)")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hist_pooled_all_species.png"), dpi=180)
plt.close()


# --- 60s proof: the stripe is an equal-share (1/N_active) effect ----------------
LO, HI = 0.013, 0.015
rows_stripe = ((Y_true >= LO) & (Y_true <= HI)).any(axis=1)
print(f"Rows with ANY species in [{LO},{HI}]: {rows_stripe.sum()} / {len(Y_true)}")

# Count "active" species at sensible thresholds (not CLIP!)
for thr in (1e-3, 2e-3, 5e-3, 1e-2):   # 0.1%, 0.2%, 0.5%, 1%
    n_act_stripe = (Y_true[rows_stripe] > thr).sum(axis=1)
    n_act_rest   = (Y_true[~rows_stripe] > thr).sum(axis=1)
    med_s = float(np.median(n_act_stripe)) if n_act_stripe.size else float("nan")
    med_r = float(np.median(n_act_rest))   if n_act_rest.size else float("nan")
    inv_s = (1.0/med_s) if med_s and np.isfinite(med_s) and med_s>0 else float("nan")
    inv_r = (1.0/med_r) if med_r and np.isfinite(med_r) and med_r>0 else float("nan")
    print(f"thr={thr:g} → median N_active stripe={med_s:.1f} (1/N≈{inv_s:.4f}) | "
          f"rest={med_r:.1f} (1/N≈{inv_r:.4f})")

# Optional: show ranked-abundance curves for stripe vs rest
K = 80
def _ranked(A):
    A = -np.sort(-A, axis=1)[:, :K]          # top-K per row
    med = np.median(A, axis=0)
    p10 = np.percentile(A, 10, axis=0)
    p90 = np.percentile(A, 90, axis=0)
    return med, p10, p90

if rows_stripe.any() and (~rows_stripe).any():
    med_s, lo_s, hi_s = _ranked(Y_true[rows_stripe])
    med_r, lo_r, hi_r = _ranked(Y_true[~rows_stripe])
    x = np.arange(1, K+1)
    plt.figure(figsize=(6.4, 4.0))
    plt.loglog(x, med_s, label="stripe rows (median)")
    plt.fill_between(x, lo_s, hi_s, alpha=0.15, linewidth=0)
    plt.loglog(x, med_r, label="other rows (median)")
    plt.fill_between(x, lo_r, hi_r, alpha=0.15, linewidth=0)
    plt.axhline((LO+HI)/2, color="red", ls="--", lw=1, alpha=0.6)
    plt.xlabel("ranked species (1 = most abundant)")
    plt.ylabel("abundance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ranked_abundance_stripe_vs_rest.png"), dpi=180)
    plt.close()
# ------------------------------------------------------------------------------


# ── quick stripe provenance check (paste right after Y_true) ──
FLOOR = 1e-10           # counts a species as "active" above this
LO, HI = 0.013, 0.015   # the vertical stripe band on x (observed)

A = Y_true.astype(np.float64)
n_active = (A >= FLOOR).sum(axis=1)                      # how many species carry non-tiny mass?
stripe_rows = ((A >= LO) & (A <= HI)).any(axis=1)        # any species in the 0.013–0.015 band?
# row entropy (natural log). Uniform-within-active rows -> high entropy

row_entropy = -np.sum(A * np.log(A + EPS), axis=1)

# Bin rows by entropy and make parity panels for top-10 species per bin
q = np.quantile(row_entropy[np.isfinite(row_entropy)], [0.2, 0.8])
low  = row_entropy <= q[0]
mid  = (row_entropy > q[0]) & (row_entropy <= q[1])
high = row_entropy > q[1]

def parity_by_mask(mask, tag):
    sub_true = Y_true[mask]; sub_pred = Y_pred[mask]
    top10 = tbl.sort_values("max_abun", ascending=False)["species"].head(10).values

    fig = plt.figure(figsize=(15, 6))
    axs = [fig.add_subplot(2,5,i) for i in range(1,11)]

    for ax, sp in zip(axs, top10):
        j = SPECIES.index(sp)
        x = sub_true[:, j]; y = sub_pred[:, j]
        m = (x > CLIP) & (y > CLIP)

        # scatter
        ax.scatter(x[m], y[m], s=4, alpha=0.35, edgecolors="none", color="#404388")

        # 1:1
        xl = np.logspace(np.log10(CLIP), 0, 200)
        ax.plot(xl, xl, lw=1.2, color="k")

        # axes
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(CLIP, 1); ax.set_ylim(CLIP, 1)
        ax.set_title(sp, fontsize=9)

        # per-panel metrics (log-space)
        if m.any():
            mae_i = float(np.mean(np.abs(np.log10(y[m] + EPS) - np.log10(x[m] + EPS))))
            r2_i  = float(r2_score(np.log10(x[m] + EPS), np.log10(y[m] + EPS)))
            n_i   = int(m.sum())
        else:
            mae_i, r2_i, n_i = np.nan, np.nan, 0

        ax.text(0.04, 0.94,
                f"MAE={mae_i:.3f} dex\nR²={r2_i:.3f}\nN={n_i}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.65))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"parity_top10_entropy_{tag}.png"), dpi=180)
    plt.close()

def write_stratum_summary(mask, tag):
    """Aggregate metrics for the whole stratum and per-species (log-space)."""
    if not np.any(mask):
        with open(os.path.join(OUT_DIR, f"metrics_entropy_{tag}.txt"), "w") as fh:
            fh.write("N_rows 0\n")
        return

    y_t = Y_true[mask]
    y_p = Y_pred[mask]

    # overall metrics in linear space
    mae   = mean_absolute_error(y_t, y_p)
    r2    = r2_score(y_t, y_p, multioutput="variance_weighted")
    bkl   = float(np.mean(np.sum(y_t * (np.log(y_t + EPS) - np.log(y_p + EPS)), axis=1)))

    with open(os.path.join(OUT_DIR, f"metrics_entropy_{tag}.txt"), "w") as fh:
        fh.write(f"N_rows  {y_t.shape[0]}\n")
        fh.write(f"MAE     {mae:.6e}\n")
        fh.write(f"R2      {r2:.6f}\n")
        fh.write(f"BAL_KL  {bkl:.6e}\n")

    # per-species metrics in log-space (dex)
    mae_dex = np.mean(np.abs(np.log10(y_p + EPS) - np.log10(y_t + EPS)), axis=0)
    r2_log  = [r2_score(np.log10(y_t[:, i] + EPS), np.log10(y_p[:, i] + EPS))
               for i in range(y_t.shape[1])]
    pd.DataFrame({"species": SPECIES, "MAE_dex": mae_dex, "R2_log": r2_log}) \
      .to_csv(os.path.join(OUT_DIR, f"per_species_entropy_{tag}.csv"), index=False)


print("Rows hitting the stripe:", int(stripe_rows.sum()))
print("Median N_active in stripe vs non-stripe:",
      int(np.median(n_active[stripe_rows])) if stripe_rows.any() else 0,
      "vs",
      int(np.median(n_active[~stripe_rows])) if (~stripe_rows).any() else 0)
print("Mean entropy in stripe vs non-stripe:",
      float(row_entropy[stripe_rows].mean()) if stripe_rows.any() else float("nan"),
      "vs",
      float(row_entropy[~stripe_rows].mean()) if (~stripe_rows).any() else float("nan"))

# also write a tiny report + a quick diagnostic scatter
with open(os.path.join(OUT_DIR, "stripe_proof.txt"), "w") as fh:
    fh.write(f"rows_in_stripe {int(stripe_rows.sum())}\n")
    if stripe_rows.any():
        fh.write(f"median_N_active_stripe {np.median(n_active[stripe_rows]):.1f}\n")
        fh.write(f"mean_entropy_stripe    {row_entropy[stripe_rows].mean():.4f}\n")
    if (~stripe_rows).any():
        fh.write(f"median_N_active_other  {np.median(n_active[~stripe_rows]):.1f}\n")
        fh.write(f"mean_entropy_other     {row_entropy[~stripe_rows].mean():.4f}\n")

import matplotlib.pyplot as plt
plt.figure(figsize=(5.2, 4.2))
plt.scatter(n_active[~stripe_rows], row_entropy[~stripe_rows], s=6, alpha=0.25, label="other")
plt.scatter(n_active[stripe_rows],  row_entropy[stripe_rows],  s=10, alpha=0.8, label="stripe")
plt.xlabel("N_active (≥ {:.0e})".format(FLOOR))
plt.ylabel("Row entropy  (−∑ p ln p)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "stripe_entropy_vs_nactive.png"), dpi=160)
plt.close()
# ── end quick check ──

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

# --- band-limited post-hoc calibration for the stripe -------------------------
with open("stripe_patch.json") as fh:
    patch = json.load(fh)

LO, HI = patch.get("band", [0.013, 0.015])
offset = patch.get("offset", {})
affine = patch.get("affine", {})
kmap   = patch.get("k", {})

applied_rows = 0

if offset:
    # Inside band: log10(yhat) += offset[sp]
    for j, sp in enumerate(SPECIES):
        d = offset.get(sp)
        if d is None:
            continue
        mask = ((Y_true[:, j] >= LO) & (Y_true[:, j] <= HI)) | \
               ((Y_pred[:, j] >= LO) & (Y_pred[:, j] <= HI))
        if not mask.any():
            continue
        yhat = Y_pred[mask, j]
        Y_pred[mask, j] = 10.0 ** (np.log10(yhat + EPS) + float(d))
        applied_rows += int(mask.sum())

elif affine:
    # legacy log-affine correction: log10(yhat) -> s*log10(yhat) + c
    for j, sp in enumerate(SPECIES):
        prm = affine.get(sp)
        if not prm:
            continue
        s = float(prm["slope"]); c = float(prm["intercept"])
        mask = ((Y_true[:, j] >= LO) & (Y_true[:, j] <= HI)) | \
               ((Y_pred[:, j] >= LO) & (Y_pred[:, j] <= HI))
        if not mask.any():
            continue
        yhat = Y_pred[mask, j]
        Y_pred[mask, j] = 10.0 ** (s * np.log10(yhat + EPS) + c)
        applied_rows += int(mask.sum())

elif kmap:
    # older multiplicative patch
    for j, sp in enumerate(SPECIES):
        if sp not in kmap:
            continue
        k = float(kmap[sp])
        mask = ((Y_true[:, j] >= LO) & (Y_true[:, j] <= HI)) | \
               ((Y_pred[:, j] >= LO) & (Y_pred[:, j] <= HI))
        if mask.any():
            Y_pred[mask, j] *= k
            applied_rows += int(mask.sum())
else:
    print("stripe_patch.json has no 'offset'/'affine'/'k'; skipping patch.")

# keep probabilities valid after patching
Y_pred = np.maximum(Y_pred, 0.0)
Y_pred /= (Y_pred.sum(axis=1, keepdims=True) + EPS)

print(f"Stripe patch applied to ~{applied_rows:,} row/species positions within [{LO},{HI}].")
# ---------------------------------------------------------------------------

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


parity_by_mask(low,  "low");   write_stratum_summary(low,  "low")
parity_by_mask(mid,  "mid");   write_stratum_summary(mid,  "mid")
parity_by_mask(high, "high");  write_stratum_summary(high, "high")

# ── 3.1  residual vs observed (top-10 species) ─────────────────
top10 = tbl.sort_values("max_abun", ascending=False)["species"].head(10).values

fig = plt.figure(figsize=(15, 6))
axs = [fig.add_subplot(2, 5, i) for i in range(1, 11)]

for ax, sp in zip(axs, top10):
    j = SPECIES.index(sp)
    x = Y_true[:, j]; y = Y_pred[:, j]
    m = (x > CLIP) & (y > CLIP)

    res = np.log10((y[m] + EPS) / (x[m] + EPS))  # dex residual
    ax.scatter(x[m], res, s=4, alpha=0.35, edgecolors="none", color="#404388")

    ax.axhline(0, color="k", lw=1.0)
    ax.axvspan(LO, HI, color="red", alpha=0.10, lw=0)  # instead of 0.013, 0.015

    ax.set_xscale("log"); ax.set_xlim(CLIP, 1); ax.set_ylim(-1, 1)
    ax.set_title(sp, fontsize=9)
    ax.set_ylabel("Δ log10 (pred/true)" if sp == top10[0] else "")
    ax.set_xlabel("observed abundance")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "residual_vs_observed_top10.png"), dpi=180)
plt.close()
# ───────────────────────────────────────────────────────────────


# --- sanity: do top-10 species actually have a spike near the stripe? ---
top10 = tbl.sort_values("max_abun", ascending=False)["species"].head(10).values

for sp in top10:
    j = SPECIES.index(sp)
    vals = Y_true[:, j]
    # save a log-scaled histogram of observed abundances
    plt.figure(figsize=(4.2, 3.2))
    plt.hist(vals, bins=np.logspace(-10, 0, 80), alpha=0.9)
    plt.axvspan(LO, HI, color="red", alpha=0.15, lw=0)
    plt.xscale("log")
    plt.xlabel(f"{sp} observed abundance")
    plt.ylabel("count")
    plt.title(f"Observed distribution – {sp}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"hist_obs_{sp}.png"), dpi=160)
    plt.close()


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

        # -------- density / scatter -----------
        if _HAS_DS:
            dsshow(pd.DataFrame({"x": x, "y": y}),
                   ds.Point("x", "y"), ax=ax, **shade_kwargs)
        else:
            ax.scatter(x, y, s=4, alpha=0.35, edgecolors="none", color="#404388")

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

        # optional down-sample to speed up KDE on very large arrays
        if x.size > max_pts:
            sel = np.random.default_rng(0).choice(x.size, size=max_pts, replace=False)
            x, y = x[sel], y[sel]

        # ------------- KDE density colours -------------
        try:
            xy   = np.vstack([np.log10(x), np.log10(y)])     # work in log-space
            z    = gaussian_kde(xy)(xy)
        except Exception as e:  # fall back gracefully
            warnings.warn(f"KDE failed for {sp}: {e}; falling back to plain scatter")
            z = np.full_like(x, 0.0)

        # densest points plotted last  → visible on top
        ord  = z.argsort()
        x, y, z = x[ord], y[ord], z[ord]

        sc = ax.scatter(x, y, c=z, s=7, cmap=white_viridis, edgecolor="none")

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
