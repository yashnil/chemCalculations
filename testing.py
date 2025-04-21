#!/usr/bin/env python3
"""
Benchmark native FastChem on  N  single‑point evaluations.

Update the three paths below before running!
"""

import os, time, tempfile, numpy as np, pandas as pd, pyfastchem

# ────────────────────────────────────────────────────────────────────────────
# 0.  USER SETTINGS  –––––––––––––––––––––––––––––––––––––––––––––––––––––––––
CSV_PATH   = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas1.csv"
LOGK_PATH  = "/Users/yashnilmohanty/Downloads/FastChem-master/input/logK/logK.dat"
COND_PATH  = "none"        # set to real file if you want condensates
N_BENCH    = 250           # number of random samples for the timing
VERBOSE    = 0             # 0 = quiet   1 = FastChem prints progress
# ────────────────────────────────────────────────────────────────────────────


def write_abundance_file(comp: dict, fname: str):
    """Minimal abundance file (H=12 scale) for a single composition dict."""
    H = comp["H"]
    def to_logX(v): return -999.0 if v <= 0 else 12.0 + np.log10(v / H)
    lines = [
        "# auto‑generated",
        "e-  0.0",
        f"H   {to_logX(comp['H']):.4f}",
        f"O   {to_logX(comp['O']):.4f}",
        f"C   {to_logX(comp['C']):.4f}",
        f"N   {to_logX(comp['N']):.4f}",
        f"S   {to_logX(comp['S']):.4f}",
    ]
    with open(fname, "w") as f:
        f.write("\n".join(lines) + "\n")


def time_one_call(T, P, comp, scratch_dir):
    """Run FastChem once and return elapsed wall time (seconds)."""
    abund_file = os.path.join(scratch_dir, "abund.dat")
    write_abundance_file(comp, abund_file)

    fc = pyfastchem.FastChem(abund_file, LOGK_PATH, COND_PATH, VERBOSE)

    inp = pyfastchem.FastChemInput()
    inp.temperature = [float(T)]
    inp.pressure    = [float(P)]
    out = pyfastchem.FastChemOutput()

    t0 = time.perf_counter()
    fc.calcDensities(inp, out)
    return time.perf_counter() - t0


def main():
    # 1) Load CSV & pick random rows -------------------------------------------------
    df = pd.read_csv(CSV_PATH)
    rng = np.random.default_rng(seed=123)
    rows = df.sample(N_BENCH, random_state=123)

    # 2) Prepare a scratch directory -------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        times = []
        for idx, row in rows.iterrows():
            comp = {"H":row["H"], "O":row["O"], "C":row["C"], "N":row["N"], "S":row["S"]}
            t = time_one_call(
                    T=row["temperature"],
                    P=row["pressure"],
                    comp=comp,
                    scratch_dir=tmp
                )
            times.append(t)

    times = np.array(times)
    print("\n──── FastChem timing benchmark ─────────────────────────────────────")
    print(f"samples      : {N_BENCH}")
    print(f"mean         : {times.mean()*1e3:8.2f}  ms")
    print(f"median       : {np.median(times)*1e3:8.2f}  ms")
    print(f"min / max    : {times.min()*1e3:8.2f} / {times.max()*1e3:8.2f}  ms")
    print(f"std‑dev      : {times.std()*1e3:8.2f}  ms")

    # 3) Handy line for surrogate script --------------------------------------------
    print("\nCopy‑paste into surrogate script:")
    print(f"FASTCHEM_PER_CALL = {times.mean():.5f}   # seconds")


if __name__ == "__main__":
    main()
