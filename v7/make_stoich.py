#!/usr/bin/env python3
import os, re
import numpy as np, pandas as pd

CSV_PATH = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
ARTE_DIR = "artefacts"
os.makedirs(ARTE_DIR, exist_ok=True)

# load full gas-phase table
df = pd.read_csv(CSV_PATH)

# identify your species columns
INPUTS = {"temperature","pressure","comp_H","comp_O","comp_C","comp_N","comp_S",
          "group_index","point_index"}
species = [c for c in df.columns if c not in INPUTS]

# build stoichiometry matrix A: (#species × #elements)
elements = ["H","O","C","N","S"]
A = np.zeros((len(species), len(elements)), dtype=int)
for i, sp in enumerate(species):
    counts = dict.fromkeys(elements, 0)
    for elem_sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", sp):
        if elem_sym in counts:
            counts[elem_sym] = int(num) if num else 1
    A[i,:] = [counts[e] for e in elements]

# build b: the mean elemental input vector from your comp_* columns
b = df[["comp_H","comp_O","comp_C","comp_N","comp_S"]].mean().values

# save
np.savetxt(os.path.join(ARTE_DIR, "stoich_matrix.csv"),
           A, fmt="%d", delimiter=",")
np.savetxt(os.path.join(ARTE_DIR, "elemental_input_vector.csv"),
           b[None], fmt="%.6f", delimiter=",")

print("Wrote new stoichiometry →", A.shape, "and", b.shape)
