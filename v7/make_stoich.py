
#!/usr/bin/env python3
import os, re
import numpy as np, pandas as pd

CSV_PATH = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
ARTE_DIR = "artefacts"
os.makedirs(ARTE_DIR, exist_ok=True)

# load data
df = pd.read_csv(CSV_PATH)

# find the 116 species
INPUTS = {"temperature","pressure",
          "comp_H","comp_O","comp_C","comp_N","comp_S",
          "group_index","point_index"}
species = [c for c in df.columns if c not in INPUTS]

# build stoichiometry matrix A (116×5)
elements = ["H","O","C","N","S"]
A = np.zeros((len(species), len(elements)), int)
for i, sp in enumerate(species):
    counts = dict.fromkeys(elements,0)
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", sp):
        if elem in counts:
            counts[elem] = int(num) if num else 1
    A[i,:] = [counts[e] for e in elements]

# build b = mean elemental fractions
b = df[["comp_H","comp_O","comp_C","comp_N","comp_S"]].mean().values

# write them out
np.savetxt(f"{ARTE_DIR}/stoich_matrix.csv",             A, fmt="%d", delimiter=",")
np.savetxt(f"{ARTE_DIR}/elemental_input_vector.csv", b[None], fmt="%.6f", delimiter=",")
print("wrote artefacts/stoich_matrix.csv and elemental_input_vector.csv")
