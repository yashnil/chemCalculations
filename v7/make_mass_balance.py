
#!/usr/bin/env python3
import re, os
import numpy as np
import pandas as pd

# — paths —
CSV_PATH  = "/Users/yashnilmohanty/Desktop/FastChem-Materials/tables/all_gas.csv"
ARTE_DIR  = "artefacts"
os.makedirs(ARTE_DIR, exist_ok=True)

# — load your full gas dataset —
df = pd.read_csv(CSV_PATH)

# — figure out which columns are species —
INPUTS = {"temperature","pressure","comp_H","comp_O","comp_C","comp_N","comp_S",
          "group_index","point_index"}
species = [c for c in df.columns if c not in INPUTS]

# — elements in fixed order —
elements = ["H","O","C","N","S"]

# — build A: species × elements —
A = np.zeros((len(species), len(elements)), dtype=int)
for i, sp in enumerate(species):
    # parse formula like "H2O" or "CO" etc.
    counts = dict.fromkeys(elements, 0)
    for elem_sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", sp):
        if elem_sym in counts:
            counts[elem_sym] = int(num) if num else 1
    A[i,:] = [counts[e] for e in elements]

# — build b: mean elemental composition from your comp_* columns —
b = df[["comp_H","comp_O","comp_C","comp_N","comp_S"]].mean().values

# — save CSVs —
np.savetxt(os.path.join(ARTE_DIR, "stoich_matrix.csv"), A, fmt="%d", delimiter=",")
np.savetxt(os.path.join(ARTE_DIR, "elemental_input_vector.csv"),
           b[None], fmt="%.6f", delimiter=",")
print("Wrote:\n • artefacts/stoich_matrix.csv  (shape {})\n • artefacts/elemental_input_vector.csv  (shape {})"
      .format(A.shape, b.shape))
