import numpy as np
data = np.load("artefacts/splits.npz")
print(data["train_idx"], type(data["train_idx"]))  # should be a scalar