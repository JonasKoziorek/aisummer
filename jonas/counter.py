# %%
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from time import time
import pandas as pd

# %%
path = Path("/mnt/c/Users/pepaz/Downloads/ai_summer_school_dataset/ai_summer_school_dataset/splits/split_seals/train.csv")
# %%


df = pd.read_csv(path, delimiter=";")
numbers = df["number"].to_numpy()
divisors = 10 ** np.arange(6, -1, -1)
digit_matrix = (numbers[:, None] // divisors) % 10
print(digit_matrix.shape)
print(digit_matrix)

col_unique_counts = {
    col_idx: dict(zip(*np.unique(digit_matrix[:, col_idx], return_counts=True)))
    for col_idx in range(digit_matrix.shape[1])
}

print(col_unique_counts)

# %%
digit_position = 0
unique_values_and_counts = col_unique_counts[digit_position]

# %%
vals, counts = np.unique(digit_matrix, return_counts=True)
overall_counts = dict(zip(vals, counts))
overall_counts

# %%
