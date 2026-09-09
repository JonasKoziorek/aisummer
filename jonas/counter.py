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
digit_position = 1
unique_values_and_counts = col_unique_counts[digit_position]
unique_values_and_counts

# %%
vals, counts = np.unique(digit_matrix, return_counts=True)
overall_counts = dict(zip(vals, counts))
overall_counts

# %%
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Setup Data ---
# Replace with your actual array `arr` of 9,902 seven-digit integers
# arr = np.array([1585678, 1939859, ...])
divisors = 10 ** np.arange(6, -1, -1)
digit_matrix = (numbers[:, None] // divisors) % 10

# Total pooled counts (digits 0 to 9 across all elements)
total_counts = np.bincount(digit_matrix.ravel(), minlength=10)

# Column-by-column counts (digits 0 to 9 for each position)
col_counts = [np.bincount(digit_matrix[:, i], minlength=10) for i in range(7)]

# --- 2. Color Palette ---
# Fix colors so digit 0 through 9 always have the exact same color across every pie chart
cmap = plt.cm.tab10
palette = {d: cmap(d) for d in range(10)}
digits = np.arange(10)

# --- 3. Plotting in a 2x4 Subplot Grid ---
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()

# Subplot 0: Total overall distribution
mask_total = total_counts > 0
axes[0].pie(
    total_counts[mask_total],
    labels=digits[mask_total],
    colors=[palette[d] for d in digits[mask_total]],
    autopct="%1.1f%%",
    startangle=90,
)
axes[0].set_title("Total Overall Digit Counts", fontsize=12, fontweight="bold")

# Subplots 1 to 7: Distribution per position
for i in range(7):
    ax = axes[i + 1]
    counts = col_counts[i]
    mask = counts > 0  # Only show digits that actually appear (e.g. position 1 often has no 0)

    ax.pie(
        counts[mask],
        labels=digits[mask],
        colors=[palette[d] for d in digits[mask]],
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",  # Hide tiny labels to prevent overlap
        startangle=90,
    )
    ax.set_title(f"Position {i + 1} (10^{6 - i}s place)", fontsize=11)

plt.tight_layout()
plt.show()

# %%
