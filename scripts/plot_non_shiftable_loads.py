"""plotting_non_shiftable_loads"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_PATH = "datasets/citylearn_datasets/citylearn_challenge_2022_phase_all"
OUTPUT_DIR = "results/non_shiftable_load_plots"
N_BUILDINGS = 17
COLUMN_NAME = "non_shiftable_load"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD & PLOT
# --------------------------------------------------
all_buildings = []

for b in range(1, N_BUILDINGS + 1):
    file_path = os.path.join(BASE_PATH, f"Building_{b}.csv")

    if not os.path.exists(file_path):
        print(f"⚠️ Missing: {file_path}")
        continue

    df = pd.read_csv(file_path)

    if COLUMN_NAME not in df.columns:
        raise ValueError(f"'{COLUMN_NAME}' missing in {file_path}")

    all_buildings.append(df[COLUMN_NAME])

    # -------- Plot & Save --------
    plt.figure(figsize=(12, 4))
    plt.plot(df[COLUMN_NAME], linewidth=1)
    plt.title(f"Non-shiftable Load — Building {b}")
    plt.xlabel("Time step")
    plt.ylabel("Load (kWh)")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, f"building_{b}_non_shiftable.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {save_path}")

# --------------------------------------------------
# AVERAGE PLOT
# --------------------------------------------------
import numpy as np

loads_mat = np.column_stack(all_buildings)
avg_load = loads_mat.mean(axis=1)

plt.figure(figsize=(14, 5))
plt.plot(avg_load, linewidth=2, color="black")
plt.title("Average Non-shiftable Load — All 17 Buildings")
plt.xlabel("Time step")
plt.ylabel("Load (kWh)")
plt.grid(True, alpha=0.3)

avg_path = os.path.join(OUTPUT_DIR, "average_non_shiftable.png")
plt.savefig(avg_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ Saved average plot: {avg_path}")
