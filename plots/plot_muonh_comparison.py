"""
Plot loss vs learning rate for muonh_dep_ah and muonh_ah methods
on transformer_gqa4_h2 with tok10.4e9dscale runs.
LR is extracted from the run name (e.g. lr2e-3 -> 0.002).
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
})

# ============================================================
# Data loading & filtering
# ============================================================

df = pd.read_csv("muoh_dep_scaling.csv")

# Extract LR from name (e.g. "_lr2e-3x" -> 0.002)
def extract_lr(name):
    m = re.search(r"_lr([\d.e\-]+)x", name)
    return float(m.group(1)) if m else None

# --- dscale runs for transformer_gqa4_h2 ---
mask = (
    df["Name"].str.contains("tok10.4e9dscale", na=False)
    & df["Name"].str.contains("transformer_gqa4_h2", na=False)
    & (df["State"] == "finished")
    & df["metric/val_loss@1x"].notna()
)
data = df.loc[mask].copy()

data["method"] = data["Name"].apply(
    lambda x: "muonh_dep_ah" if "muonh_dep_ah" in x
    else ("muonh_ah" if "muonh_ah" in x else "other")
)
data = data[data["method"] != "other"]
data["lr_from_name"] = data["Name"].apply(extract_lr)
data = data.dropna(subset=["lr_from_name"])
data = data[data["lr_from_name"] <= 2e-2]

data = (
    data.sort_values("Runtime", ascending=False)
    .drop_duplicates(subset=["method", "depth", "lr_from_name"], keep="first")
)

# --- non-dscale d8 muonh_ah runs (tok10.4e9 without dscale) ---
mask_nd = (
    df["Name"].str.contains("tok10.4e9_", na=False)
    & ~df["Name"].str.contains("dscale", na=False)
    & df["Name"].str.contains("transformer_gqa4_h2", na=False)
    & df["Name"].str.contains("muonh_ah", na=False)
    & (df["depth"] == 8)
    & (df["State"] == "finished")
    & df["metric/val_loss@1x"].notna()
)
data_nd = df.loc[mask_nd].copy()
data_nd["method"] = "muonh_ah"
data_nd["lr_from_name"] = data_nd["Name"].apply(extract_lr)
data_nd = data_nd.dropna(subset=["lr_from_name"])
data_nd = data_nd[data_nd["lr_from_name"] <= 2e-2]
data_nd = (
    data_nd.sort_values("Runtime", ascending=False)
    .drop_duplicates(subset=["depth", "lr_from_name"], keep="first")
)

# Combine dscale and non-dscale d8 muonh_ah into one dataset
data = pd.concat([data, data_nd], ignore_index=True)
data = (
    data.sort_values("Runtime", ascending=False)
    .drop_duplicates(subset=["method", "depth", "lr_from_name"], keep="first")
)

# ============================================================
# Plotting
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
cmap = matplotlib.colormaps["tab10"]
depths = sorted(data["depth"].unique())
depth_colors = {d: cmap(i) for i, d in enumerate(depths)}
depth_markers = {8: "o", 12: "s", 16: "^", 20: "D", 24: "v"}

for ax, method, title_method in [
    (ax1, "muonh_dep_ah", "muonh_dep_ah (depth-scaled LR)"),
    (ax2, "muonh_ah", "muonh_ah (no depth-mup)"),
]:
    sub = data[data["method"] == method]

    for d in depths:
        grp = sub[sub["depth"] == d].sort_values("lr_from_name")
        if grp.empty:
            continue
        ax.plot(
            grp["lr_from_name"],
            grp["metric/val_loss@1x"],
            marker=depth_markers.get(d, "o"),
            color=depth_colors[d],
            linestyle="-",
            markersize=10,
            linewidth=2,
            label=f"d{int(d)}",
        )
        # Annotate optimal LR
        best_idx = grp["metric/val_loss@1x"].idxmin()
        best_lr = grp.loc[best_idx, "lr_from_name"]
        best_loss = grp.loc[best_idx, "metric/val_loss@1x"]
        ax.plot(
            best_lr, best_loss, "*",
            color=depth_colors[d], markersize=18,
            markeredgecolor="black", markeredgewidth=0.8, zorder=5,
        )
        ax.annotate(
            f"{best_loss:.3f}",
            xy=(best_lr, best_loss),
            xytext=(0, -18),
            textcoords="offset points",
            fontsize=13, color=depth_colors[d],
            ha="center", fontweight="bold",
        )

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Validation Loss")
    ax.legend(title="Model Size")
    ax.grid(True, which='both', linestyle='--', alpha=0.7, linewidth=1.5)

plt.tight_layout()
plt.savefig("muonh_comparison_plots.png", dpi=150, bbox_inches="tight")
print("Saved to muonh_comparison_plots.png")
