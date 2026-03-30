import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

df = pd.read_csv("muonh_moe_balloss.csv")

records = []
for _, row in df.iterrows():
    name = row["Name"]
    lr = float(row["base_hps.eta0"])
    val_loss = float(row["metric/val_loss@1x"])
    maxvio = float(row["metric/maxvio_batch_avg"])
    m = re.search(r"gblaux([\d.e\-]+)", name)
    aux_weight = m.group(1) if m else "unknown"
    records.append({"lr": lr, "val_loss": val_loss, "maxvio": maxvio, "aux_weight": aux_weight})

data = pd.DataFrame(records)
aux_weights = sorted(data["aux_weight"].unique(), key=float)

fig, ax = plt.subplots(figsize=(7, 5.5))

colors = ["#2196F3", "#4CAF50", "#E53935"]

for aux_w, color in zip(aux_weights, colors):
    subset = data[data["aux_weight"] == aux_w].sort_values("lr")
    lrs = subset["lr"].values
    losses = subset["val_loss"].values

    opt_loss = losses.min()
    ax.scatter(lrs, losses, color=color, s=50, zorder=5,
               label=f"γ = {aux_w}, L*={opt_loss:.3f}")
    ax.plot(lrs, losses, color=color, linewidth=1.5, alpha=0.7)

ax.set_xscale("log")
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("val_loss_vs_lr.png", dpi=150, bbox_inches="tight")
print("Plot saved to val_loss_vs_lr.png")
