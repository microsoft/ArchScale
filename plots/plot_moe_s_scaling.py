import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
})

df = pd.read_csv("moe_s_scaling.csv")

s_values = sorted(df["sparsity"].unique())
colors = {1: "#e74c3c", 2: "#3498db", 4: "#2ecc71", 8: "#9b59b6", 16: "#f39c12", 32: "#1abc9c"}
markers = {1: "o", 2: "s", 4: "D", 8: "^", 16: "v", 32: "P"}

from scipy.optimize import curve_fit

def power_law_offset(x, A, b, C):
    return A * x**b + C

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Left panel: LR vs Val Loss for each sparsity ---
opt_s_list = []
opt_lr_list = []
opt_loss_list = []

for s in s_values:
    subset = df[df["sparsity"] == s].sort_values("learning_rate")
    if subset.empty:
        continue
    lrs = subset["learning_rate"].values
    losses = subset["metric/val_loss@1x"].values

    log_lrs = np.log(lrs)
    coeffs = np.polyfit(log_lrs, losses, 2)
    poly = np.poly1d(coeffs)

    opt_log_lr = -coeffs[1] / (2 * coeffs[0])
    opt_lr = np.exp(opt_log_lr)
    opt_loss = poly(opt_log_lr)

    opt_s_list.append(s)
    opt_lr_list.append(opt_lr)
    opt_loss_list.append(opt_loss)

    ax1.scatter(lrs, losses, color=colors[s], marker=markers[s], s=50, zorder=5,
                label=f"S={s} (L*={opt_loss:.3f})")

    log_lr_fine = np.linspace(log_lrs.min() - 0.3, log_lrs.max() + 0.3, 200)
    ax1.plot(np.exp(log_lr_fine), poly(log_lr_fine), color=colors[s],
             linewidth=1.5, alpha=0.7)

    ax1.scatter([opt_lr], [opt_loss], color=colors[s],
                marker="*", s=250, edgecolors="black", linewidths=0.8, zorder=6)

ax1.set_xscale("log")
ax1.set_xlabel("Learning Rate")
ax1.set_ylabel("Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Right panel: Optimal Loss vs Sparsity with power-law fit ---
opt_s = np.array(opt_s_list, dtype=float)
opt_lrs = np.array(opt_lr_list)
opt_losses = np.array(opt_loss_list)

popt_loss, _ = curve_fit(power_law_offset, opt_s, opt_losses, p0=[0.5, -0.3, 2.2], maxfev=10000)

ax2.scatter(opt_s, opt_losses, color="#FF5722", s=80, zorder=5, marker="o")
s_fine = np.linspace(0.8, 40, 200)
ax2.plot(s_fine, power_law_offset(s_fine, *popt_loss), color="#FF5722", linewidth=1.5, alpha=0.7,
         label=rf"$L^* = {popt_loss[0]:.3f} \cdot S^{{{popt_loss[1]:.3f}}} + {popt_loss[2]:.3f}$")
ax2.set_xscale("log", base=2)
ax2.set_xlabel("Sparsity (S)")
ax2.set_ylabel("Optimal Validation Loss (L*)")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(opt_s)
ax2.set_xticklabels([str(int(x)) for x in opt_s])

fig.tight_layout()
fig.savefig("moe_s_scaling_laws.png", dpi=150, bbox_inches="tight")
print("Plot saved to moe_s_scaling_laws.png")

print("\n=== Optimal LR per Sparsity ===")
for s, lr, loss in zip(opt_s_list, opt_lr_list, opt_loss_list):
    print(f"  s={s:>2d}  opt_lr={lr:.6f}  opt_loss={loss:.4f}")

print(f"\n=== Power-law fit: L* = {popt_loss[0]:.4e} · S^({popt_loss[1]:.4f}) + {popt_loss[2]:.4f} ===")
