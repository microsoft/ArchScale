"""
Line plot: learning rate (x) vs validation loss (y), with weight decay as legend,
for Muon optimizer + MuonH (muonh_dep_ah d8), with quadratic fit in log(lr) space.
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

# ── Muon data ──
muon = pd.read_csv("muon_optimal.csv")
lr_muon = muon["learning_rate"].values
wd_muon = muon["weight_decay"].values
loss_muon = muon["metric/val_loss@1x"].values

# ── MuonH data (muonh_dep_ah, dscale, d8) ──
df_h = pd.read_csv("muoh_dep_scaling.csv")

def extract_lr(name):
    m = re.search(r"_lr([\d.e\-]+)x", name)
    return float(m.group(1)) if m else None

mask_h = (
    df_h["Name"].str.contains("tok10.4e9dscale", na=False)
    & df_h["Name"].str.contains("transformer_gqa4_h2", na=False)
    & df_h["Name"].str.contains("muonh_dep_ah", na=False)
    & (df_h["depth"] == 8)
    & (df_h["State"] == "finished")
    & df_h["metric/val_loss@1x"].notna()
)
data_h = df_h.loc[mask_h].copy()
data_h["lr_from_name"] = data_h["Name"].apply(extract_lr)
data_h = data_h.dropna(subset=["lr_from_name"])
muonh_lrs = {0.002, 0.004, 0.008, 0.01, 0.02}
data_h = data_h[data_h["lr_from_name"].isin(muonh_lrs)]
data_h = data_h.sort_values("lr_from_name")

lr_h = data_h["lr_from_name"].values
loss_h = data_h["metric/val_loss@1x"].values

# ── Plotting ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
cmap = plt.cm.viridis

wd_unique = np.sort(np.unique(wd_muon))
colors = cmap(np.linspace(0.1, 0.9, len(wd_unique)))

print(f"{'Method':>16s}  {'Weight Decay':>14s}  {'Fitted opt LR':>14s}  {'Fitted opt Loss':>16s}  {'R²':>8s}")
print("-" * 76)

def fit_and_plot(ax, lr_arr, loss_arr, color, label, marker="o"):
    order = np.argsort(lr_arr)
    lr_arr, loss_arr = lr_arr[order], loss_arr[order]
    ax.plot(lr_arr, loss_arr, marker, color=color, label=label, markersize=6, zorder=3)

    log_lr = np.log10(lr_arr)
    coeffs = np.polyfit(log_lr, loss_arr, 2)
    log_lr_fine = np.linspace(log_lr.min(), log_lr.max(), 200)
    loss_fit = np.polyval(coeffs, log_lr_fine)
    ax.plot(10**log_lr_fine, loss_fit, "-", color=color, linewidth=1.5, alpha=0.8)

    lr_opt = 10 ** (-coeffs[1] / (2 * coeffs[0]))
    loss_opt = np.polyval(coeffs, np.log10(lr_opt))
    ax.plot(lr_opt, loss_opt, "*", color=color, markersize=14, zorder=5,
            markeredgecolor="black", markeredgewidth=0.5)

    ss_res = np.sum((loss_arr - np.polyval(coeffs, log_lr)) ** 2)
    ss_tot = np.sum((loss_arr - np.mean(loss_arr)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return lr_opt, loss_opt, r2

# ── Left panel: Muon ──
muon_fits = {}
for i, w in enumerate(wd_unique):
    mask = wd_muon == w
    lr_opt, loss_opt, r2 = fit_and_plot(ax1, lr_muon[mask], loss_muon[mask], colors[i], f"{w:.0e}")
    muon_fits[w] = (lr_opt, loss_opt, r2)
    print(f"{'Muon':>16s}  {w:>14.0e}  {lr_opt:>14.4e}  {loss_opt:>16.4f}  {r2:>8.4f}")

best_muon_wd = min(muon_fits, key=lambda w: muon_fits[w][1])
muon_best_lr, muon_best_loss, _ = muon_fits[best_muon_wd]

ax1.set_xscale("log")
ax1.set_xlabel("Learning Rate")
ax1.set_ylabel("Validation Loss")
ax1.legend(title="Weight Decay",title_fontsize=13)
ax1.grid(True, alpha=0.3)

# ── Right panel: MuonH ──
muonh_color = "tab:red"
lr_opt_h, loss_opt_h, r2_h = fit_and_plot(ax2, lr_h, loss_h, muonh_color, "MuonH", marker="D")
print(f"{'MuonH':>16s}  {'0':>14s}  {lr_opt_h:>14.4e}  {loss_opt_h:>16.4f}  {r2_h:>8.4f}")

ax2.set_xscale("log")
ax2.set_xlabel("Learning Rate")
ax2.set_ylabel("Validation Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)


plt.tight_layout()
out = "optimal_hparams_lines.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out}")
