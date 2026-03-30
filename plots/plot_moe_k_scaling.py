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

df = pd.read_csv("moe_k_scaling.csv")

k_values = sorted(df["top_k"].unique())
colors = {2: "#e74c3c", 4: "#3498db", 8: "#2ecc71", 16: "#9b59b6", 32: "#f39c12", 64: "#1abc9c"}
markers = {2: "o", 4: "s", 8: "D", 16: "^", 32: "v", 64: "P"}

fig = plt.figure(figsize=(18, 5.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.15, 1.05],
                       wspace=0.05)
ax_left = fig.add_subplot(gs[0])
ax_mid = fig.add_subplot(gs[1], sharey=ax_left)
ax_right = fig.add_subplot(gs[3])

# --- Left & Middle: LR vs Loss for without / with SqrtGate ---
for ax, sgate, title_suffix in zip(
    [ax_left, ax_mid], [False, True], ["Without SqrtGate", "With SqrtGate"]
):
    for k in k_values:
        subset = df[(df["sqrt_gate"] == sgate) & (df["top_k"] == k)].sort_values("learning_rate")
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

        ax.scatter(lrs, losses, color=colors[k], marker=markers[k], s=50, zorder=5,
                   label=f"k={k} (L*={opt_loss:.3f})")

        log_lr_fine = np.linspace(log_lrs.min() - 0.3, log_lrs.max() + 0.3, 200)
        ax.plot(np.exp(log_lr_fine), poly(log_lr_fine), color=colors[k],
                linewidth=1.5, alpha=0.7)

        ax.scatter([opt_lr], [opt_loss], color=colors[k],
                   marker="*", s=250, edgecolors="black", linewidths=0.8, zorder=6)

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_title(title_suffix)
    ax.legend()
    ax.grid(True, alpha=0.3)

ax_left.set_ylabel("Validation Loss")
plt.setp(ax_mid.get_yticklabels(), visible=False)

# --- Right: Optimal Loss vs k ---
style = {
    False: {"color": "#e74c3c", "label": "Without SqrtGate", "marker": "o"},
    True:  {"color": "#3498db", "label": "With SqrtGate",    "marker": "s"},
}

for sgate in [False, True]:
    ks, opt_losses = [], []
    for k in k_values:
        subset = df[(df["sqrt_gate"] == sgate) & (df["top_k"] == k)].sort_values("learning_rate")
        if subset.empty:
            continue
        lrs = subset["learning_rate"].values
        losses = subset["metric/val_loss@1x"].values
        coeffs = np.polyfit(np.log(lrs), losses, 2)
        opt_loss = np.poly1d(coeffs)(-coeffs[1] / (2 * coeffs[0]))
        ks.append(k)
        opt_losses.append(opt_loss)

    ks = np.array(ks, dtype=float)
    opt_losses = np.array(opt_losses)
    s = style[sgate]

    ax_right.scatter(ks, opt_losses, color=s["color"], marker=s["marker"], s=80, zorder=5,
                     label=s["label"])
    ax_right.plot(ks, opt_losses, color=s["color"], linewidth=1.5, alpha=0.5)

ax_right.set_xscale("log", base=2)
ax_right.set_xlabel("Granularity (k)")
ax_right.set_ylabel("Optimal Validation Loss (L*)")
ax_right.set_xticks(k_values)
ax_right.set_xticklabels([str(k) for k in k_values])
ax_right.legend()
ax_right.grid(True, alpha=0.3)

fig.savefig("moe_k_scaling_combined.png", dpi=150, bbox_inches="tight")
print("Plot saved to moe_k_scaling_combined.png")

for sgate in [False, True]:
    tag = "sgate" if sgate else "no_sgate"
    print(f"\n=== {tag} ===")
    for k in k_values:
        subset = df[(df["sqrt_gate"] == sgate) & (df["top_k"] == k)].sort_values("learning_rate")
        if subset.empty:
            continue
        lrs = subset["learning_rate"].values
        losses = subset["metric/val_loss@1x"].values
        coeffs = np.polyfit(np.log(lrs), losses, 2)
        opt_lr = np.exp(-coeffs[1] / (2 * coeffs[0]))
        opt_loss = np.poly1d(coeffs)(-coeffs[1] / (2 * coeffs[0]))
        print(f"  k={k:>2d}  opt_lr={opt_lr:.6f}  opt_loss={opt_loss:.4f}")
