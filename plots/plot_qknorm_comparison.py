import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
})

df = pd.read_csv("qknorm_vs_ga_qknorm_vs_baseline.csv")

def classify_method(name):
    if "ga_qknorm_nonorm" in name:
        return "NoNorm"
    elif "ga_qknorm" in name:
        return "GA QK-Norm"
    elif "qknorm" in name:
        return "QK-Norm"
    else:
        return "Baseline"

def parse_lr(name):
    m = re.search(r'_lr([\d.e\-]+)x', name)
    return float(m.group(1)) if m else None

records = []
for _, row in df.iterrows():
    if row["State"] != "finished":
        continue
    method = classify_method(row["Name"])
    lr = parse_lr(row["Name"])
    if lr is not None:
        records.append({
            "method": method,
            "lr": lr,
            "val_loss_1x": row["metric/val_loss@1x"],
            "val_loss_2x": row["metric/val_loss@2x"],
            "val_loss_3x": row["metric/val_loss@3x"],
            "val_loss_4x": row["metric/val_loss@4x"],
            "train_loss": row["metric/train_loss"],
        })

data = pd.DataFrame(records)
methods = ["GA QK-Norm", "QK-Norm", "Baseline"]
method_colors = {
    "GA QK-Norm": "#e74c3c",
    "QK-Norm": "#3498db",
    "Baseline": "#2ecc71",
}
method_linestyles = {
    "GA QK-Norm": "-",
    "QK-Norm": "--",
    "Baseline": "-.",
    "NoNorm": ":",
}
method_markers = {
    "GA QK-Norm": "o",
    "QK-Norm": "s",
    "Baseline": "D",
    "NoNorm": "^",
}

# ============================================================
# Figure 1: Val Loss vs LR + Optimal Loss Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax1 = axes[0]
optimal_results = []

LR_EXCLUDE = 0.02

for method in methods:
    subset = data[data["method"] == method].sort_values("lr")
    if len(subset) < 3:
        continue
    lrs = subset["lr"].values
    losses = subset["val_loss_1x"].values
    color = method_colors[method]

    marker = method_markers[method]
    ls = method_linestyles[method]

    fit_mask = lrs != LR_EXCLUDE
    ax1.scatter(lrs[fit_mask], losses[fit_mask], color=color, s=60, zorder=5, label=method,
                marker=marker, edgecolors='black', linewidths=0.5)
    ax1.scatter(lrs[~fit_mask], losses[~fit_mask], color=color, s=60, zorder=5,
                marker=marker, edgecolors='black', linewidths=0.5, alpha=0.35)

    log_lrs = np.log(lrs[fit_mask])
    coeffs = np.polyfit(log_lrs, losses[fit_mask], 2)
    poly = np.poly1d(coeffs)

    log_lr_fine = np.linspace(log_lrs.min() - 0.3, log_lrs.max() + 0.3, 200)
    ax1.plot(np.exp(log_lr_fine), poly(log_lr_fine), color=color, linewidth=2, alpha=0.8, linestyle=ls)

    opt_log_lr = -coeffs[1] / (2 * coeffs[0])
    opt_lr = np.exp(opt_log_lr)
    opt_loss = poly(opt_log_lr)
    optimal_results.append({"method": method, "opt_lr": opt_lr, "opt_loss": opt_loss})

    ax1.scatter([opt_lr], [opt_loss], color=color, marker='*', s=200,
                edgecolors='black', zorder=6)
    ax1.annotate(f"{opt_loss:.4f}", (opt_lr, opt_loss),
                 textcoords="offset points", xytext=(10, -10),
                 fontsize=8, fontweight='bold', color=color)

ax1.set_xscale("log")
ax1.set_xlabel("Learning Rate")
ax1.set_ylabel("Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: bar chart comparing optimal losses
ax2 = axes[1]
opt_df = pd.DataFrame(optimal_results)
x_pos = np.arange(len(opt_df))
bars = ax2.bar(x_pos, opt_df["opt_loss"],
               color=[method_colors[m] for m in opt_df["method"]],
               edgecolor='black', alpha=0.85)

for bar, row in zip(bars, opt_df.itertuples()):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
             f"LR={row.opt_lr:.4f}\nLoss={row.opt_loss:.4f}",
             ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(opt_df["method"], fontsize=10)
ax2.set_ylabel("Optimal Validation Loss")
ax2.grid(True, alpha=0.3, axis='y')
ymin = opt_df["opt_loss"].min() - 0.03
ymax = opt_df["opt_loss"].max() + 0.06
ax2.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig("qknorm_analysis.png", dpi=150, bbox_inches="tight")
print("Plot saved to qknorm_analysis.png")

# ============================================================
# Figure 2: Multi-horizon val loss comparison
# ============================================================
horizons = ["val_loss_1x", "val_loss_2x", "val_loss_3x", "val_loss_4x"]
horizon_labels = ["@1x", "@2x", "@3x", "@4x"]

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))

for idx, (hcol, hlabel) in enumerate(zip(horizons, horizon_labels)):
    ax = axes2[idx // 2][idx % 2]
    for method in methods:
        subset = data[data["method"] == method].sort_values("lr")
        if len(subset) < 3:
            continue
        lrs = subset["lr"].values
        losses = subset[hcol].values
        color = method_colors[method]

        marker = method_markers[method]
        ls = method_linestyles[method]
        fit_mask = lrs != LR_EXCLUDE
        ax.scatter(lrs[fit_mask], losses[fit_mask], color=color, s=60, zorder=5, label=method,
                   marker=marker, edgecolors='black', linewidths=0.5)
        ax.scatter(lrs[~fit_mask], losses[~fit_mask], color=color, s=60, zorder=5,
                   marker=marker, edgecolors='black', linewidths=0.5, alpha=0.35)

        log_lrs = np.log(lrs[fit_mask])
        coeffs = np.polyfit(log_lrs, losses[fit_mask], 2)
        poly = np.poly1d(coeffs)

        log_lr_fine = np.linspace(log_lrs.min() - 0.3, log_lrs.max() + 0.3, 200)
        ax.plot(np.exp(log_lr_fine), poly(log_lr_fine), color=color, linewidth=2, alpha=0.8, linestyle=ls)

        opt_log_lr = -coeffs[1] / (2 * coeffs[0])
        opt_lr = np.exp(opt_log_lr)
        opt_loss = poly(opt_log_lr)

        ax.scatter([opt_lr], [opt_loss], color=color, marker='*', s=200,
                   edgecolors='black', zorder=6)
        ax.annotate(f"{opt_loss:.3f}", (opt_lr, opt_loss),
                    textcoords="offset points", xytext=(10, -10),
                    fontsize=7, fontweight='bold', color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel(f"Validation Loss {hlabel}")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("qknorm_multihorizon.png", dpi=150, bbox_inches="tight")
print("Plot saved to qknorm_multihorizon.png")

# ============================================================
# Figure 3: optimal loss delta vs baseline across horizons
# ============================================================
fig3, ax3 = plt.subplots(figsize=(10, 5.5))

baseline_opt = {}
all_opt = {}

for hcol, hlabel in zip(horizons, horizon_labels):
    for method in methods:
        subset = data[data["method"] == method].sort_values("lr")
        if len(subset) < 3:
            continue
        all_lrs = subset["lr"].values
        fit_mask = all_lrs != LR_EXCLUDE
        log_lrs = np.log(all_lrs[fit_mask])
        losses = subset[hcol].values[fit_mask]
        coeffs = np.polyfit(log_lrs, losses, 2)
        poly = np.poly1d(coeffs)
        opt_log_lr = -coeffs[1] / (2 * coeffs[0])
        opt_loss = poly(opt_log_lr)
        all_opt[(method, hlabel)] = opt_loss
        if method == "Baseline":
            baseline_opt[hlabel] = opt_loss

x = np.arange(len(horizon_labels))
bar_width = 0.2
offsets = {m: i for i, m in enumerate(methods)}

for method in methods:
    deltas = []
    for hlabel in horizon_labels:
        if (method, hlabel) in all_opt and hlabel in baseline_opt:
            deltas.append(all_opt[(method, hlabel)] - baseline_opt[hlabel])
        else:
            deltas.append(0)
    offset = (offsets[method] - 1.5) * bar_width
    bars = ax3.bar(x + offset, deltas, bar_width, label=method,
                   color=method_colors[method], edgecolor='black', alpha=0.85)
    for bar, d in zip(bars, deltas):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.003 if d >= 0 else -0.015),
                 f"{d:+.4f}", ha='center', va='bottom', fontsize=7, fontweight='bold')

ax3.axhline(0, color='black', linewidth=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(horizon_labels, fontsize=12)
ax3.set_xlabel("Eval Horizon")
ax3.set_ylabel("Δ Optimal Loss (vs Baseline)")
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("qknorm_delta_baseline.png", dpi=150, bbox_inches="tight")
print("Plot saved to qknorm_delta_baseline.png")

# ============================================================
# Print summary
# ============================================================
print("\n" + "=" * 70)
print("=== Optimal LR and Loss by Method (Val Loss @1x) ===")
print("=" * 70)
for r in optimal_results:
    print(f"  {r['method']:>15s}  |  Optimal LR: {r['opt_lr']:.6f}  |  Min Loss: {r['opt_loss']:.4f}")

print("\n=== Multi-Horizon Optimal Losses ===")
print(f"{'Method':>15s}  |  {'@1x':>8s}  {'@2x':>8s}  {'@3x':>8s}  {'@4x':>8s}")
print("-" * 65)
for method in methods:
    vals = []
    for hlabel in horizon_labels:
        v = all_opt.get((method, hlabel), float('nan'))
        vals.append(f"{v:.4f}")
    print(f"  {method:>13s}  |  {'  '.join(vals)}")

print("\n=== Delta vs Baseline ===")
print(f"{'Method':>15s}  |  {'@1x':>8s}  {'@2x':>8s}  {'@3x':>8s}  {'@4x':>8s}")
print("-" * 65)
for method in methods:
    vals = []
    for hlabel in horizon_labels:
        v = all_opt.get((method, hlabel), float('nan'))
        b = baseline_opt.get(hlabel, float('nan'))
        vals.append(f"{v - b:+.4f}")
    print(f"  {method:>13s}  |  {'  '.join(vals)}")
