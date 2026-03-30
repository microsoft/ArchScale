import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

df = pd.read_csv("muonh-moe-ablation.csv")

main_cols = [c for c in df.columns if c != "Step" and "__MIN" not in c and "__MAX" not in c]

def classify_method(name):
    if "sgate_shexp" in name:
        return "SharedExp + SqrtGate"
    elif "sgate" in name:
        return "SqrtGate"
    elif "shexp" in name:
        return "SharedExp"
    else:
        return "Unknown"

def parse_lr(name):
    m = re.search(r'_lr([\d.e\-]+)x', name)
    return float(m.group(1)) if m else None

records = []
for col in main_cols:
    run_name = col.replace(" - metric/val_loss", "")
    method = classify_method(run_name)
    lr = parse_lr(run_name)
    if lr is None:
        continue

    vals = df[col].dropna()
    if len(vals) == 0:
        continue
    final_loss = vals.iloc[-1]

    records.append({
        "method": method,
        "lr": lr,
        "val_loss_1x": final_loss,
    })

data = pd.DataFrame(records)

methods = ["SqrtGate", "SharedExp", "SharedExp + SqrtGate"]
method_colors = {
    "SqrtGate": "#e74c3c",
    "SharedExp": "#2ecc71",
    "SharedExp + SqrtGate": "#3498db",
}
method_linestyles = {
    "SqrtGate": "-",
    "SharedExp": "-",
    "SharedExp + SqrtGate": "-",
}
method_markers = {
    "SqrtGate": "o",
    "SharedExp": "s",
    "SharedExp + SqrtGate": "D",
}

# ============================================================
# Figure 1: Val Loss vs LR + Optimal Loss Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax1 = axes[0]
optimal_results = []

for method in methods:
    subset = data[data["method"] == method].sort_values("lr")
    if len(subset) < 3:
        continue
    lrs = subset["lr"].values
    losses = subset["val_loss_1x"].values
    color = method_colors[method]
    marker = method_markers[method]
    ls = method_linestyles[method]

    log_lrs = np.log(lrs)
    coeffs = np.polyfit(log_lrs, losses, 2)
    poly = np.poly1d(coeffs)

    opt_log_lr = -coeffs[1] / (2 * coeffs[0])
    opt_lr = np.exp(opt_log_lr)
    opt_loss = poly(opt_log_lr)
    optimal_results.append({"method": method, "opt_lr": opt_lr, "opt_loss": opt_loss})

    ax1.scatter(lrs, losses, color=color, s=60, zorder=5,
                label=f"{method} (L*={opt_loss:.3f})",
                marker=marker, edgecolors='black', linewidths=0.3)

    log_lr_fine = np.linspace(log_lrs.min() - 0.3, log_lrs.max() + 0.3, 200)
    ax1.plot(np.exp(log_lr_fine), poly(log_lr_fine), color=color, linewidth=2, alpha=0.8, linestyle=ls)

    ax1.scatter([opt_lr], [opt_loss], color=color, marker='*', s=200,
                edgecolors='black', zorder=6)

ax1.set_xscale("log")
ax1.set_xlabel("Learning Rate", fontsize=14)
ax1.set_ylabel("Validation Loss", fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

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
ax2.set_xticklabels(opt_df["method"], fontsize=9, rotation=15, ha='right')
ax2.set_ylabel("Optimal Validation Loss", fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')
ymin = opt_df["opt_loss"].min() - 0.03
ymax = opt_df["opt_loss"].max() + 0.06
ax2.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig("moe_ablation_analysis.png", dpi=150, bbox_inches="tight")
print("Plot saved to moe_ablation_analysis.png")

# ============================================================
# Figure 2: Training curves over steps for each method
# ============================================================
fig2, ax_curve = plt.subplots(figsize=(10, 6))

for method in methods:
    color = method_colors[method]
    ls = method_linestyles[method]
    marker = method_markers[method]

    method_curves = []
    for col in main_cols:
        run_name = col.replace(" - metric/val_loss@1x", "")
        if classify_method(run_name) != method:
            continue
        lr = parse_lr(run_name)
        if lr is None:
            continue
        vals = df[["Step", col]].dropna(subset=[col])
        if len(vals) <= 1:
            continue
        method_curves.append((lr, vals))

    if not method_curves:
        continue

    for lr, vals in method_curves:
        alpha = 0.3
        ax_curve.plot(vals["Step"], vals[vals.columns[1]], color=color, alpha=alpha,
                      linewidth=1, linestyle=ls)

    best_lr_row = data[(data["method"] == method)].sort_values("val_loss_1x").iloc[0]
    best_lr = best_lr_row["lr"]
    for lr, vals in method_curves:
        if lr == best_lr:
            ax_curve.plot(vals["Step"], vals[vals.columns[1]], color=color,
                          linewidth=2.5, linestyle=ls, label=f"{method} (LR={best_lr})",
                          marker=marker, markersize=5, markeredgecolor='black', markeredgewidth=0.5)
            break

ax_curve.set_xlabel("Training Step", fontsize=13)
ax_curve.set_ylabel("Validation Loss", fontsize=14)
ax_curve.legend(fontsize=9)
ax_curve.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("moe_ablation_curves.png", dpi=150, bbox_inches="tight")
print("Plot saved to moe_ablation_curves.png")

# ============================================================
# Figure 3: Delta vs best method
# ============================================================
fig3, ax3 = plt.subplots(figsize=(10, 5.5))

ref_method = min(optimal_results, key=lambda r: r["opt_loss"])["method"]
ref_loss = min(r["opt_loss"] for r in optimal_results)

x_pos = np.arange(len(optimal_results))
deltas = [r["opt_loss"] - ref_loss for r in optimal_results]
bar_colors = [method_colors[r["method"]] for r in optimal_results]
bars = ax3.bar(x_pos, deltas, color=bar_colors, edgecolor='black', alpha=0.85)

for bar, d, r in zip(bars, deltas, optimal_results):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.001,
             f"{d:+.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax3.axhline(0, color='black', linewidth=0.8)
ax3.set_xticks(x_pos)
ax3.set_xticklabels([r["method"] for r in optimal_results], fontsize=10, rotation=15, ha='right')
ax3.set_ylabel(f"Δ Optimal Loss (vs {ref_method})", fontsize=13)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("moe_ablation_delta.png", dpi=150, bbox_inches="tight")
print("Plot saved to moe_ablation_delta.png")

# ============================================================
# Print summary
# ============================================================
print("\n" + "=" * 70)
print("=== Optimal LR and Loss by Method (Val Loss @1x) ===")
print("=" * 70)
for r in sorted(optimal_results, key=lambda x: x["opt_loss"]):
    print(f"  {r['method']:>25s}  |  Optimal LR: {r['opt_lr']:.6f}  |  Min Loss: {r['opt_loss']:.4f}")

print("\n=== Raw Data (final val_loss@1x per run) ===")
print(f"{'Method':>25s}  |  {'LR':>8s}  |  {'Val Loss':>10s}")
print("-" * 55)
for _, row in data.sort_values(["method", "lr"]).iterrows():
    print(f"  {row['method']:>23s}  |  {row['lr']:>8.4f}  |  {row['val_loss_1x']:>10.4f}")

print(f"\n=== Delta vs {ref_method} ===")
for r in sorted(optimal_results, key=lambda x: x["opt_loss"]):
    d = r["opt_loss"] - ref_loss
    print(f"  {r['method']:>25s}  |  Δ = {d:+.4f}")
