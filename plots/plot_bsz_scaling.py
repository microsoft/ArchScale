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

df = pd.read_csv("muonh_bsz_optimal.csv")

def parse_name(name):
    bsz_match = re.search(r'_bsz(\d+)_', name)
    lr_match = re.search(r'_lr([\d.e\-]+)x', name)
    tok_match = re.search(r'_tok([\d.e]+)', name)
    if bsz_match and lr_match and tok_match:
        bsz = int(bsz_match.group(1))
        lr = float(lr_match.group(1))
        tok = float(tok_match.group(1))
        return bsz, lr, tok
    return None, None, None

records = []
for _, row in df.iterrows():
    bsz, lr, tokens = parse_name(row["Name"])
    if bsz is not None:
        records.append({
            "bsz": bsz,
            "lr": lr,
            "tokens": tokens,
            "val_loss": row["metric/val_loss@1x"],
        })

data = pd.DataFrame(records)
batch_sizes = sorted(data["bsz"].unique())

# --- Panel 1: Val loss vs LR for each batch size ---
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))

ax1 = axes[0]
colors = plt.cm.tab10(np.linspace(0, 0.8, len(batch_sizes)))
optimal_lrs = []

for bsz, color in zip(batch_sizes, colors):
    subset = data[data["bsz"] == bsz].sort_values("lr")
    lrs = subset["lr"].values
    losses = subset["val_loss"].values

    label = f"bsz={bsz//1024}K" if bsz < 1048576 else f"bsz={bsz//1048576}M"
    ax1.scatter(lrs, losses, color=color, s=50, zorder=5, label=label)

    log_lrs = np.log(lrs)
    coeffs = np.polyfit(log_lrs, losses, 2)
    poly = np.poly1d(coeffs)

    log_lr_fine = np.linspace(log_lrs.min() - 0.3, log_lrs.max() + 0.3, 200)
    ax1.plot(np.exp(log_lr_fine), poly(log_lr_fine), color=color, linewidth=1.5, alpha=0.7)

    opt_log_lr = -coeffs[1] / (2 * coeffs[0])
    opt_lr = np.exp(opt_log_lr)
    opt_loss = poly(opt_log_lr)
    optimal_lrs.append((bsz, opt_lr, opt_loss))

    ax1.scatter([opt_lr], [opt_loss], color=color, marker='*', s=200,
                edgecolors='black', zorder=6)
    ax1.annotate(f"{opt_loss:.3f}", (opt_lr, opt_loss),
                 textcoords="offset points", xytext=(8, -12),
                 fontsize=8, fontweight='bold', color=color)

ax1.set_xscale("log")
ax1.set_xlabel("Learning Rate")
ax1.set_ylabel("Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

opt_df = pd.DataFrame(optimal_lrs, columns=["bsz", "opt_lr", "opt_loss"])

# --- Panel 2: Optimal LR vs BSZ with power law fits ---
def power_law(x, a, b):
    return a * x ** b

def power_law_irr(x, a, b, c):
    return a * x ** b + c

popt, _ = curve_fit(power_law, opt_df["bsz"].values, opt_df["opt_lr"].values, p0=[1.0, 0.3])
a_fit, b_fit = popt

popt_irr, _ = curve_fit(
    power_law_irr, opt_df["bsz"].values, opt_df["opt_lr"].values,
    p0=[1.0, 0.3, 0.001], maxfev=10000
)
a_irr, b_irr, c_irr = popt_irr

ax2 = axes[1]
ax2.scatter(opt_df["bsz"], opt_df["opt_lr"], s=100, color="royalblue",
            edgecolors="black", zorder=5, label="Optimal LR (quadratic fit)")

bsz_fine = np.geomspace(opt_df["bsz"].min() * 0.3, opt_df["bsz"].max() * 5, 200)
ax2.plot(bsz_fine, power_law(bsz_fine, *popt), "r--", linewidth=1.5, alpha=0.6,
         label=f"PL: {a_fit:.4g} * B^({b_fit:.4f})")
# ax2.plot(bsz_fine, power_law_irr(bsz_fine, *popt_irr), "darkgreen", linewidth=2,
#          label=f"PL+irr: {a_irr:.4g} * B^({b_irr:.4f}) + {c_irr:.5f}")
# ax2.axhline(c_irr, color="gray", linestyle=":", alpha=0.6,
#             label=f"Irreducible LR = {c_irr:.5f}")

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Batch Size (tokens)")
ax2.set_ylabel("Optimal Learning Rate")
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

# # --- Panel 3: Leave-one-out extrapolation ---
# all_bsz = opt_df["bsz"].values
# all_opt_lrs = opt_df["opt_lr"].values
# n = len(all_bsz)

# pl_errors = []
# irr_errors = []
# rows = []

# for i in range(n):
#     mask = np.arange(n) != i
#     train_B = all_bsz[mask]
#     train_lr = all_opt_lrs[mask]
#     test_B = all_bsz[i]
#     test_lr = all_opt_lrs[i]

#     try:
#         p1, _ = curve_fit(power_law, train_B, train_lr, p0=[1.0, 0.3])
#         pred_pl = power_law(test_B, *p1)
#     except RuntimeError:
#         pred_pl = np.nan

#     try:
#         p2, _ = curve_fit(power_law_irr, train_B, train_lr, p0=[1.0, 0.3, 0.001], maxfev=10000)
#         pred_irr = power_law_irr(test_B, *p2)
#     except RuntimeError:
#         pred_irr = np.nan

#     err_pl = (pred_pl - test_lr) / test_lr * 100
#     err_irr = (pred_irr - test_lr) / test_lr * 100
#     pl_errors.append(abs(err_pl))
#     irr_errors.append(abs(err_irr))
#     rows.append((test_B, test_lr, pred_pl, err_pl, pred_irr, err_irr))

# ax3 = axes[2]
# x_pos = np.arange(n)
# bsz_labels = [f"{b//1024}K" if b < 1048576 else f"{b//1048576}M" for b in all_bsz]

# ax3.bar(x_pos - 0.18, [r[3] for r in rows], 0.35, color="salmon", edgecolor="black",
#         label="Power Law err%")
# ax3.bar(x_pos + 0.18, [r[5] for r in rows], 0.35, color="lightgreen", edgecolor="black",
#         label="PL + irr err%")
# ax3.axhline(0, color="black", linewidth=0.5)
# ax3.set_xticks(x_pos)
# ax3.set_xticklabels(bsz_labels)
# ax3.set_xlabel("Held-out Batch Size", fontsize=13)
# ax3.set_ylabel("Prediction Error (%)", fontsize=13)
# ax3.set_title("Leave-One-Out Extrapolation\nError Comparison", fontsize=13)
# ax3.legend(fontsize=10)
# ax3.grid(True, alpha=0.3, axis="y")

# pl_mean = np.mean(pl_errors)
# irr_mean = np.mean(irr_errors)
# ax3.text(0.98, 0.02, f"Mean |err|: PL={pl_mean:.2f}%, PL+irr={irr_mean:.2f}%",
#          transform=ax3.transAxes, fontsize=9, ha="right", va="bottom",
#          bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

plt.tight_layout()
plt.savefig("bsz_scaling_analysis.png", dpi=150, bbox_inches="tight")
print("Plot saved to bsz_scaling_analysis.png")

print("\n=== Optimal Learning Rates per Batch Size ===")
for bsz, lr, loss in optimal_lrs:
    label = f"{bsz//1024}K" if bsz < 1048576 else f"{bsz//1048576}M"
    print(f"  BSZ: {label:>5s} ({bsz:>8d})  |  Optimal LR: {lr:.6f}  |  Min Loss: {loss:.4f}")

print(f"\n=== Power Law Fit (no irreducible) ===")
print(f"  optimal_lr = {a_fit:.6g} * bsz^({b_fit:.6f})")

print(f"\n=== Power Law Fit (with irreducible LR) ===")
print(f"  optimal_lr = {a_irr:.6g} * bsz^({b_irr:.6f}) + {c_irr:.6f}")
print(f"  Irreducible LR: {c_irr:.6f}")

# print(f"\n{'='*70}")
# print("=== Leave-One-Out Extrapolation ===")
# print("="*70)
# print(f"{'Held-out':>10s} | {'True LR':>10s} | {'PL pred':>10s} | {'PL err%':>8s} | {'PL+irr pred':>12s} | {'PL+irr err%':>11s}")
# print("-"*70)
# for b, true, pred_p, err_p, pred_i, err_i in rows:
#     label = f"{b//1024}K" if b < 1048576 else f"{b//1048576}M"
#     print(f"  {label:>6s}    | {true:>10.6f} | {pred_p:>10.6f} | {err_p:>+7.2f}% | {pred_i:>12.6f} | {err_i:>+10.2f}%")
# print("-"*70)
# print(f"  Mean |err%|:  Power Law = {np.mean(pl_errors):.2f}%    Power Law + irr = {np.mean(irr_errors):.2f}%")
