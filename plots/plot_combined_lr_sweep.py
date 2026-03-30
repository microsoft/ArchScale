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

# ── QK-Norm data ──────────────────────────────────────────────
df_qk = pd.read_csv("qknorm_vs_ga_qknorm_vs_baseline.csv")

def classify_qknorm(name):
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

records_qk = []
for _, row in df_qk.iterrows():
    if row["State"] != "finished":
        continue
    method = classify_qknorm(row["Name"])
    lr = parse_lr(row["Name"])
    if lr is not None:
        records_qk.append({"method": method, "lr": lr, "val_loss_1x": row["metric/val_loss@1x"]})

data_qk = pd.DataFrame(records_qk)
qk_methods = ["GA QK-Norm", "QK-Norm", "Baseline"]
qk_colors = {"GA QK-Norm": "#e74c3c", "QK-Norm": "#3498db", "Baseline": "#2ecc71"}
qk_linestyles = {"GA QK-Norm": "-", "QK-Norm": "--", "Baseline": "-."}
qk_markers = {"GA QK-Norm": "o", "QK-Norm": "s", "Baseline": "D"}
LR_EXCLUDE = 0.02

# ── MoE ablation data ────────────────────────────────────────
df_moe = pd.read_csv("muonh-moe-ablation.csv")
moe_cols = [c for c in df_moe.columns if c != "Step" and "__MIN" not in c and "__MAX" not in c]

def classify_moe(name):
    if "sgate_shexp" in name:
        return "SharedExp + SqrtGate"
    elif "sgate" in name:
        return "SqrtGate"
    elif "shexp" in name:
        return "SharedExp"
    else:
        return "Unknown"

records_moe = []
for col in moe_cols:
    run_name = col.replace(" - metric/val_loss", "")
    method = classify_moe(run_name)
    lr = parse_lr(run_name)
    if lr is None:
        continue
    vals = df_moe[col].dropna()
    if len(vals) == 0:
        continue
    records_moe.append({"method": method, "lr": lr, "val_loss_1x": vals.iloc[-1]})

data_moe = pd.DataFrame(records_moe)
moe_methods = ["SqrtGate", "SharedExp", "SharedExp + SqrtGate"]
moe_colors = {"SqrtGate": "#3498db", "SharedExp": "#2ecc71", "SharedExp + SqrtGate": "#e74c3c"}
moe_linestyles = {"SqrtGate": "-.", "SharedExp": "--", "SharedExp + SqrtGate": "-"}
moe_markers = {"SqrtGate": "o", "SharedExp": "s", "SharedExp + SqrtGate": "D"}

# ── Combined figure ───────────────────────────────────────────
fig, (ax_qk, ax_moe) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left panel: QK-Norm LR sweep
for method in qk_methods:
    subset = data_qk[data_qk["method"] == method].sort_values("lr")
    if len(subset) < 3:
        continue
    lrs = subset["lr"].values
    losses = subset["val_loss_1x"].values
    color = qk_colors[method]
    marker = qk_markers[method]
    ls = qk_linestyles[method]

    fit_mask = lrs != LR_EXCLUDE
    ax_qk.scatter(lrs[fit_mask], losses[fit_mask], color=color, s=60, zorder=5,
                   label=method, marker=marker, edgecolors='black', linewidths=0.5)
    ax_qk.scatter(lrs[~fit_mask], losses[~fit_mask], color=color, s=60, zorder=5,
                   marker=marker, edgecolors='black', linewidths=0.5, alpha=0.35)

    log_lrs = np.log(lrs[fit_mask])
    coeffs = np.polyfit(log_lrs, losses[fit_mask], 2)
    poly = np.poly1d(coeffs)

    log_lr_fine = np.linspace(log_lrs.min() - 0.3, log_lrs.max() + 0.3, 200)
    ax_qk.plot(np.exp(log_lr_fine), poly(log_lr_fine), color=color, linewidth=2, alpha=0.8, linestyle=ls)

    opt_log_lr = -coeffs[1] / (2 * coeffs[0])
    opt_lr = np.exp(opt_log_lr)
    opt_loss = poly(opt_log_lr)

    ax_qk.scatter([opt_lr], [opt_loss], color=color, marker='*', s=200,
                   edgecolors='black', zorder=6)
    ax_qk.annotate(f"{opt_loss:.4f}", (opt_lr, opt_loss),
                    textcoords="offset points", xytext=(10, -10),
                    fontsize=8, fontweight='bold', color=color)

ax_qk.set_xscale("log")
ax_qk.set_xlabel("Learning Rate")
ax_qk.set_ylabel("Validation Loss")
ax_qk.legend()
ax_qk.grid(True, alpha=0.3)
from matplotlib.ticker import FixedLocator, FuncFormatter
ax_qk.xaxis.set_major_locator(FixedLocator([0.004, 0.006, 0.01, 0.02]))
ax_qk.xaxis.set_minor_locator(FixedLocator([]))
ax_qk.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))

# Right panel: MoE ablation LR sweep
for method in moe_methods:
    subset = data_moe[data_moe["method"] == method].sort_values("lr")
    if len(subset) < 3:
        continue
    lrs = subset["lr"].values
    losses = subset["val_loss_1x"].values
    color = moe_colors[method]
    marker = moe_markers[method]
    ls = moe_linestyles[method]

    log_lrs = np.log(lrs)
    coeffs = np.polyfit(log_lrs, losses, 2)
    poly = np.poly1d(coeffs)

    opt_log_lr = -coeffs[1] / (2 * coeffs[0])
    opt_lr = np.exp(opt_log_lr)
    opt_loss = poly(opt_log_lr)

    ax_moe.scatter(lrs, losses, color=color, s=60, zorder=5,
                    label=f"{method} (L*={opt_loss:.3f})",
                    marker=marker, edgecolors='black', linewidths=0.3)

    log_lr_fine = np.linspace(log_lrs.min() - 0.3, log_lrs.max() + 0.3, 200)
    ax_moe.plot(np.exp(log_lr_fine), poly(log_lr_fine), color=color, linewidth=2, alpha=0.8, linestyle=ls)

    ax_moe.scatter([opt_lr], [opt_loss], color=color, marker='*', s=200,
                    edgecolors='black', zorder=6)

ax_moe.set_xscale("log")
ax_moe.set_xlabel("Learning Rate")
ax_moe.set_ylabel("Validation Loss")
ax_moe.legend()
ax_moe.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("combined_lr_sweep.png", dpi=150, bbox_inches="tight")
print("Plot saved to combined_lr_sweep.png")
