"""
Compare MoE scaling curves (loss vs FLOPs) across gate configurations.
  - SharedExp+SqrtGate  (sgate_shexp, LR=1.4e-2)
  - SharedExp only       (shexp,       LR=1.4e-2)
  - SqrtGate only        (sgate,       LR=1.4e-2)
  - SharedExp+SqrtGate  (sgate_shexp, LR=1e-2)

Data: muonh-moe-flops-scaling.csv  (W&B panel export, Step = training step)
FLOPs: same dense-equivalent formula as plot_scaling_comparison.py
Fits:  L = A · F^(-b)   (no irreducible-loss offset — few data points)
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
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
# FLOPs computation  (dense-equivalent, per user request)
# ============================================================

def get_n_params(d):
    ar = 128
    n_head = 2 * d
    head_size = 128
    n_query_groups = 4
    vocab_size = 32000
    n_mult = 12 * (ar ** 2)
    n_base = n_mult * (d ** 3)
    n_head_base = d * ar * vocab_size * 2
    n_base += n_head_base
    n_base += ar * d**2 * n_head * head_size * 2
    n_base += ar * d**2 * n_query_groups * head_size * 2
    return n_base


def compute_flops_per_token(d):
    block_size = 4096
    n_layer = d
    n_head = 2 * d
    head_size = 128
    n_params = get_n_params(d)
    flops_per_seq = 2 * n_params * block_size
    ctx_flops_per_seq = block_size ** 2
    attn_flops_per_seq = n_layer * 2 * 2 * (n_head * head_size * ctx_flops_per_seq)
    total_per_seq = flops_per_seq + attn_flops_per_seq
    total_per_seq *= 3
    return total_per_seq / block_size


TPP50_TOKENS = {
    8:  10_400_000_000,
    12: 28_488_664_987,
    16: 61_876_070_528,
    20: 115_591_939_546,
    24: 194_665_994_962,
}


def power_law(F, A, b):
    return A * F ** (-b)


# ============================================================
# Parse MoE CSV
# ============================================================

df = pd.read_csv("muonh-moe-flops-scaling.csv")

loss_cols = [
    c for c in df.columns
    if "metric/val_loss@1x" in c and "__MIN" not in c and "__MAX" not in c
]


def classify_method(name):
    if "sgate_shexp" in name:
        return "SharedExp+SqrtGate"
    elif "sgate" in name:
        return "SqrtGate"
    elif "shexp" in name:
        return "SharedExp"
    return "Vanilla"


def parse_depth(name):
    m = re.search(r"_d(\d+)_ctx", name)
    return int(m.group(1)) if m else None


def parse_lr(name):
    m = re.search(r"_lr([\d.e\-]+)x", name)
    return float(m.group(1)) if m else None


records = []
for col in loss_cols:
    run_name = col.split(" - ")[0]
    method = classify_method(run_name)
    d = parse_depth(run_name)
    lr = parse_lr(run_name)
    if d is None or lr is None:
        continue

    vals = df[["Step", col]].copy()
    vals[col] = pd.to_numeric(vals[col], errors="coerce")
    vals = vals.dropna(subset=[col])
    if vals.empty:
        continue

    final_loss = float(vals[col].iloc[-1])
    total_tokens = TPP50_TOKENS.get(d)
    if total_tokens is None:
        continue
    flops = compute_flops_per_token(d) * total_tokens

    records.append(dict(
        method=method, depth=d, lr=lr,
        total_tokens=total_tokens,
        flops=flops, loss=final_loss,
    ))

data = pd.DataFrame(records)
data = data[data["depth"] != 20]

# ============================================================
# Build series  (only those with ≥ 3 depths)
# ============================================================

SERIES_ORDER = [
    ("SharedExp+SqrtGate", 0.01),
    ("SharedExp",         0.014),
    ("SqrtGate",          0.014),
    ("SharedExp+SqrtGate", 0.014),
]

COLORS = {
    ("SqrtGate",          0.014): "#e74c3c",
    ("SharedExp",         0.014): "#2ecc71",
    ("SharedExp+SqrtGate", 0.01):  "#ff7f0e",
    ("SharedExp+SqrtGate", 0.014): "#3498db",
}

MARKERS = {
    ("SqrtGate",          0.014): "s",
    ("SharedExp",         0.014): "D",
    ("SharedExp+SqrtGate", 0.01):  "^",
    ("SharedExp+SqrtGate", 0.014): "o",
}

series = {}
for method, lr in SERIES_ORDER:
    mask = (data["method"] == method) & np.isclose(data["lr"], lr)
    pts = data[mask].sort_values("flops").reset_index(drop=True)
    if len(pts) < 3:
        print(f"  Skipping {method} LR={lr}: only {len(pts)} depth(s)")
        continue
    label = f"{method} (LR={lr})"
    series[label] = dict(
        pts=pts,
        color=COLORS[(method, lr)],
        marker=MARKERS[(method, lr)],
    )

# ============================================================
# Power-law fits:  L = A · F^(−b)
# ============================================================

fit_params = {}
for label, s in series.items():
    flops = s["pts"]["flops"].values
    losses = s["pts"]["loss"].values

    log_f = np.log(flops)
    log_l = np.log(losses)
    slope, intercept = np.polyfit(log_f, log_l, 1)
    b_guess = -slope
    a_guess = np.exp(intercept)

    try:
        params, _ = curve_fit(
            power_law, flops, losses,
            p0=[a_guess, max(b_guess, 1e-4)],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=50_000,
        )
        fit_params[label] = params
        A, b = params
        print(f"  {label}:  L = {A:.4e} · F^(−{b:.4f})")
    except RuntimeError:
        print(f"  Could not fit {label}")

# ============================================================
# Print summary table
# ============================================================

print("\n=== Scaling points ===")
for label, s in series.items():
    print(f"\n  {label}:")
    for _, r in s["pts"].iterrows():
        print(f"    d{int(r['depth']):2d}  "
              f"tokens={r['total_tokens']:.2e}  FLOPs={r['flops']:.2e}  loss={r['loss']:.4f}")

# ============================================================
# Plot: scaling comparison (left) + compute leverage (right)
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- Left panel: Loss vs FLOPs ---
for label, s in series.items():
    pts = s["pts"]
    flops = pts["flops"].values
    losses = pts["loss"].values

    ax1.plot(flops, losses, marker=s["marker"], linestyle="none",
             label=label, color=s["color"], markersize=10,
             markeredgecolor="black", markeredgewidth=0.5)

    if label in fit_params:
        A, b = fit_params[label]
        f_range = np.geomspace(flops.min() * 0.7, flops.max() * 1.4, 200)
        ax1.plot(f_range, power_law(f_range, A, b),
                 linestyle="--", color=s["color"],
                 label=f"L = {A:.2e}·C^(-{b:.3f})",
                 linewidth=2, alpha=0.8)


ax1.set_xscale("log")
ax1.set_xlabel("Training FLOPs (C)")
ax1.set_ylabel("Validation Loss (L)")
ax1.legend(loc="upper right")
ax1.grid(True, which="both", linestyle="--", alpha=0.7, linewidth=1.5)

# --- Right panel: Compute Efficiency Leverage ---
baseline_label = list(series.keys())[0]
print(f"\n=== Compute Efficiency Leverage (baseline = {baseline_label}) ===")

if baseline_label in fit_params:
    A_base, b_base = fit_params[baseline_label]

    def invert_power_law(loss, A, b):
        return (loss / A) ** (-1.0 / b)

    for label, s in series.items():
        if label == baseline_label:
            continue
        if label not in fit_params:
            continue

        alt_pts = s["pts"]
        alt_flops = alt_pts["flops"].values
        alt_losses = alt_pts["loss"].values
        alt_depths = alt_pts["depth"].values

        baseline_equiv_flops = invert_power_law(alt_losses, A_base, b_base)
        leverage = baseline_equiv_flops / alt_flops

        ax2.plot(alt_flops, leverage, marker=s["marker"], linestyle="-",
                 label=label, color=s["color"], markersize=10, linewidth=2,
                 markeredgecolor="black", markeredgewidth=0.5)

        print(f"\n  {label}:")
        for d_val, f_val, l_val, eq_f, lev in zip(
                alt_depths, alt_flops, alt_losses, baseline_equiv_flops, leverage):
            print(f"    d{int(d_val):2d}  FLOPs={f_val:.2e}  loss={l_val:.4f}"
                  f"  baseline equiv FLOPs={eq_f:.2e}  leverage={lev:.2f}×")

    ax2.axhline(y=1.0, color="gray", linestyle=":", linewidth=2,
                label=f"{baseline_label} baseline")
    ax2.set_xscale("log")
    ax2.set_xlabel("Training FLOPs")
    ax2.set_ylabel("Compute Efficiency Leverage")
    ax2.legend(loc="upper left")
    ax2.grid(True, which="both", linestyle="--", alpha=0.7, linewidth=1.5)

fig.tight_layout()
fig.savefig("moe_scaling_comparison.png", dpi=300, bbox_inches="tight")
print("\nSaved to moe_scaling_comparison.png")
