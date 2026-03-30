"""
Plot transferable stability metrics across depth scales (d8, d12, d16, d20).
Six panels: Attention Z, Router Z, Attention RMSNorm, MLP RMSNorm,
            Attention Outlier %, MLP Outlier %.
Data from W&B CSV exports in stability/ folder.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
})

DEPTHS = [8, 12, 16, 20]
COLORS = {8: "#1f77b4", 12: "#ff7f0e", 16: "#2ca02c", 20: "#d62728"}
LABELS = {d: f"d={d}" for d in DEPTHS}

METRICS = [
    ("stability/attn_z_mean.csv",           "attn_z_mean",           "Attention $Z$-value"),
    ("stability/router_z_mean.csv",         "router_z_mean",         "Router $Z$-value"),
    ("stability/attn_output_rmsnorm_mean.csv", "attn_output_rmsnorm_mean", "Attention Output RMS"),
    ("stability/mlp_output_rmsnorm_mean.csv",  "mlp_output_rmsnorm_mean",  "MoE Output RMS"),
    ("stability/attn_outlier_pct_mean.csv", "attn_outlier_pct_mean", "Attention Output Outlier %"),
    ("stability/mlp_outlier_pct_mean.csv",  "mlp_outlier_pct_mean",  "MoE Output Outlier %"),
]


def extract_depth(col_name):
    for d in sorted(DEPTHS, reverse=True):
        if f"_d{d}_" in col_name:
            return d
    return None


def load_metric(csv_path, metric_name):
    df = pd.read_csv(csv_path)
    result = {}
    for d in DEPTHS:
        mean_cols = [c for c in df.columns
                     if f"_d{d}_" in c
                     and metric_name in c
                     and "__MIN" not in c
                     and "__MAX" not in c]
        if not mean_cols:
            continue
        col = mean_cols[0]
        sub = df[["Step", col]].dropna()
        sub = sub.rename(columns={col: "value"})
        sub["Step"] = sub["Step"].astype(float)
        sub["value"] = sub["value"].astype(float)
        sub = sub.sort_values("Step")
        result[d] = sub
    return result


TOKENS_PER_STEP = 113.8e9 / 11285

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (csv_path, metric_key, title) in enumerate(METRICS):
    ax = axes[idx // 3, idx % 3]
    data = load_metric(csv_path, metric_key)

    for d in DEPTHS:
        if d not in data:
            continue
        sub = data[d]
        tokens = sub["Step"].values * TOKENS_PER_STEP
        tokens_b = tokens / 1e9
        vals = sub["value"].values

        ax.plot(tokens_b, vals, color=COLORS[d], label=LABELS[d],
                linewidth=1.2, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Training Tokens (B)")
    ax.grid(True, alpha=0.3)

    if idx == 0:
        ax.legend(loc="best")

    if "outlier" in metric_key.lower():
        ax.set_ylabel("Outlier %")
    elif "rmsnorm" in metric_key.lower():
        ax.set_ylabel("RMS")
    else:
        ax.set_ylabel("$Z$-value (LSE$^2$)")

plt.tight_layout()
plt.savefig("stability_metrics.png", dpi=200, bbox_inches="tight")
print("Saved stability_metrics.png")
