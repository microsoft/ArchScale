"""
Plot architecture ablation stability metrics.
Left:  Router Z-value for MoE ablations (d=16): SqrtGate+SharedExp vs SharedExp vs SqrtGate
Right: MLP Output RMS for dense ablations (d=20): GatedAttn+QKNorm vs Vanilla vs QKNorm
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

TOKENS_PER_STEP = 113.8e9 / 11285

# --- Left panel: Router Z-value (MoE, d=16) ---
moe_df = pd.read_csv("ablation_router_z_mean.csv")

MOE_CONFIGS = {
    "sgate_shexp": {"label": "SqrtGate + SharedExp", "color": "#2ca02c", "ls": "-"},
    "shexp":       {"label": "SharedExp",            "color": "#ff7f0e", "ls": "-"},
    "sgate":       {"label": "SqrtGate",             "color": "#1f77b4", "ls": "-"},
}

def classify_moe_col(col_name):
    if "__MIN" in col_name or "__MAX" in col_name or col_name == "Step":
        return None
    has_sgate = "sgate" in col_name
    has_shexp = "shexp" in col_name
    if has_sgate and has_shexp:
        return "sgate_shexp"
    elif has_shexp:
        return "shexp"
    elif has_sgate:
        return "sgate"
    return None

# --- Right panel: MLP Output RMS (Dense, d=20) ---
dense_df = pd.read_csv("dense-ablation_mlp_output_rmsnorm_mean.csv")

DENSE_CONFIGS = {
    "ga_qknorm": {"label": "GatedAttn + QKNorm", "color": "#2ca02c", "ls": "-"},
    "vanilla":   {"label": "Baseline",            "color": "#d62728", "ls": "-"},
    "qknorm":    {"label": "QKNorm",             "color": "#1f77b4", "ls": "-"},
}

def classify_dense_col(col_name):
    if "__MIN" in col_name or "__MAX" in col_name or col_name == "Step":
        return None
    has_ga = "_ga_" in col_name
    has_qknorm = "qknorm" in col_name
    if has_ga and has_qknorm:
        return "ga_qknorm"
    elif has_qknorm:
        return "qknorm"
    elif not has_ga and not has_qknorm:
        return "vanilla"
    return None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Router Z-value
for col in moe_df.columns:
    cfg = classify_moe_col(col)
    if cfg is None:
        continue
    sub = moe_df[["Step", col]].dropna()
    tokens_b = sub["Step"].values * TOKENS_PER_STEP / 1e9
    vals = sub[col].astype(float).values
    style = MOE_CONFIGS[cfg]
    ax1.plot(tokens_b, vals, color=style["color"], linestyle=style["ls"],
             label=style["label"], linewidth=1.2, alpha=0.9)

# ax1.set_title("Router $Z$-value (MoE, $d = 16$)")
ax1.set_xlabel("Training Tokens (B)")
ax1.set_ylabel("$Z$-value (LSE$^2$)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# Right: MLP Output RMS
for col in dense_df.columns:
    cfg = classify_dense_col(col)
    if cfg is None:
        continue
    sub = dense_df[["Step", col]].dropna()
    tokens_b = sub["Step"].values * TOKENS_PER_STEP / 1e9
    vals = sub[col].astype(float).values
    style = DENSE_CONFIGS[cfg]
    ax2.plot(tokens_b, vals, color=style["color"], linestyle=style["ls"],
             label=style["label"], linewidth=1.2, alpha=0.9)

# ax2.set_title("MLP Output RMS (Dense, $d = 20$)")
ax2.set_xlabel("Training Tokens (B)")
ax2.set_ylabel("RMS")
ax2.legend(loc="best")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ablation_stability.png", dpi=200, bbox_inches="tight")
print("Saved ablation_stability.png")
