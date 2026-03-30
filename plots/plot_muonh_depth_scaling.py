"""
Plot loss vs training tokens and loss vs FLOPs for transformer_gqa4_h2 architecture.
Uses the estimate_flops logic from lit_gpt/speed_monitor.py to compute FLOPs per token.
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
# FLOPs computation (mirrors estimate_flops in speed_monitor.py)
# ============================================================

def get_n_params(d):
    """
    Compute parameter count for transformer_gqa4_h2_d{d}
    using get_parameters_count() logic from config.py with v2scale train_config.
    """
    ar = 128
    n_head = 2 * d
    head_size = 128
    n_query_groups = 4
    vocab_size = 32000

    # transformer: n_mult = 12 * ar^2
    n_mult = 12 * (ar ** 2)
    n_base = n_mult * (d ** 3)

    # v2scale addition (not tied_embed -> 2x embedding)
    n_head_base = d * ar * vocab_size * 2
    n_base += n_head_base

    # transformer: qo proj + kv proj
    n_base += ar * d**2 * n_head * head_size * 2      # qo proj
    n_base += ar * d**2 * n_query_groups * head_size * 2  # kv proj

    return n_base


def compute_flops_per_token(d):
    """
    Compute training FLOPs per token for transformer_gqa4_h2_d{d}.

    Follows flops_per_param() and estimate_flops() from speed_monitor.py:
      flops_per_seq  = 2 * n_params * block_size
      attn_flops     = n_layer * 2 * 2 * (n_head * head_size * block_size^2)
      estimate_flops = 3 * (flops_per_seq + attn_flops)   [training: fwd+bwd+grad]
      flops_per_token = estimate_flops / block_size
    """
    block_size = 4096
    n_layer = d
    n_head = 2 * d
    head_size = 128
    n_params = get_n_params(d)

    # flops_per_param
    flops_per_seq = 2 * n_params * block_size
    ctx_flops_per_seq = block_size ** 2                       # full attention (no local window)
    attn_flops_per_seq = n_layer * 2 * 2 * (n_head * head_size * ctx_flops_per_seq)
    total_per_seq = flops_per_seq + attn_flops_per_seq

    # Training mode: forward + backward + gradients = 3×
    total_per_seq *= 3

    return total_per_seq / block_size


# ============================================================
# Data loading & filtering
# ============================================================

df = pd.read_csv("muonh-depth-scaling.csv")

# Keep only transformer_gqa4_h2, finished runs, with valid val_loss
mask = (
    (df["train_model"] == "transformer_gqa4_h2")
    & (df["State"] == "finished")
    & df["metric/val_loss@1x"].notna()
)
df_h2 = df.loc[mask].copy()

# Deduplicate: keep the run with longer runtime for each (depth, lr, train_tokens) combo
df_h2 = (
    df_h2.sort_values("Runtime", ascending=False)
    .drop_duplicates(subset=["depth", "learning_rate", "train_tokens"], keep="first")
)

# Compute total FLOPs = flops_per_token * train_tokens
df_h2["flops_per_token"] = df_h2["depth"].apply(compute_flops_per_token)
df_h2["total_flops"] = df_h2["flops_per_token"] * df_h2["train_tokens"]

# Print summary
print("=== Parameter counts and FLOPs per token ===")
for d in sorted(df_h2["depth"].unique()):
    n = get_n_params(int(d))
    fpt = compute_flops_per_token(int(d))
    print(f"  d={int(d):2d}:  n_params = {n:>14,}   flops/token = {fpt:.4e}")
print()

# ============================================================
# Plotting
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- Color & marker setup ---
lr_list = sorted(df_h2["learning_rate"].unique())
cmap = matplotlib.colormaps["tab10"]
depth_markers = {8: "o", 12: "s", 16: "^", 20: "D", 24: "v"}

# ----------------------------------------------------------------
# Plot 1: muonh + hyperP — Loss vs base LR (different depths / FLOPs)
# ----------------------------------------------------------------
df_muonh = pd.read_csv("muonh-flops-scaling.csv")

mask_muonh = (
    (df_muonh["State"] == "finished")
    & df_muonh["metric/val_loss@1x"].notna()
)
df_muonh = df_muonh.loc[mask_muonh].copy()

df_muonh = (
    df_muonh.sort_values("Runtime", ascending=False)
    .drop_duplicates(subset=["depth", "base_hps.eta0"], keep="first")
)

df_muonh["flops_per_token"] = df_muonh["depth"].apply(compute_flops_per_token)
df_muonh["total_flops"] = df_muonh["flops_per_token"] * df_muonh["train_tokens"]

muonh_depths = [8, 12, 16, 20]#sorted(df_muonh["depth"].unique())
muonh_depth_colors = {d: cmap(i) for i, d in enumerate(muonh_depths)}

annot_idx_m = 0
for d in muonh_depths:
    grp = df_muonh[df_muonh["depth"] == d].sort_values("base_hps.eta0")
    if grp.empty:
        continue
    flops_val = grp["total_flops"].iloc[0]
    label = f"d{int(d)}, {flops_val:.1e} FLOPs"
    ax1.plot(
        grp["base_hps.eta0"],
        grp["metric/val_loss@1x"],
        marker=depth_markers.get(int(d), "o"),
        color=muonh_depth_colors[d],
        linestyle="-",
        markersize=10,
        linewidth=2,
        label=label,
    )
    best_idx = grp["metric/val_loss@1x"].idxmin()
    best_lr = grp.loc[best_idx, "base_hps.eta0"]
    best_loss = grp.loc[best_idx, "metric/val_loss@1x"]
    ax1.plot(best_lr, best_loss, "*", color=muonh_depth_colors[d], markersize=18,
             markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    y_off = -22 if annot_idx_m % 2 == 0 else 16
    ax1.annotate(
        f"{best_loss:.3f}",
        xy=(best_lr, best_loss),
        xytext=(0, y_off),
        textcoords="offset points",
        fontsize=13,
        color=muonh_depth_colors[d],
        ha="center",
        fontweight="bold",
    )
    annot_idx_m += 1

ax1.set_xlabel("Learning Rate")
ax1.set_ylabel("Validation Loss")
ax1.set_xscale("log")
ax1.legend(title="Scale", loc="upper left")
ax1.grid(True, which='both', linestyle='--', alpha=0.7, linewidth=1.5)

# ----------------------------------------------------------------
# Plot 2: Loss vs LR for d8, d12, d16, d20 under different FLOPs
# ----------------------------------------------------------------
target_depths = [8, 12, 16, 20]
sub_all = df_h2[df_h2["depth"].isin(target_depths)].copy()

# Group by (depth, train_tokens) — each combo is a distinct total_flops budget
groups = (
    sub_all.groupby(["depth", "train_tokens"])
    .first()
    .reset_index()[["depth", "train_tokens", "total_flops"]]
    .drop_duplicates()
    .sort_values("total_flops")
)
flops_budgets = sorted(groups["total_flops"].unique())
flops_colors = {f: cmap(i % 10) for i, f in enumerate(flops_budgets)}

# For d8 keep only the smallest flops (smallest train_tokens);
# for d12 and d16 keep only the largest flops (largest train_tokens);
# for d20 keep everything (only one token budget).
for d in [8, 12, 16]:
    d_mask = sub_all["depth"] == d
    toks = sub_all.loc[d_mask, "train_tokens"]
    if d == 8:
        keep_tok = toks.min()
    else:
        keep_tok = toks.max()
    drop_mask = d_mask & (sub_all["train_tokens"] != keep_tok)
    sub_all = sub_all[~drop_mask]

annot_idx = 0  # to alternate annotation offset and avoid overlap
for d in target_depths:
    d_data = sub_all[sub_all["depth"] == d]
    for tok, grp in d_data.groupby("train_tokens"):
        grp = grp.sort_values("learning_rate")
        flops_val = grp["total_flops"].iloc[0]
        label = f"d{d}, {flops_val:.1e} FLOPs"
        color = ax2.plot(
            grp["learning_rate"],
            grp["metric/val_loss@1x"],
            marker=depth_markers[d],
            linestyle="-",
            markersize=10,
            linewidth=2,
            label=label,
        )[0].get_color()
        best_idx = grp["metric/val_loss@1x"].idxmin()
        best_lr = grp.loc[best_idx, "learning_rate"]
        best_loss = grp.loc[best_idx, "metric/val_loss@1x"]
        y_off = -22 if annot_idx % 2 == 0 else 16
        ax2.annotate(
            f"{best_loss:.3f}",
            xy=(best_lr, best_loss),
            xytext=(0, y_off),
            textcoords="offset points",
            fontsize=13,
            color=color,
            ha="center",
            fontweight="bold",
        )
        ax2.plot(best_lr, best_loss, "*", color=color, markersize=18,
                 markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        annot_idx += 1

ax2.set_xlabel("Learning Rate")
ax2.set_ylabel("Validation Loss")
ax2.set_xscale("log")
ax2.grid(True, which='both', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.legend(title="Scale", loc="upper left")

plt.tight_layout()
plt.savefig("transformer_gqa4_h2_plots.png", dpi=150, bbox_inches="tight")
print("Saved plot to transformer_gqa4_h2_plots.png")

# ============================================================
# Quadratic-fit figure (separate)
# ============================================================

def quad_fit_optimum(lrs, losses):
    """Fit loss = a*(log(lr))^2 + b*log(lr) + c, return (opt_lr, opt_loss, coeffs)."""
    log_lrs = np.log(lrs)
    coeffs = np.polyfit(log_lrs, losses, 2)
    a, b, c = coeffs
    if a <= 0:
        return None
    log_lr_opt = -b / (2 * a)
    loss_opt = np.polyval(coeffs, log_lr_opt)
    return np.exp(log_lr_opt), loss_opt, coeffs

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 7))

# ----------------------------------------------------------------
# Panel 1: muonh + hyperP — quadratic fit (different depths / FLOPs)
# ----------------------------------------------------------------
annot_idx_mq = 0
for d in muonh_depths:
    grp = df_muonh[df_muonh["depth"] == d].sort_values("base_hps.eta0")
    if grp.empty or len(grp) < 3:
        continue
    lrs = grp["base_hps.eta0"].values
    losses = grp["metric/val_loss@1x"].values
    flops_val = grp["total_flops"].iloc[0]
    color = muonh_depth_colors[d]
    label = f"d{int(d)}, {flops_val:.1e} FLOPs"

    ax3.plot(lrs, losses, marker=depth_markers.get(int(d), "o"),
             linestyle="none", color=color, markersize=10, label=label)

    result = quad_fit_optimum(lrs, losses)
    if result is None:
        continue
    opt_lr, opt_loss, coeffs = result

    lr_smooth = np.geomspace(lrs.min() * 0.8, lrs.max() * 1.2, 200)
    loss_smooth = np.polyval(coeffs, np.log(lr_smooth))
    ax3.plot(lr_smooth, loss_smooth, "-", color=color, linewidth=2, alpha=0.7)

    ax3.plot(opt_lr, opt_loss, "*", color=color, markersize=18,
             markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    y_off = -22 if annot_idx_mq % 2 == 0 else 16
    ax3.annotate(
        f"{opt_loss:.3f}",
        xy=(opt_lr, opt_loss),
        xytext=(0, y_off),
        textcoords="offset points",
        fontsize=13, color=color, ha="center", fontweight="bold",
    )
    annot_idx_mq += 1

ax3.set_xlabel("Base Learning Rate (eta0)")
ax3.set_ylabel("Validation Loss")
ax3.legend(title="Depth / FLOPs", loc="upper left")
ax3.set_xscale("log")
ax3.grid(True, which='both', linestyle='--', alpha=0.7, linewidth=1.5)

# ----------------------------------------------------------------
# Panel 2: d8/d12/d16/d20 at ~50TPP — quadratic fit
# ----------------------------------------------------------------
annot_idx2 = 0
for d in target_depths:
    d_data = sub_all[sub_all["depth"] == d]
    for tok, grp in d_data.groupby("train_tokens"):
        grp = grp.sort_values("learning_rate")
        if len(grp) < 3:
            continue
        lrs = grp["learning_rate"].values
        losses = grp["metric/val_loss@1x"].values
        flops_val = grp["total_flops"].iloc[0]
        label = f"d{d}, {flops_val:.1e} FLOPs"

        color = ax4.plot(
            lrs, losses,
            marker=depth_markers[d], linestyle="none",
            markersize=10, label=label,
        )[0].get_color()

        result = quad_fit_optimum(lrs, losses)
        if result is None:
            continue
        opt_lr, opt_loss, coeffs = result

        lr_smooth = np.geomspace(lrs.min() * 0.8, lrs.max() * 1.2, 200)
        loss_smooth = np.polyval(coeffs, np.log(lr_smooth))
        ax4.plot(lr_smooth, loss_smooth, "-", color=color, linewidth=2, alpha=0.7)

        ax4.plot(opt_lr, opt_loss, "*", color=color, markersize=18,
                 markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        y_off = -22 if annot_idx2 % 2 == 0 else 16
        ax4.annotate(
            f"{opt_loss:.3f}",
            xy=(opt_lr, opt_loss),
            xytext=(0, y_off),
            textcoords="offset points",
            fontsize=13, color=color, ha="center", fontweight="bold",
        )
        annot_idx2 += 1

ax4.set_xlabel("Learning Rate")
ax4.set_ylabel("Validation Loss")
ax4.set_xscale("log")
ax4.grid(True, which='both', linestyle='--', alpha=0.7, linewidth=1.5)
ax4.legend(ncol=2, loc="upper left")

plt.tight_layout()
plt.savefig("transformer_gqa4_h2_quadfit.png", dpi=150, bbox_inches="tight")
print("Saved quadratic-fit plot to transformer_gqa4_h2_quadfit.png")
