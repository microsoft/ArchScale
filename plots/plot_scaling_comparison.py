"""
Compare scaling curves (loss vs FLOPs) between:
  - muonh hyperP: muonh with muP hyperparameter transfer (muonh-flops-scaling.csv)
  - muonh:        muonh without hyperP transfer (muonh_ah from muoh_dep_scaling.csv)
  - muon hyperP:  muon with muP hyperparameter transfer (muon_flops_scaling.csv)
All use the observed d8-optimal LR for all depths. Power-law fits overlaid.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import re

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 20,
})

# ============================================================
# FLOPs computation
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

def power_law_with_offset(F, A, b, C):
    return A * F**(-b) + C

# ============================================================
# 1. muonh hyperP  (muonh-flops-scaling.csv)
# ============================================================

df_hp = pd.read_csv("muonh-flops-scaling.csv")
df_hp = df_hp[(df_hp["State"] == "finished") & df_hp["metric/val_loss@1x"].notna()].copy()
df_hp = (
    df_hp.sort_values("Runtime", ascending=False)
    .drop_duplicates(subset=["depth", "base_hps.eta0"], keep="first")
)
df_hp = df_hp[~np.isclose(df_hp["base_hps.eta0"], 0.002)]

d8_hp = df_hp[df_hp["depth"] == 8]
hp_d8_best_idx = d8_hp["metric/val_loss@1x"].idxmin()
hp_d8_opt_lr = d8_hp.loc[hp_d8_best_idx, "base_hps.eta0"]
print(f"muonh hyperP  d8 optimal base LR: {hp_d8_opt_lr}")

hp_points = []
for d, tok in TPP50_TOKENS.items():
    row = df_hp[(df_hp["depth"] == d) & np.isclose(df_hp["base_hps.eta0"], hp_d8_opt_lr)]
    if row.empty:
        continue
    flops = compute_flops_per_token(d) * tok
    loss = row["metric/val_loss@1x"].values[0]
    hp_points.append({"depth": d, "flops": flops, "loss": loss})

hp_pts = pd.DataFrame(hp_points)

# ============================================================
# 2. muonh (no hyperP)  —  muonh_ah from muoh_dep_scaling.csv
# ============================================================

df_ah = pd.read_csv("muoh_dep_scaling.csv")
df_ah = df_ah[
    (df_ah["State"] == "finished")
    & df_ah["metric/val_loss@1x"].notna()
    & df_ah["Name"].str.contains("muonh_ah", na=False)
    & ~df_ah["Name"].str.contains("muonh_dep_ah", na=False)
].copy()

def extract_lr(name):
    m = re.search(r"_lr([\d.e\-]+)x", name)
    return float(m.group(1)) if m else None

df_ah["lr_from_name"] = df_ah["Name"].apply(extract_lr)
df_ah = df_ah.dropna(subset=["lr_from_name"])
df_ah = (
    df_ah.sort_values("Runtime", ascending=False)
    .drop_duplicates(subset=["depth", "lr_from_name", "train_tokens"], keep="first")
)

common_lrs = None
for d in [8, 12, 16, 20]:
    tok = TPP50_TOKENS[d]
    d_sub = df_ah[(df_ah["depth"] == d) & np.isclose(df_ah["train_tokens"], tok, rtol=0.01)]
    lrs_avail = set(d_sub["lr_from_name"].unique())
    common_lrs = lrs_avail if common_lrs is None else common_lrs & lrs_avail

print(f"muonh_ah common LRs at 50 TPP (d8–d20): {sorted(common_lrs)}")

d8_ah = df_ah[
    (df_ah["depth"] == 8)
    & np.isclose(df_ah["train_tokens"], TPP50_TOKENS[8], rtol=0.01)
    & df_ah["lr_from_name"].isin(common_lrs)
]
ah_d8_best_idx = d8_ah["metric/val_loss@1x"].idxmin()
ah_d8_opt_lr = d8_ah.loc[ah_d8_best_idx, "lr_from_name"]
print(f"muonh (no hyperP)  d8 optimal LR: {ah_d8_opt_lr}")

ah_points = []
for d, tok in TPP50_TOKENS.items():
    row = df_ah[
        (df_ah["depth"] == d)
        & np.isclose(df_ah["train_tokens"], tok, rtol=0.01)
        & np.isclose(df_ah["lr_from_name"], ah_d8_opt_lr)
    ]
    if row.empty:
        continue
    flops = compute_flops_per_token(d) * tok
    loss = row["metric/val_loss@1x"].values[0]
    ah_points.append({"depth": d, "flops": flops, "loss": loss})

ah_pts = pd.DataFrame(ah_points)

# ============================================================
# 3. Muon hyperP  (muon_flops_scaling.csv)
# ============================================================

df_muon = pd.read_csv("muon_flops_scaling.csv")
df_muon = df_muon[(df_muon["State"] == "finished") & df_muon["metric/val_loss@1x"].notna()].copy()
df_muon = (
    df_muon.sort_values("Runtime", ascending=False)
    .drop_duplicates(subset=["depth", "base_hps.eta0"], keep="first")
)

muon_d8 = df_muon[df_muon["depth"] == 8]
muon_d8_best_idx = muon_d8["metric/val_loss@1x"].idxmin()
muon_d8_opt_lr = muon_d8.loc[muon_d8_best_idx, "base_hps.eta0"]
print(f"Muon hyperP  d8 optimal base LR: {muon_d8_opt_lr}")

muon_points = []
for d, tok in TPP50_TOKENS.items():
    row = df_muon[(df_muon["depth"] == d) & np.isclose(df_muon["base_hps.eta0"], muon_d8_opt_lr)]
    if row.empty:
        continue
    flops = compute_flops_per_token(d) * tok
    loss = row["metric/val_loss@1x"].values[0]
    muon_points.append({"depth": d, "flops": flops, "loss": loss})

muon_pts = pd.DataFrame(muon_points)

# ============================================================
# 4. MoE  (muonh-moe-flops-scaling.csv)
# ============================================================

df_moe_raw = pd.read_csv("muonh-moe-flops-scaling.csv")

moe_loss_cols = [
    c for c in df_moe_raw.columns
    if 'metric/val_loss@1x' in c and '__MIN' not in c and '__MAX' not in c
    and 'sgate_shexp' in c and '_lr1e-2' in c
]

moe_points = []
for col in moe_loss_cols:
    m = re.search(r'_d(\d+)_ctx', col)
    if not m:
        continue
    d = int(m.group(1))

    sub = df_moe_raw[['Step', col]].copy()
    sub = sub[sub[col].notna() & (sub[col] != '')]
    sub[col] = sub[col].astype(float)
    if sub.empty:
        continue

    loss = sub[col].iloc[-1]
    tokens = TPP50_TOKENS.get(d)
    if tokens is None:
        continue
    flops = compute_flops_per_token(d) * tokens
    moe_points.append({"depth": d, "flops": flops, "loss": loss, "tokens": tokens})
    print(f"  MoE d{d}: tokens={tokens:.3e}  loss={loss:.4f}  FLOPs={flops:.3e}")

moe_pts = pd.DataFrame(moe_points).sort_values("depth")

# ============================================================
# Print summary
# ============================================================

print("\n=== Scaling points ===")
print(f"{'depth':>5s}  {'FLOPs':>12s}  {'hyperP':>10s}  {'muonh':>10s}  {'muon':>10s}")
merged = hp_pts.merge(ah_pts, on=["depth", "flops"], suffixes=("_hp", "_ah"), how="outer")
merged = merged.merge(muon_pts, on=["depth", "flops"], how="outer")
merged = merged.rename(columns={"loss": "loss_muon"})
for _, r in merged.sort_values("flops").iterrows():
    hp_l = f"{r.get('loss_hp', np.nan):.4f}" if pd.notna(r.get("loss_hp")) else "—"
    ah_l = f"{r.get('loss_ah', np.nan):.4f}" if pd.notna(r.get("loss_ah")) else "—"
    mu_l = f"{r.get('loss_muon', np.nan):.4f}" if pd.notna(r.get("loss_muon")) else "—"
    print(f"  d{int(r['depth']):2d}  {r['flops']:>12.2e}  {hp_l:>10s}  {ah_l:>10s}  {mu_l:>10s}")

# ============================================================
# Power-law fits
# ============================================================

markers = ['o', 'D', 's', '^']
cmap = plt.cm.get_cmap('tab10', 4)

series = {
    f'Muon': (muon_pts, cmap(1)),
    f'MuonH': (ah_pts, cmap(2)),
    f'MuonH+HyperP': (hp_pts, cmap(3)),
    'MuonH+HyperP MoE': (moe_pts, cmap(0)),
}

fit_params = {}
for label, (pts, color) in series.items():
    flops = pts["flops"].values
    losses = pts["loss"].values
    c_guess = np.min(losses) * 0.95
    b_guess = 0.1
    c_guess_safe = min(c_guess, losses[0] * 0.95)
    a_guess = (losses[0] - c_guess_safe) * (flops[0] ** b_guess)
    if a_guess <= 0:
        a_guess = 1.0
    try:
        params, _ = curve_fit(
            power_law_with_offset, flops, losses,
            p0=[a_guess, b_guess, c_guess],
            bounds=([0, 0, 0], [np.inf, np.inf, np.min(losses)]),
            maxfev=50000,
        )
        fit_params[label] = params
        A, b, C_fit = params
        print(f"\n  {label}:")
        print(f"    L = {A:.4e} * F^(-{b:.4f}) + {C_fit:.4f}")
    except RuntimeError:
        print(f"  Could not fit {label} to power law with offset.")

# ============================================================
# Leave-one-out analysis for irreducible loss C
# ============================================================

print("\n=== Leave-One-Out analysis for irreducible loss C ===")
for label, (pts, color) in series.items():
    flops_all = pts["flops"].values
    losses_all = pts["loss"].values
    n = len(flops_all)
    if n < 4:
        print(f"  {label}: too few points ({n}) for LOO")
        continue

    depths_all = pts["depth"].values
    loo_A, loo_b, loo_C = [], [], []
    for i in range(n):
        f_loo = np.delete(flops_all, i)
        l_loo = np.delete(losses_all, i)
        c_g = np.min(l_loo) * 0.95
        b_g = 0.1
        c_g_safe = min(c_g, l_loo[0] * 0.95)
        a_g = (l_loo[0] - c_g_safe) * (f_loo[0] ** b_g)
        if a_g <= 0:
            a_g = 1.0
        try:
            p, _ = curve_fit(
                power_law_with_offset, f_loo, l_loo,
                p0=[a_g, b_g, c_g],
                bounds=([0, 0, 0], [np.inf, np.inf, np.min(l_loo)]),
                maxfev=50000,
            )
            loo_A.append(p[0])
            loo_b.append(p[1])
            loo_C.append(p[2])
            print(f"  {label} LOO (drop d{int(depths_all[i])}): "
                  f"A={p[0]:.4e}  b={p[1]:.4f}  C={p[2]:.4f}")
        except RuntimeError:
            print(f"  {label} LOO (drop d{int(depths_all[i])}): fit failed")

    if loo_C:
        loo_C = np.array(loo_C)
        loo_b = np.array(loo_b)
        loo_A = np.array(loo_A)
        print(f"  {label} LOO summary:")
        print(f"    C: mean={loo_C.mean():.4f}  std={loo_C.std():.4f}  "
              f"range=[{loo_C.min():.4f}, {loo_C.max():.4f}]")
        print(f"    b: mean={loo_b.mean():.4f}  std={loo_b.std():.4f}  "
              f"range=[{loo_b.min():.4f}, {loo_b.max():.4f}]")
        print(f"    A: mean={loo_A.mean():.4e}  std={loo_A.std():.4e}  "
              f"range=[{loo_A.min():.4e}, {loo_A.max():.4e}]")

# ============================================================
# Plot: scaling comparison (left) + compute leverage (right)
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- Left panel: Loss vs FLOPs ---
for i, (label, (pts, color)) in enumerate(series.items()):
    flops = pts["flops"].values
    losses = pts["loss"].values
    ax1.plot(flops, losses, marker=markers[i], linestyle='none',
             label=label, color=color, markersize=10)
    if label in fit_params:
        A, b, C_fit = fit_params[label]
        plot_flops = np.geomspace(flops.min(), flops.max(), 100)
        fitted = power_law_with_offset(plot_flops, A, b, C_fit)
        ax1.plot(plot_flops, fitted, linestyle='--', color=color,
                 label=f'L = {A:.2e} ⋅ C^(-{b:.2f}) + {C_fit:.2f}',
                 linewidth=2)

ax1.set_xscale('log')
ax1.set_xlabel("Training FLOPs (C)")
ax1.set_ylabel("Validation Loss (L)")
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.7, linewidth=1.5)
# Annotate 1.58x leverage at d24 on the left panel
hp_label = f'MuonH+HyperP'
muon_label_tmp = f'Muon'
if hp_label in fit_params and muon_label_tmp in fit_params:
    A_hp, b_hp, C_hp = fit_params[hp_label]
    d24_muon_flops = muon_pts[muon_pts["depth"] == 24]["flops"].values[0]
    d24_muon_loss = muon_pts[muon_pts["depth"] == 24]["loss"].values[0]
    d24_hp_equiv_flops = ((d24_muon_loss - C_hp) / A_hp) ** (-1.0 / b_hp)
    lev = d24_muon_flops / d24_hp_equiv_flops

    ax1.annotate(
        '', xy=(d24_hp_equiv_flops, d24_muon_loss),
        xytext=(d24_muon_flops, d24_muon_loss),
        arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
    )
    mid_flops = (d24_hp_equiv_flops**0.85) * (d24_muon_flops**0.15)
    ax1.text(mid_flops, d24_muon_loss + 0.025, f'{lev:.2f}×',
             ha='center', va='top', fontsize=14, fontweight='bold',
             color='red')

moe_label_key = 'MuonH+HyperP MoE'
if moe_label_key in fit_params and hp_label in fit_params:
    A_moe, b_moe, C_moe = fit_params[moe_label_key]
    d24_hp_flops = hp_pts[hp_pts["depth"] == 24]["flops"].values[0]
    d24_hp_loss = hp_pts[hp_pts["depth"] == 24]["loss"].values[0]
    if d24_hp_loss > C_moe:
        d24_moe_equiv_flops = ((d24_hp_loss - C_moe) / A_moe) ** (-1.0 / b_moe)
        moe_lev = d24_hp_flops / d24_moe_equiv_flops

        ax1.annotate(
            '', xy=(d24_moe_equiv_flops, d24_hp_loss),
            xytext=(d24_hp_flops, d24_hp_loss),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
        )
        mid_moe = (d24_moe_equiv_flops**0.85) * (d24_hp_flops**0.15)
        ax1.text(mid_moe, d24_hp_loss + 0.025, f'{moe_lev:.2f}×',
                 ha='center', va='top', fontsize=14, fontweight='bold',
                 color='red')

# --- Right panel: Compute Efficiency Leverage ---
muon_label = f'Muon'
if muon_label in fit_params:

    def invert_power_law(loss, A, b, C):
        """Invert L = A * F^(-b) + C  =>  F = ((L - C) / A)^(-1/b)"""
        return ((loss - C) / A) ** (-1.0 / b)

    muon_flops = muon_pts["flops"].values
    muon_losses = muon_pts["loss"].values
    muon_depths = muon_pts["depth"].values

    leverage_series = {k: v for k, v in series.items() if k != muon_label}

    print("\n=== Compute Efficiency Leverage (baseline = Muon) ===")
    print("  (At each Muon FLOPs scale: what % of FLOPs does the new method need for the same loss?)")
    for i_off, (label, (pts, color)) in enumerate(leverage_series.items()):
        if label not in fit_params:
            print(f"  Skipping {label}: no fit available")
            continue
        A_new, b_new, C_new = fit_params[label]

        valid = muon_losses > C_new
        flops_v = muon_flops[valid]
        losses_v = muon_losses[valid]
        depths_v = muon_depths[valid]

        new_equiv_flops = invert_power_law(losses_v, A_new, b_new, C_new)
        pct_flops = new_equiv_flops / flops_v
        leverage = 1.0 / pct_flops

        marker_idx = list(series.keys()).index(label)
        ax2.plot(flops_v, leverage, marker=markers[marker_idx], linestyle='-',
                 label=label, color=color, markersize=10, linewidth=2)

        print(f"\n  {label}:")
        for d_val, f_val, l_val, eq_f, pct, lev in zip(
                depths_v, flops_v, losses_v, new_equiv_flops, pct_flops, leverage):
            print(f"    d{int(d_val):2d}  Muon FLOPs={f_val:.2e}  Muon loss={l_val:.4f}"
                  f"  new method FLOPs={eq_f:.2e}  pct={pct:.2%}  leverage={lev:.2f}x")

    ax2.set_xscale('log')
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=2, label='Muon baseline')
    ax2.set_xlabel("Training FLOPs")
    ax2.set_ylabel("Compute Efficiency Leverage ")
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', alpha=0.7, linewidth=1.5)
fig.tight_layout()
fig.savefig("scaling_comparison.png", dpi=300, bbox_inches='tight')
print("\nSaved to scaling_comparison.png")
