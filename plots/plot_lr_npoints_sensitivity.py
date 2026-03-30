import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from itertools import combinations

df = pd.read_csv("muonh_data_scaling.csv")

def parse_name(name):
    lr_match = re.search(r'_lr([\d.e\-]+)x', name)
    tok_match = re.search(r'_tok([\d.e]+)', name)
    if lr_match and tok_match:
        lr = float(lr_match.group(1))
        tok = float(tok_match.group(1))
        return lr, tok
    return None, None

records = []
for _, row in df.iterrows():
    lr, tokens = parse_name(row["Name"])
    if lr is not None:
        records.append({
            "lr": lr,
            "tokens": tokens,
            "val_loss": row["metric/val_loss@1x"],
        })

data = pd.DataFrame(records)
token_budgets = sorted(data["tokens"].unique())

def fit_optimal_lr(lrs, losses):
    log_lrs = np.log(lrs)
    coeffs = np.polyfit(log_lrs, losses, 2)
    if coeffs[0] <= 0:
        return np.nan
    opt_log_lr = -coeffs[1] / (2 * coeffs[0])
    return np.exp(opt_log_lr)

def fit_quadratic_coeffs(lrs, losses):
    log_lrs = np.log(lrs)
    return np.polyfit(log_lrs, losses, 2)

def eval_quadratic(coeffs, lr):
    return np.polyval(coeffs, np.log(lr))

# --- Figure 1: Optimal LR distribution vs number of fitting points ---
fig1, axes1 = plt.subplots(1, len(token_budgets), figsize=(5 * len(token_budgets), 5), sharey=False)
if len(token_budgets) == 1:
    axes1 = [axes1]

for ax, tok in zip(axes1, token_budgets):
    subset = data[data["tokens"] == tok].sort_values("lr")
    all_lrs = subset["lr"].values
    all_losses = subset["val_loss"].values
    n = len(all_lrs)

    full_opt = fit_optimal_lr(all_lrs, all_losses)

    ks = range(3, n + 1)
    medians, means, stds = [], [], []
    q25s, q75s, mins, maxs = [], [], [], []

    for k in ks:
        opt_lrs_k = []
        for combo in combinations(range(n), k):
            idx = list(combo)
            opt = fit_optimal_lr(all_lrs[idx], all_losses[idx])
            if not np.isnan(opt):
                opt_lrs_k.append(opt)

        if opt_lrs_k:
            arr = np.array(opt_lrs_k)
            medians.append(np.median(arr))
            means.append(np.mean(arr))
            stds.append(np.std(arr))
            q25s.append(np.percentile(arr, 25))
            q75s.append(np.percentile(arr, 75))
            mins.append(np.min(arr))
            maxs.append(np.max(arr))
        else:
            medians.append(np.nan)
            means.append(np.nan)
            stds.append(np.nan)
            q25s.append(np.nan)
            q75s.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)

    ks_arr = np.array(list(ks))
    medians = np.array(medians)
    means = np.array(means)
    q25s = np.array(q25s)
    q75s = np.array(q75s)
    mins = np.array(mins)
    maxs = np.array(maxs)

    ax.fill_between(ks_arr, mins, maxs, alpha=0.15, color="royalblue", label="min-max")
    ax.fill_between(ks_arr, q25s, q75s, alpha=0.35, color="royalblue", label="25-75%")
    ax.plot(ks_arr, medians, "o-", color="royalblue", markersize=5, linewidth=1.5, label="Median")
    ax.axhline(full_opt, color="red", linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"All {n} pts: {full_opt:.5f}")

    ax.set_xlabel("# LR data points used", fontsize=12)
    ax.set_ylabel("Fitted optimal LR", fontsize=12)
    ax.set_xticks(ks_arr)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lr_npoints_sensitivity.png", dpi=150, bbox_inches="tight")
print("Saved lr_npoints_sensitivity.png")

# --- Figure 2: Relative error ---
fig2, ax_re = plt.subplots(figsize=(7, 5.5))

colors = plt.cm.viridis(np.linspace(0, 0.9, len(token_budgets)))

for tok, color in zip(token_budgets, colors):
    subset = data[data["tokens"] == tok].sort_values("lr")
    all_lrs = subset["lr"].values
    all_losses = subset["val_loss"].values
    n = len(all_lrs)
    full_opt = fit_optimal_lr(all_lrs, all_losses)

    ks = list(range(3, n))
    rel_errs = []

    for k in ks:
        opt_lrs_k = []
        for combo in combinations(range(n), k):
            idx = list(combo)
            opt = fit_optimal_lr(all_lrs[idx], all_losses[idx])
            if not np.isnan(opt):
                opt_lrs_k.append(opt)

        if opt_lrs_k:
            arr = np.array(opt_lrs_k)
            rel_errs.append(np.mean(np.abs(arr - full_opt) / full_opt) * 100)
        else:
            rel_errs.append(np.nan)

    label = f"{tok/1e9:.1f}B"
    ax_re.plot(ks, rel_errs, "o-", color=color, markersize=6, linewidth=1.5, label=label)

ax_re.set_xlabel("Data Points", fontsize=14)
ax_re.set_ylabel("Mean Relative Error (%)", fontsize=14)
ax_re.legend(fontsize=12, title="Token Budget", title_fontsize=12)
ax_re.grid(True, alpha=0.3)
ax_re.set_yscale("log")
ax_re.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig("lr_npoints_cv_error.png", dpi=150, bbox_inches="tight")
print("Saved lr_npoints_cv_error.png")

# --- Figure 3: Impact on downstream power-law fit (LR vs tokens) ---
from scipy.optimize import curve_fit

def power_law(T, a, b):
    return a * T ** b

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5))

n_bootstrap = 200
rng = np.random.default_rng(42)
all_n_per_tok = {}
for tok in token_budgets:
    subset = data[data["tokens"] == tok].sort_values("lr")
    all_n_per_tok[tok] = len(subset)

min_k, max_k = 3, min(all_n_per_tok.values())
k_values = list(range(min_k, max_k + 1))

ax_l = axes3[0]
ax_r = axes3[1]

full_opt_lrs = []
for tok in token_budgets:
    subset = data[data["tokens"] == tok].sort_values("lr")
    full_opt_lrs.append(fit_optimal_lr(subset["lr"].values, subset["val_loss"].values))

try:
    popt_full, _ = curve_fit(power_law, np.array(token_budgets), np.array(full_opt_lrs), p0=[1.0, -0.1])
except RuntimeError:
    popt_full = [np.nan, np.nan]

pl_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(k_values)))

fitted_as, fitted_bs = {k: [] for k in k_values}, {k: [] for k in k_values}

for k, col in zip(k_values, pl_colors):
    for _ in range(n_bootstrap):
        sampled_opt_lrs = []
        for tok in token_budgets:
            subset = data[data["tokens"] == tok].sort_values("lr")
            lrs_arr = subset["lr"].values
            losses_arr = subset["val_loss"].values
            n_tok = len(lrs_arr)
            idx = sorted(rng.choice(n_tok, size=k, replace=False))
            opt = fit_optimal_lr(lrs_arr[idx], losses_arr[idx])
            sampled_opt_lrs.append(opt)

        if any(np.isnan(v) for v in sampled_opt_lrs):
            continue

        try:
            popt_k, _ = curve_fit(power_law, np.array(token_budgets),
                                  np.array(sampled_opt_lrs), p0=[1.0, -0.1])
            fitted_as[k].append(popt_k[0])
            fitted_bs[k].append(popt_k[1])
        except RuntimeError:
            pass

    if fitted_bs[k]:
        tok_fine = np.geomspace(min(token_budgets) * 0.5, max(token_budgets) * 50, 200)
        median_a = np.median(fitted_as[k])
        median_b = np.median(fitted_bs[k])
        ax_l.plot(tok_fine, power_law(tok_fine, median_a, median_b),
                  color=col, linewidth=1.2, alpha=0.7, label=f"k={k}")

ax_l.scatter(token_budgets, full_opt_lrs, s=100, color="black", edgecolors="black",
             zorder=10, label="Full-data optimal LRs")
if not np.isnan(popt_full[0]):
    tok_fine = np.geomspace(min(token_budgets) * 0.5, max(token_budgets) * 50, 200)
    ax_l.plot(tok_fine, power_law(tok_fine, *popt_full), "k--", linewidth=2.5,
              alpha=0.6, label=f"Full fit: b={popt_full[1]:.4f}")

ax_l.set_xscale("log")
ax_l.set_yscale("log")
ax_l.set_xlabel("Training Tokens", fontsize=14)
ax_l.set_ylabel("Optimal LR", fontsize=14)
ax_l.legend(fontsize=12, loc="upper right")
ax_l.grid(True, alpha=0.3)

# Right panel: box plot of exponent b
b_data = [fitted_bs[k] for k in k_values if fitted_bs[k]]
b_labels = [f"{k}" for k in k_values if fitted_bs[k]]
bp = ax_r.boxplot(b_data, tick_labels=b_labels, patch_artist=True, widths=0.6)
for patch, col in zip(bp["boxes"], pl_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)

if not np.isnan(popt_full[1]):
    ax_r.axhline(popt_full[1], color="black", linestyle="--", linewidth=2,
                 label=f"Full-data b={popt_full[1]:.4f}")

ax_r.set_xlabel("# LR points per token budget (k)", fontsize=13)
ax_r.set_ylabel("Power law exponent (b)", fontsize=13)
ax_r.legend(fontsize=10)
ax_r.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("lr_npoints_powerlaw_impact.png", dpi=150, bbox_inches="tight")
print("Saved lr_npoints_powerlaw_impact.png")

# --- Print summary ---
print("\n" + "=" * 70)
print("=== Summary: Optimal LR Sensitivity to # of LR Sweep Points ===")
print("=" * 70)

for tok in token_budgets:
    subset = data[data["tokens"] == tok].sort_values("lr")
    all_lrs = subset["lr"].values
    all_losses = subset["val_loss"].values
    n = len(all_lrs)
    full_opt = fit_optimal_lr(all_lrs, all_losses)

    full_coeffs = fit_quadratic_coeffs(all_lrs, all_losses)
    full_opt_loss = eval_quadratic(full_coeffs, full_opt)

    print(f"\n  Token budget: {tok/1e9:.1f}B  ({n} LRs)  |  Full-data optimal LR: {full_opt:.6f}  |  Full-data optimal loss: {full_opt_loss:.6f}")
    print(f"  {'k':>4s}  {'#combos':>8s}  {'median':>10s}  {'mean':>10s}  {'std':>10s}  {'CV%':>8s}  {'relErr%':>8s}  {'lossRelErr%':>12s}")

    for k in range(3, n + 1):
        opt_lrs_k = []
        for combo in combinations(range(n), k):
            idx = list(combo)
            opt = fit_optimal_lr(all_lrs[idx], all_losses[idx])
            if not np.isnan(opt):
                opt_lrs_k.append(opt)
        if opt_lrs_k:
            arr = np.array(opt_lrs_k)
            cv = np.std(arr) / np.mean(arr) * 100
            rel_err = np.mean(np.abs(arr - full_opt) / full_opt) * 100
            loss_at_sub_opt = np.array([eval_quadratic(full_coeffs, lr) for lr in arr])
            loss_rel_err = np.mean(np.abs(loss_at_sub_opt - full_opt_loss) / full_opt_loss) * 100
            print(f"  {k:>4d}  {len(opt_lrs_k):>8d}  {np.median(arr):>10.6f}  {np.mean(arr):>10.6f}  "
                  f"{np.std(arr):>10.6f}  {cv:>7.2f}%  {rel_err:>7.2f}%  {loss_rel_err:>11.4f}%")

print(f"\n  Full-data power law exponent: b = {popt_full[1]:.6f}")
for k in k_values:
    if fitted_bs[k]:
        arr = np.array(fitted_bs[k])
        print(f"  k={k}: b median={np.median(arr):.6f}, std={np.std(arr):.6f}, "
              f"IQR=[{np.percentile(arr,25):.6f}, {np.percentile(arr,75):.6f}]")
