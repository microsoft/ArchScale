import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from itertools import combinations

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
})

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

def fit_optimal_loss(lrs, losses):
    log_lrs = np.log(lrs)
    coeffs = np.polyfit(log_lrs, losses, 2)
    if coeffs[0] <= 0:
        return np.nan
    opt_log_lr = -coeffs[1] / (2 * coeffs[0])
    return np.polyval(coeffs, opt_log_lr)

fig, (ax_lr, ax_loss) = plt.subplots(1, 2, figsize=(14, 5.5))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(token_budgets)))

for tok, color in zip(token_budgets, colors):
    subset = data[data["tokens"] == tok].sort_values("lr")
    all_lrs = subset["lr"].values
    all_losses = subset["val_loss"].values
    n = len(all_lrs)
    full_opt_lr = fit_optimal_lr(all_lrs, all_losses)
    full_opt_loss = fit_optimal_loss(all_lrs, all_losses)

    ks = list(range(3, n))
    rel_errs_lr, rel_errs_loss = [], []

    for k in ks:
        opt_lrs_k, opt_losses_k = [], []
        for combo in combinations(range(n), k):
            idx = list(combo)
            opt_lr = fit_optimal_lr(all_lrs[idx], all_losses[idx])
            opt_loss = fit_optimal_loss(all_lrs[idx], all_losses[idx])
            if not np.isnan(opt_lr):
                opt_lrs_k.append(opt_lr)
            if not np.isnan(opt_loss):
                opt_losses_k.append(opt_loss)

        if opt_lrs_k:
            arr = np.array(opt_lrs_k)
            rel_errs_lr.append(np.mean(np.abs(arr - full_opt_lr) / full_opt_lr) * 100)
        else:
            rel_errs_lr.append(np.nan)

        if opt_losses_k:
            arr = np.array(opt_losses_k)
            rel_errs_loss.append(np.mean(np.abs(arr - full_opt_loss) / full_opt_loss) * 100)
        else:
            rel_errs_loss.append(np.nan)

    label = f"{tok/1e9:.1f}B"
    ax_lr.plot(ks, rel_errs_lr, "o-", color=color, markersize=10, linewidth=2, label=label)
    ax_loss.plot(ks, rel_errs_loss, "o-", color=color, markersize=10, linewidth=2, label=label)

for ax, title in [(ax_lr, "Optimal LR"), (ax_loss, "Optimal Loss")]:
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Mean Relative Error (%)")
    ax.legend(title="Tokens",title_fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig("npoints_relative_error.png", dpi=150, bbox_inches="tight")
print("Saved npoints_relative_error.png")
