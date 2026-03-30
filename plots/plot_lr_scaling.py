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

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax1 = axes[0]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(token_budgets)))
optimal_lrs = []

for tok, color in zip(token_budgets, colors):
    subset = data[data["tokens"] == tok].sort_values("lr")
    lrs = subset["lr"].values
    losses = subset["val_loss"].values

    ax1.scatter(lrs, losses, color=color, s=50, zorder=5,
                label=f"{tok/1e9:.1f}B tokens")

    log_lrs = np.log(lrs)
    coeffs = np.polyfit(log_lrs, losses, 2)
    poly = np.poly1d(coeffs)

    log_lr_fine = np.linspace(log_lrs.min() - 0.3, log_lrs.max() + 0.3, 200)
    ax1.plot(np.exp(log_lr_fine), poly(log_lr_fine), color=color, linewidth=1.5, alpha=0.7)

    opt_log_lr = -coeffs[1] / (2 * coeffs[0])
    opt_lr = np.exp(opt_log_lr)
    opt_loss = poly(opt_log_lr)
    optimal_lrs.append((tok, opt_lr, opt_loss))

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

opt_df = pd.DataFrame(optimal_lrs, columns=["tokens", "opt_lr", "opt_loss"])

def power_law(T, a, b):
    return a * T ** b

def power_law_irr(T, a, b, c):
    return a * T ** b + c

popt, pcov = curve_fit(power_law, opt_df["tokens"].values, opt_df["opt_lr"].values, p0=[1.0, -0.1])
a_fit, b_fit = popt

popt_irr, pcov_irr = curve_fit(
    power_law_irr, opt_df["tokens"].values, opt_df["opt_lr"].values,
    p0=[1.0, -0.1, 0.001], maxfev=10000
)
a_irr, b_irr, c_irr = popt_irr

ax2 = axes[1]
ax2.scatter(opt_df["tokens"], opt_df["opt_lr"], s=100, color="royalblue",
            edgecolors="black", zorder=5, label="Optimal LR (from quadratic fit)")

tok_fine = np.geomspace(opt_df["tokens"].min() * 0.5, opt_df["tokens"].max() * 50, 200)
ax2.plot(tok_fine, power_law(tok_fine, *popt), "r--", linewidth=1.5, alpha=0.6,
         label=f"Power law: {a_fit:.2f} * T^({b_fit:.4f})")
# ax2.plot(tok_fine, power_law_irr(tok_fine, *popt_irr), "darkgreen", linewidth=2,
#          label=f"w/ irr: {a_irr:.2f} * T^({b_irr:.4f}) + {c_irr:.6f}")
# ax2.axhline(c_irr, color="gray", linestyle=":", alpha=0.6,
#             label=f"Irreducible LR = {c_irr:.6f}")

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Training Tokens")
ax2.set_ylabel("Optimal Learning Rate")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

# # --- Panel 3: Leave-one-out error bar chart ---
# all_tokens = opt_df["tokens"].values
# all_opt_lrs_arr = opt_df["opt_lr"].values
# n = len(all_tokens)
# loo_rows = []

# for i in range(n):
#     mask = np.arange(n) != i
#     train_T = all_tokens[mask]
#     train_lr = all_opt_lrs_arr[mask]
#     test_T = all_tokens[i]
#     test_lr = all_opt_lrs_arr[i]

#     try:
#         p1, _ = curve_fit(power_law, train_T, train_lr, p0=[1.0, -0.1])
#         pred_pl = power_law(test_T, *p1)
#     except RuntimeError:
#         pred_pl = np.nan
#     try:
#         p2, _ = curve_fit(power_law_irr, train_T, train_lr, p0=[1.0, -0.1, 0.001], maxfev=10000)
#         pred_irr = power_law_irr(test_T, *p2)
#     except RuntimeError:
#         pred_irr = np.nan

#     err_pl = (pred_pl - test_lr) / test_lr * 100
#     err_irr = (pred_irr - test_lr) / test_lr * 100
#     loo_rows.append((test_T, test_lr, pred_pl, err_pl, pred_irr, err_irr))

# ax3 = axes[2]
# x_pos = np.arange(n)
# tok_labels = [f"{t/1e9:.1f}B" for t in all_tokens]

# ax3.bar(x_pos - 0.18, [r[3] for r in loo_rows], 0.35, color="salmon", edgecolor="black",
#         label="Power Law err%")
# ax3.bar(x_pos + 0.18, [r[5] for r in loo_rows], 0.35, color="lightgreen", edgecolor="black",
#         label="PL + irr err%")
# ax3.axhline(0, color="black", linewidth=0.5)
# ax3.set_xticks(x_pos)
# ax3.set_xticklabels(tok_labels)
# ax3.set_xlabel("Held-out Token Budget", fontsize=13)
# ax3.set_ylabel("Prediction Error (%)", fontsize=13)
# ax3.set_title("Leave-One-Out Extrapolation\nError Comparison", fontsize=13)
# ax3.legend(fontsize=10)
# ax3.grid(True, alpha=0.3, axis="y")

# pl_mean = np.mean([abs(r[3]) for r in loo_rows])
# irr_mean = np.mean([abs(r[5]) for r in loo_rows])
# ax3.text(0.98, 0.02, f"Mean |err|: PL={pl_mean:.2f}%, PL+irr={irr_mean:.2f}%",
#          transform=ax3.transAxes, fontsize=9, ha="right", va="bottom",
#          bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

plt.tight_layout()
plt.savefig("lr_scaling_analysis.png", dpi=150, bbox_inches="tight")
print("Plot saved to lr_scaling_analysis.png")

print("\n=== Optimal Learning Rates ===")
for tok, lr, loss in optimal_lrs:
    print(f"  Tokens: {tok/1e9:>6.1f}B  |  Optimal LR: {lr:.6f}  |  Min Loss: {loss:.4f}")

print(f"\n=== Power Law Fit (no irreducible) ===")
print(f"  optimal_lr = {a_fit:.6f} * T^({b_fit:.6f})")
print(f"  500B tokens: {power_law(500e9, *popt):.6f}")
print(f"  1T tokens:   {power_law(1e12, *popt):.6f}")

print(f"\n=== Power Law Fit (with irreducible LR) ===")
print(f"  optimal_lr = {a_irr:.6f} * T^({b_irr:.6f}) + {c_irr:.6f}")
print(f"  Irreducible LR (c): {c_irr:.6f}")
print(f"  500B tokens: {power_law_irr(500e9, *popt_irr):.6f}")
print(f"  1T tokens:   {power_law_irr(1e12, *popt_irr):.6f}")
print(f"  T -> inf:    {c_irr:.6f}")

# --- Leave-one-out extrapolation comparison ---
all_tokens = opt_df["tokens"].values
all_opt_lrs = opt_df["opt_lr"].values
n = len(all_tokens)

print("\n" + "=" * 70)
print("=== Leave-One-Out Extrapolation Comparison ===")
print("=" * 70)
print(f"{'Held-out':>12s} | {'True LR':>10s} | {'PL pred':>10s} | {'PL err%':>8s} | {'PL+irr pred':>12s} | {'PL+irr err%':>11s}")
print("-" * 70)

fig2, axes2 = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)

for i in range(n):
    mask = np.arange(n) != i
    train_T = all_tokens[mask]
    train_lr = all_opt_lrs[mask]
    test_T = all_tokens[i]
    test_lr = all_opt_lrs[i]

    try:
        p1, _ = curve_fit(power_law, train_T, train_lr, p0=[1.0, -0.1])
        pred_pl = power_law(test_T, *p1)
    except RuntimeError:
        pred_pl = np.nan

    try:
        p2, _ = curve_fit(power_law_irr, train_T, train_lr,
                          p0=[1.0, -0.1, 0.001], maxfev=10000)
        pred_irr = power_law_irr(test_T, *p2)
    except RuntimeError:
        pred_irr = np.nan

    err_pl = (pred_pl - test_lr) / test_lr * 100
    err_irr = (pred_irr - test_lr) / test_lr * 100

    print(f"  {test_T/1e9:>6.1f}B    | {test_lr:>10.6f} | {pred_pl:>10.6f} | {err_pl:>+7.2f}% | {pred_irr:>12.6f} | {err_irr:>+10.2f}%")

    ax = axes2[i]
    t_fine = np.geomspace(all_tokens.min() * 0.5, all_tokens.max() * 3, 200)
    ax.scatter(train_T, train_lr, s=60, color="royalblue", edgecolors="black", zorder=5, label="Train")
    ax.scatter([test_T], [test_lr], s=120, color="gold", edgecolors="black", zorder=6, marker="*", label=f"True ({test_lr:.5f})")
    ax.plot(t_fine, power_law(t_fine, *p1), "r--", lw=1.5, alpha=0.7, label=f"PL ({pred_pl:.5f})")
    if not np.isnan(pred_irr):
        ax.plot(t_fine, power_law_irr(t_fine, *p2), "g-", lw=1.5, alpha=0.7, label=f"PL+irr ({pred_irr:.5f})")
    ax.scatter([test_T], [pred_pl], s=80, color="red", marker="x", zorder=6)
    if not np.isnan(pred_irr):
        ax.scatter([test_T], [pred_irr], s=80, color="green", marker="x", zorder=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tokens")
    if i == 0:
        ax.set_ylabel("Optimal LR")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

# Summary
pl_errors = []
irr_errors = []
for i in range(n):
    mask = np.arange(n) != i
    try:
        p1, _ = curve_fit(power_law, all_tokens[mask], all_opt_lrs[mask], p0=[1.0, -0.1])
        pl_errors.append(abs((power_law(all_tokens[i], *p1) - all_opt_lrs[i]) / all_opt_lrs[i] * 100))
    except RuntimeError:
        pass
    try:
        p2, _ = curve_fit(power_law_irr, all_tokens[mask], all_opt_lrs[mask], p0=[1.0, -0.1, 0.001], maxfev=10000)
        irr_errors.append(abs((power_law_irr(all_tokens[i], *p2) - all_opt_lrs[i]) / all_opt_lrs[i] * 100))
    except RuntimeError:
        pass

print("-" * 70)
print(f"  Mean |err%|:  Power Law = {np.mean(pl_errors):.2f}%    Power Law + irr = {np.mean(irr_errors):.2f}%")

plt.tight_layout()
plt.savefig("lr_scaling_extrapolation.png", dpi=150, bbox_inches="tight")
print("\nExtrapolation plot saved to lr_scaling_extrapolation.png")

# --- Sensitivity of irreducible LR to number of fitting points ---
print("\n" + "=" * 70)
print("=== Sensitivity of Irreducible LR (c) to # of Fitting Points ===")
print("=" * 70)

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5))

fit_results = []
for k in range(3, n + 1):
    T_k = all_tokens[:k]
    lr_k = all_opt_lrs[:k]

    p_pl, _ = curve_fit(power_law, T_k, lr_k, p0=[1.0, -0.1])
    try:
        p_irr, pcov_k = curve_fit(power_law_irr, T_k, lr_k,
                                   p0=[1.0, -0.1, 0.001], maxfev=10000)
        c_stderr = np.sqrt(np.diag(pcov_k))[2] if np.all(np.isfinite(pcov_k)) else np.nan
    except RuntimeError:
        p_irr = [np.nan, np.nan, np.nan]
        c_stderr = np.nan

    max_tok_label = f"{T_k[-1]/1e9:.1f}B"
    fit_results.append({
        "k": k, "max_tok": T_k[-1], "max_tok_label": max_tok_label,
        "a_pl": p_pl[0], "b_pl": p_pl[1],
        "a_irr": p_irr[0], "b_irr": p_irr[1], "c_irr": p_irr[2],
        "c_stderr": c_stderr,
        "p_pl": p_pl, "p_irr": p_irr,
    })

    print(f"  First {k} points (up to {max_tok_label}):")
    print(f"    PL:     a={p_pl[0]:.4f}, b={p_pl[1]:.6f}")
    print(f"    PL+irr: a={p_irr[0]:.4f}, b={p_irr[1]:.6f}, c={p_irr[2]:.6f} +/- {c_stderr:.6f}")

# Left panel: fitted curves overlaid
ax_l = axes3[0]
ax_l.scatter(all_tokens, all_opt_lrs, s=100, color="black", edgecolors="black",
             zorder=10, label="All 5 optimal LRs")

tok_ext = np.geomspace(all_tokens.min() * 0.3, all_tokens.max() * 100, 300)
colors_k = ["#e74c3c", "#3498db", "#2ecc71"]

for idx, res in enumerate(fit_results):
    k = res["k"]
    col = colors_k[idx]
    ax_l.plot(tok_ext, power_law_irr(tok_ext, *res["p_irr"]), color=col, linewidth=2,
              label=f"First {k} pts (c={res['c_irr']:.4f})")
    ax_l.axhline(res["c_irr"], color=col, linestyle=":", alpha=0.4)

ax_l.set_xscale("log")
ax_l.set_yscale("log")
ax_l.set_xlabel("Training Tokens")
ax_l.set_ylabel("Optimal Learning Rate")
ax_l.legend()
ax_l.grid(True, alpha=0.3)
ax_l.set_xlim(tok_ext.min(), tok_ext.max())

# Right panel: c value with error bars
ax_r = axes3[1]
ks = [r["k"] for r in fit_results]
cs = [r["c_irr"] for r in fit_results]
c_errs = [r["c_stderr"] for r in fit_results]
x_labels = [f"First {r['k']}\n(to {r['max_tok_label']})" for r in fit_results]

bars = ax_r.bar(range(len(ks)), cs, yerr=c_errs, capsize=8, color=colors_k,
                edgecolor="black", alpha=0.85)
for i, (bar, c_val, c_err) in enumerate(zip(bars, cs, c_errs)):
    err_str = f" +/- {c_err:.4f}" if np.isfinite(c_err) else ""
    ax_r.text(bar.get_x() + bar.get_width()/2, bar.get_height() + c_err + 0.0002,
              f"{c_val:.4f}{err_str}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax_r.set_xticks(range(len(ks)))
ax_r.set_xticklabels(x_labels)
ax_r.set_ylabel("Irreducible LR (c)")
ax_r.grid(True, alpha=0.3, axis="y")
ax_r.set_ylim(0, max(c + e for c, e in zip(cs, [e if np.isfinite(e) else 0 for e in c_errs])) * 1.5)

plt.tight_layout()
plt.savefig("lr_irr_sensitivity.png", dpi=150, bbox_inches="tight")
print("\nSensitivity plot saved to lr_irr_sensitivity.png")

# --- LOO sensitivity of irreducible LR ---
print("\n" + "=" * 70)
print("=== LOO Sensitivity of Irreducible LR (c) ===")
print("=" * 70)

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5.5))

loo_fit_results = []
for i in range(n):
    mask = np.arange(n) != i
    T_train = all_tokens[mask]
    lr_train = all_opt_lrs[mask]
    held_tok = all_tokens[i]

    p_pl, _ = curve_fit(power_law, T_train, lr_train, p0=[1.0, -0.1])
    try:
        p_irr, pcov_loo = curve_fit(power_law_irr, T_train, lr_train,
                                     p0=[1.0, -0.1, 0.001], maxfev=10000)
        c_stderr = np.sqrt(np.diag(pcov_loo))[2] if np.all(np.isfinite(pcov_loo)) else np.nan
    except RuntimeError:
        p_irr = [np.nan, np.nan, np.nan]
        c_stderr = np.nan

    label = f"{held_tok/1e9:.1f}B"
    loo_fit_results.append({
        "held_out": held_tok, "label": label,
        "a_irr": p_irr[0], "b_irr": p_irr[1], "c_irr": p_irr[2],
        "c_stderr": c_stderr, "p_irr": p_irr, "p_pl": p_pl,
    })
    print(f"  Hold out {label:>6s}:  c = {p_irr[2]:>+10.6f}  +/- {c_stderr:.6f}   (b = {p_irr[1]:.4f})")

# Left panel: curves from each LOO fit
ax_l = axes4[0]
ax_l.scatter(all_tokens, all_opt_lrs, s=100, color="black", edgecolors="black",
             zorder=10, label="All 5 optimal LRs")

tok_ext = np.geomspace(all_tokens.min() * 0.3, all_tokens.max() * 100, 300)
loo_colors = plt.cm.Set1(np.linspace(0, 0.6, n))

for res, col in zip(loo_fit_results, loo_colors):
    if np.isnan(res["c_irr"]):
        continue
    ax_l.plot(tok_ext, power_law_irr(tok_ext, *res["p_irr"]), color=col, linewidth=1.5, alpha=0.8,
              label=f"Drop {res['label']} (c={res['c_irr']:.4f})")
    ax_l.axhline(res["c_irr"], color=col, linestyle=":", alpha=0.3)

ax_l.plot(tok_ext, power_law_irr(tok_ext, *popt_irr), "k-", linewidth=2.5, alpha=0.5,
          label=f"All 5 pts (c={c_irr:.4f})")
ax_l.axhline(c_irr, color="black", linestyle=":", alpha=0.5)

ax_l.set_xscale("log")
ax_l.set_yscale("log")
ax_l.set_xlabel("Training Tokens")
ax_l.set_ylabel("Optimal Learning Rate")
ax_l.legend(loc="upper right")
ax_l.grid(True, alpha=0.3)

# Right panel: bar chart of c values
ax_r = axes4[1]
x_labels = [f"Drop\n{r['label']}" for r in loo_fit_results]
cs_loo = [r["c_irr"] for r in loo_fit_results]
errs_loo = [r["c_stderr"] if np.isfinite(r["c_stderr"]) else 0 for r in loo_fit_results]

bars = ax_r.bar(range(n), cs_loo, yerr=errs_loo, capsize=8, color=loo_colors,
                edgecolor="black", alpha=0.85)
ax_r.axhline(c_irr, color="black", linewidth=2, linestyle="--", alpha=0.6,
             label=f"All 5 pts: c={c_irr:.4f}")

for i_b, (bar, c_val, c_err) in enumerate(zip(bars, cs_loo, errs_loo)):
    y_pos = max(c_val + c_err, c_val) + 0.0003
    ax_r.text(bar.get_x() + bar.get_width()/2, y_pos,
              f"{c_val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax_r.set_xticks(range(n))
ax_r.set_xticklabels(x_labels)
ax_r.set_ylabel("Irreducible LR (c)")
ax_r.legend()
ax_r.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("lr_irr_loo_sensitivity.png", dpi=150, bbox_inches="tight")
print("\nLOO sensitivity plot saved to lr_irr_loo_sensitivity.png")
