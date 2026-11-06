#!/usr/bin/env python3
"""
Baltic Ecosystem Time Series Summary
Figure 1: Asynchronous Structural Regime Shifts
Panel A: Nutrients (5 breaks)
Panel B: MSFD Biogeochemistry Standardized ALL REGION (2 breaks)
Panel C: Coupled System Drivers + Response (3 breaks)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import log_ndtr
from scipy.optimize import brentq

# --- 1. GLOBAL SETTINGS ---
S_DATE = 1954
E_DATE = 2022
YEARS_VEC = np.arange(S_DATE, E_DATE + 1)

# Okabe-Ito colorblind-friendly palette
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

def r_scale(df_or_arr):
    """Replicates R's scale() with sample standard deviation (ddof=1)."""
    return (df_or_arr - np.mean(df_or_arr, axis=0)) / np.std(df_or_arr, axis=0, ddof=1)

def _pargmaxV(x, phi1=1.0, phi2=1.0, xi=1.0):
    """Exact CDF of R's strucchange::pargmaxV (Bai 1997 heterogeneous two-sided Brownian motion)."""
    if x == 0:
        return 0.5
    if x < 0:
        return 1.0 - _pargmaxV(-x, phi2, phi1, 1.0 / xi)
    phi = xi * (phi2 / phi1)
    s = np.sqrt(x)
    term1 = np.sqrt(x / (2.0 * np.pi)) * np.exp(-x / 8.0)
    log_term2 = np.log((phi + 0.5) / phi) + (phi * (phi + 1.0) * x / 2.0) + log_ndtr(-(phi + 0.5) * s)
    term2 = np.exp(log_term2)
    term3 = ((2.0 * phi + 0.5) / phi + 0.5 * x) * norm.cdf(-0.5 * s)
    return 1.0 + term1 + term2 - term3

def get_bp_ci_mapped(ts_matrix, years_vector, h=0.15, max_breaks=5, level=0.95):
    """
    Exact replica of R's strucchange::breakpoints(ts_data ~ 1) and confint(bp, level = 0.95)
    with default heterogeneous segment error variances (het.err = TRUE).
    """
    arr = np.asarray(ts_matrix, dtype=float)
    y = arr[:, 0] if arr.ndim > 1 else arr
    n = len(y)
    h_size = int(np.floor(h * n))

    # Precompute segment RSS
    rss = np.full((n + 1, n + 1), np.inf)
    for i in range(n):
        for j in range(i + h_size, n + 1):
            seg = y[i:j]
            rss[i, j] = np.sum((seg - np.mean(seg)) ** 2)

    # Dynamic programming for m = 0..max_breaks
    cost = np.full((max_breaks + 1, n + 1), np.inf)
    pos = np.zeros((max_breaks + 1, n + 1), dtype=int)
    cost[0, :] = rss[0, :]

    for m in range(1, max_breaks + 1):
        for j in range((m + 1) * h_size, n + 1):
            for i in range(m * h_size, j - h_size + 1):
                c = cost[m - 1, i] + rss[i, j]
                if c < cost[m, j]:
                    cost[m, j] = c
                    pos[m, j] = i

    # Select optimal m via BIC (matching strucchange::BIC.breakpointsfull)
    bics = {}
    breaks_dict = {}
    for m in range(0, max_breaks + 1):
        total_rss = cost[m, n]
        bic = n * (np.log(total_rss / n) + 1.0 + np.log(2.0 * np.pi)) + np.log(n) * 2.0 * (m + 1)
        bics[m] = bic
        b = []
        curr = n
        for step in range(m, 0, -1):
            prev = pos[step, curr]
            b.append(prev)
            curr = prev
        breaks_dict[m] = sorted(b)

    best_m = min(bics, key=bics.get)
    opt_breaks = breaks_dict[best_m]  # 1-based indices

    # Calculate 95% confidence intervals (Bai 1997 with het.err = TRUE)
    a2 = 1.0 - (1.0 - level) / 2.0
    bounds = [0] + opt_breaks + [n]
    means = [np.mean(y[bounds[i]:bounds[i + 1]]) for i in range(best_m + 1)]
    omegas = [
        np.sum((y[bounds[i]:bounds[i + 1]] - means[i]) ** 2) / len(y[bounds[i]:bounds[i + 1]])
        for i in range(best_m + 1)
    ]

    rows = []
    for i, b_1based in enumerate(opt_breaks):
        delta = means[i + 1] - means[i]
        scale1 = omegas[i] / (delta ** 2)
        scale2 = omegas[i + 1] / (delta ** 2)
        q_pos = brentq(lambda x: _pargmaxV(x, omegas[i], omegas[i + 1], 1.0) - a2, 0.0, 1000.0)
        q_neg = brentq(lambda x: _pargmaxV(x, omegas[i + 1], omegas[i], 1.0) - a2, 0.0, 1000.0)

        low_idx = max(0, b_1based - int(np.ceil(q_pos * scale2)) - 1)
        up_idx = min(n - 1, b_1based + int(np.ceil(q_neg * scale1)) - 1)
        rows.append({
            "Lower": int(years_vector[low_idx]),
            "Break": int(years_vector[b_1based - 1]),
            "Upper": int(years_vector[up_idx])
        })
    return pd.DataFrame(rows)

def style_pubr_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=11)

def main():
    # --- 2. DATA LOADING & PROCESSING ---
    df_all = pd.read_csv("Baltic_variables_Mt_4_Causal_Input.csv")
    df_all = df_all[(df_all["Year"] >= S_DATE) & (df_all["Year"] <= E_DATE)].reset_index(drop=True)

    # A. External Nutrient Loads
    nutrients_wide = df_all[["Year", "TN_load", "TP_load"]].copy()
    ts_nutrients = r_scale(nutrients_wide[["TN_load", "TP_load"]].values)
    ci_nutrients = get_bp_ci_mapped(ts_nutrients, YEARS_VEC)

    # B. HELCOM Biogeochemistry ALL REGION
    msfd_vars = ["DIP_ALL", "DIN_ALL", "CHLA_ALL", "DO2_Mt_ALL", "Secchi_ALL"]
    msfd_scaled = pd.DataFrame(r_scale(df_all[msfd_vars].values), columns=[v.replace("_ALL", "") for v in msfd_vars])
    msfd_scaled["Year"] = df_all["Year"].values
    ci_msfd = get_bp_ci_mapped(msfd_scaled[[v.replace("_ALL", "") for v in msfd_vars]].values, YEARS_VEC)

    # C. Coupled System (TTI, GES, LAG1, TN, TP)
    coupled_cols = ["TTI_ALL", "GES_ALL", "LAG1_ALL", "TN_load", "TP_load"]
    coupled_names = ["TTI", "GES", "LAG1", "TN_load", "TP_load"]
    grand_scaled = pd.DataFrame(r_scale(df_all[coupled_cols].values), columns=coupled_names)
    grand_scaled["Year"] = df_all["Year"].values
    ci_grand = get_bp_ci_mapped(grand_scaled[coupled_names].values, YEARS_VEC)

    print("--- Figure 1: Structural Breakpoints & 95% CIs ---")
    print("Panel A (Nutrient Loads):\n", ci_nutrients.to_string(index=False))
    print("Panel B (MSFD Indicators):\n", ci_msfd.to_string(index=False))
    print("Panel C (Coupled System):\n", ci_grand.to_string(index=False))

    # --- 3. PLOT GENERATION ---
    fig = plt.figure(figsize=(12, 14), dpi=300)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.05], hspace=0.38)

    # Panel A: 2 facets side-by-side (TN_load, TP_load)
    gs_a = gs[0].subgridspec(1, 2, wspace=0.18)
    ax_a1 = fig.add_subplot(gs_a[0])
    ax_a2 = fig.add_subplot(gs_a[1])

    for idx, (col, ax, color) in enumerate([("TN_load", ax_a1, OKABE_ITO[0]), ("TP_load", ax_a2, OKABE_ITO[1])]):
        for _, row in ci_nutrients.iterrows():
            ax.axvspan(row["Lower"], row["Upper"], color="#B3B3B3", alpha=0.4, zorder=1)
            ax.axvline(row["Break"], color="black", linestyle="--", linewidth=1.1, zorder=2)
        ax.plot(nutrients_wide["Year"], nutrients_wide[col], color=color, linewidth=1.5, zorder=3)
        ax.set_title(col, fontsize=12, fontweight="bold", bbox=dict(boxstyle="square,pad=0.3", facecolor="#F2F2F2", edgecolor="black", linewidth=0.6))
        ax.set_xticks(range(1960, 2030, 10))
        if idx == 0:
            ax.set_ylabel("Load (kt)", fontsize=12, fontweight="bold")
        style_pubr_ax(ax)

    bp_str_a = ", ".join(str(b) for b in ci_nutrients["Break"])
    ax_a1.text(0.0, 1.24, "Baltic Sea Nutrient Loadings (TN & TP)", transform=ax_a1.transAxes, fontsize=15, fontweight="bold")
    ax_a1.text(0.0, 1.13, f"Detected Breakpoints: {bp_str_a}", transform=ax_a1.transAxes, fontsize=11.5, color="#333333")
    ax_a1.text(-0.12, 1.26, "A", transform=ax_a1.transAxes, fontsize=18, fontweight="bold")

    # Panel B: Standardized MSFD Indicators
    ax_b = fig.add_subplot(gs[1])
    for _, row in ci_msfd.iterrows():
        ax_b.axvspan(row["Lower"], row["Upper"], color="#B3B3B3", alpha=0.4, zorder=1)
        ax_b.axvline(row["Break"], color="black", linestyle="--", linewidth=1.1, zorder=2)
    ax_b.axhline(0, color="black", linestyle=":", alpha=0.6, linewidth=1.0, zorder=2)

    ind_order = sorted([v.replace("_ALL", "") for v in msfd_vars])
    for i, var in enumerate(ind_order):
        ax_b.plot(msfd_scaled["Year"], msfd_scaled[var], label=var, color=OKABE_ITO[i], linewidth=1.4, alpha=0.85, zorder=3)

    bp_str_b = " and ".join(str(b) for b in ci_msfd["Break"])
    ax_b.set_title(f"Temporal Evolution of Standardized MSFD Indicators (1954-2022)\nMultivariate Regime Shifts detected at {bp_str_b}",
                   loc="left", fontsize=14, fontweight="bold", pad=10)
    ax_b.set_ylabel("Standardized Anomaly (Z-Score)", fontsize=12, fontweight="bold")
    ax_b.set_xticks(range(1960, 2030, 10))
    ax_b.legend(title="Indicator", title_fontsize=11, fontsize=10, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax_b.text(-0.06, 1.12, "B", transform=ax_b.transAxes, fontsize=18, fontweight="bold")
    style_pubr_ax(ax_b)

    # Panel C: Coupled System
    ax_c = fig.add_subplot(gs[2])
    for _, row in ci_grand.iterrows():
        ax_c.axvspan(row["Lower"], row["Upper"], color="#B3B3B3", alpha=0.4, zorder=1)
        ax_c.axvline(row["Break"], color="black", linestyle="--", linewidth=1.1, zorder=2)

    metric_order = sorted(coupled_names)
    for i, m in enumerate(metric_order):
        ls = "-" if m in ["TN_load", "TP_load"] else "--"
        lbl_type = "Driver (Nutrients)" if m in ["TN_load", "TP_load"] else "Response (Eco)"
        ax_c.plot(grand_scaled["Year"], grand_scaled[m], label=f"{m} ({lbl_type})", color=OKABE_ITO[i], linestyle=ls, linewidth=1.4, zorder=3)

    bp_str_c = ", ".join(str(b) for b in ci_grand["Break"])
    ax_c.set_title(f"Baltic Sea Coupled System (Drivers + Response)\nUnified Breakpoints: {bp_str_c}",
                   loc="left", fontsize=14, fontweight="bold", pad=10)
    ax_c.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax_c.set_ylabel("Standardized Value (Z-Score)", fontsize=12, fontweight="bold")
    ax_c.set_xticks(range(1960, 2030, 10))
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=10, frameon=False)
    ax_c.text(-0.06, 1.12, "C", transform=ax_c.transAxes, fontsize=18, fontweight="bold")
    style_pubr_ax(ax_c)

    plt.savefig("Figure_1_Baltic_Asynchronous_Shifts.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Analysis complete. Figure 1 successfully saved.")

if __name__ == "__main__":
    main()
