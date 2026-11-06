#!/usr/bin/env python3
"""
Baltic Ecosystem Nutrients Narrative
SUPPLEMENTARY FIGURE 3: TP, Breakpoints, CCF and snail trail
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

GES_THRESHOLD = 0.71
BREAKS_FINAL = [1974, 1988, 2012]
HIGHLIGHT_YEAR = 1988

def r_ccf(x, y, lag_max=24):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    denom = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
    lags = np.arange(-lag_max, lag_max + 1)
    corrs = np.zeros(len(lags))
    for idx, k in enumerate(lags):
        if k >= 0:
            corrs[idx] = np.sum(xm[k:] * ym[:n - k]) / denom
        else:
            kk = -k
            corrs[idx] = np.sum(xm[:n - kk] * ym[kk:]) / denom
    return lags, corrs

def style_pubr_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10.5)

def main():
    grand_df = pd.read_csv("Baltic_variables_Mt_4_Causal_Input.csv")
    grand_df = grand_df[(grand_df["Year"] >= 1954) & (grand_df["Year"] <= 2022)].reset_index(drop=True)
    grand_df = grand_df.rename(columns={"TTI_ALL": "TTI", "GES_ALL": "GES", "LAG1_ALL": "LAG1"})

    print("--- Unified System Breakpoints (5 Vars) ---")
    print(BREAKS_FINAL)

    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.1], hspace=0.35)

    # --- PLOT A: TIME SERIES (The History) ---
    ax_top = fig.add_subplot(gs[0])
    ax_top.axvspan(1954, 1974, color="#009E73", alpha=0.06, zorder=0)
    ax_top.axvspan(1974, 1988, color="#E69F00", alpha=0.06, zorder=0)
    ax_top.axvspan(1988, 2012, color="#D55E00", alpha=0.06, zorder=0)
    ax_top.axvspan(2012, 2022, color="#56B4E9", alpha=0.06, zorder=0)

    ax_top.axhline(GES_THRESHOLD, linestyle="--", color="#009E73", linewidth=1.2, zorder=2)
    ax_top.text(1960, GES_THRESHOLD + 0.005, f"GES ({GES_THRESHOLD})", color="#009E73", fontsize=12, ha="center", va="bottom", zorder=3)

    ax_top.plot(grand_df["Year"], grand_df["TTI"], color="black", linewidth=1.5, zorder=3)
    ax_top.scatter(grand_df["Year"], grand_df["TTI"], color="black", s=22, zorder=4)

    for bp in BREAKS_FINAL:
        ax_top.axvline(bp, color="black", linestyle=":", linewidth=1.0, zorder=2)

    ax_top.set_xticks(range(1960, 2030, 10))
    ax_top.set_title("Baltic Sea Ecosystem State History (1954-2022)\nShaded regions indicate regimes",
                     loc="left", fontsize=14, fontweight="bold", pad=10)
    ax_top.set_ylabel("Trophic Transfer Index (TTI)", fontsize=12, fontweight="bold")
    style_pubr_ax(ax_top)

    # Bottom row: CCF + Snail Trail
    gs_bot = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.22)

    # --- PLOT B: CCF ---
    ax_ccf = fig.add_subplot(gs_bot[0])
    lags, corrs = r_ccf(grand_df["TP_load"], grand_df["TTI"], lag_max=24)
    max_idx = np.argmax(corrs)
    max_lag = int(lags[max_idx])
    sig_level = 2.0 / np.sqrt(len(grand_df))

    bar_colors = ["#D55E00" if l == max_lag else "#B3B3B3" for l in lags]
    ax_ccf.bar(lags, corrs, width=0.8, color=bar_colors, zorder=2)
    ax_ccf.scatter([max_lag], [corrs[max_idx]], color="orange", s=50, zorder=4)

    ax_ccf.axhline(0, color="black", linewidth=0.6, zorder=3)
    ax_ccf.axhline(sig_level, color="blue", linestyle="--", alpha=0.5, linewidth=1.0, zorder=3)
    ax_ccf.axhline(-sig_level, color="blue", linestyle="--", alpha=0.5, linewidth=1.0, zorder=3)

    ax_ccf.set_xlim(-6.5, 24.5)
    ax_ccf.set_xticks(range(-6, 25, 2))
    ax_ccf.set_title(f"Cross-Correlation: TP Load vs. TTI Score\nMax Correlation at Lag: {max_lag} years",
                     loc="left", fontsize=13.5, fontweight="bold", pad=10)
    ax_ccf.set_xlabel("Lag (Years) - Positive means TP lead TTI", fontsize=12, fontweight="bold")
    ax_ccf.set_ylabel("Correlation Coefficient", fontsize=12, fontweight="bold")
    style_pubr_ax(ax_ccf)

    # --- PLOT C: SNAIL TRAIL (HYSTERESIS) ---
    gs_hyst = gs_bot[1].subgridspec(1, 2, width_ratios=[18, 1], wspace=0.05)
    ax_hyst = fig.add_subplot(gs_hyst[0])
    cax = fig.add_subplot(gs_hyst[1])

    x_tp = grand_df["TP_load"].values
    y_tti = grand_df["TTI"].values
    cmap = plt.get_cmap("magma_r")
    norm = Normalize(vmin=grand_df["Year"].min(), vmax=grand_df["Year"].max())

    ax_hyst.plot(x_tp, y_tti, color="#999999", linewidth=1.2, alpha=0.8, zorder=1)
    ax_hyst.annotate("", xy=(x_tp[-1], y_tti[-1]), xytext=(x_tp[-2], y_tti[-2]),
                     arrowprops=dict(arrowstyle="-|>", color="#808080", lw=1.2, mutation_scale=14), zorder=2)

    sc = ax_hyst.scatter(x_tp, y_tti, c=grand_df["Year"], cmap=cmap, norm=norm,
                         s=65, edgecolors="black", linewidths=0.5, zorder=3)

    row_88 = grand_df[grand_df["Year"] == HIGHLIGHT_YEAR].iloc[0]
    ax_hyst.scatter([row_88["TP_load"]], [row_88["TTI"]], facecolors="none", edgecolors="red",
                    s=220, linewidths=2.0, zorder=4)

    for yr in BREAKS_FINAL:
        row = grand_df[grand_df["Year"] == yr].iloc[0]
        ax_hyst.annotate(str(yr), xy=(row["TP_load"], row["TTI"]),
                         xytext=(0, 10), textcoords="offset points",
                         fontsize=10.5, fontweight="bold", ha="center", zorder=5)

    load_lim = (x_tp.min() - 5, x_tp.max() + 1)
    ax_hyst.axhline(GES_THRESHOLD, linestyle="--", color="#009E73", linewidth=1.2, zorder=2)
    ax_hyst.text(load_lim[0] * 1.07 + 15, GES_THRESHOLD + 0.005, f"GES ({GES_THRESHOLD})",
                 color="#009E73", fontsize=11.5, ha="center", va="bottom", zorder=3)

    ax_hyst.set_xlim(load_lim)
    ax_hyst.set_title("Phase-Space Trajectories Baltic Sea: TTI\nClockwise trajectory: Degradation vs Recovery",
                      loc="left", fontsize=13.5, fontweight="bold", pad=10)
    ax_hyst.set_xlabel("Total TP Load (kt)", fontsize=12, fontweight="bold")
    ax_hyst.set_ylabel("TTI [-]", fontsize=12, fontweight="bold")
    style_pubr_ax(ax_hyst)

    cb = plt.colorbar(sc, cax=cax)
    cb.set_label("Year", fontsize=11, fontweight="bold")

    fig.suptitle("Dynamics of Baltic Sea Ecosystem Degradation", fontsize=18, fontweight="bold", x=0.12, ha="left", y=0.97)

    plt.savefig("Figure_SM3_Baltic_TTI_LAG_TP_CCF_Analysis_Composite.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Baltic narrative Analysis complete")

if __name__ == "__main__":
    main()
