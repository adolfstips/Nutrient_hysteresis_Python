#!/usr/bin/env python3
"""
Baltic Ecosystem Phase-Space Trajectories (Hysteresis)
Figure 4 for hysteresis article (Deep Regions)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

GES_THRESHOLD = 0.71
BP_YEARS = [1974, 1988, 2012]
HIGHLIGHT_YEAR = 1988

def style_pubr_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10.5)

def main():
    grand_df = pd.read_csv("Baltic_variables_Mt_4_Causal_Input.csv")
    grand_df = grand_df[(grand_df["Year"] >= 1954) & (grand_df["Year"] <= 2022)].sort_values("Year").reset_index(drop=True)

    cmap = plt.get_cmap("magma_r")
    norm = Normalize(vmin=grand_df["Year"].min(), vmax=grand_df["Year"].max())

    fig = plt.figure(figsize=(12, 11), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.35)

    # --- PLOT A: TTI HYSTERESIS (PHASE-SPACE) ---
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[22, 1], wspace=0.04)
    ax_a = fig.add_subplot(gs_top[0])
    cax = fig.add_subplot(gs_top[1])

    x_tp = grand_df["TP_load"].values
    y_tti = grand_df["TTI_DEEP"].values

    ax_a.plot(x_tp, y_tti, color="#999999", linewidth=1.2, alpha=0.8, zorder=1)
    # Add directional arrow at end of trajectory
    ax_a.annotate("", xy=(x_tp[-1], y_tti[-1]), xytext=(x_tp[-2], y_tti[-2]),
                  arrowprops=dict(arrowstyle="-|>", color="#808080", lw=1.2, mutation_scale=14), zorder=2)

    ax_a.axhline(GES_THRESHOLD, linestyle="--", color="#009E73", linewidth=1.3, zorder=2)
    ax_a.text(x_tp.min() + 3, GES_THRESHOLD + 0.008, f"GES ({GES_THRESHOLD})", color="#009E73",
              fontsize=12, fontweight="bold", ha="center", va="bottom", zorder=3)

    sc = ax_a.scatter(x_tp, y_tti, c=grand_df["Year"], cmap=cmap, norm=norm,
                      s=85, edgecolors="black", linewidths=0.7, zorder=4)

    # Annotate breakpoint years
    offsets_a = {1974: (-12, -16), 1988: (-18, 2), 2012: (6, 12)}
    for yr in BP_YEARS:
        row = grand_df[grand_df["Year"] == yr].iloc[0]
        dx, dy = offsets_a.get(yr, (5, 5))
        ax_a.annotate(str(yr), xy=(row["TP_load"], row["TTI_DEEP"]),
                      xytext=(dx, dy), textcoords="offset points",
                      fontsize=11, fontweight="bold", color="black",
                      arrowprops=dict(arrowstyle="-", color="grey", lw=0.6), zorder=6)

    # Emphasize 1988 Regime Shift (Red Halo)
    row_88 = grand_df[grand_df["Year"] == HIGHLIGHT_YEAR].iloc[0]
    ax_a.scatter([row_88["TP_load"]], [row_88["TTI_DEEP"]], facecolors="none", edgecolors="red",
                 s=240, linewidths=2.2, zorder=5)
    ax_a.annotate("Regime Shift", xy=(row_88["TP_load"], row_88["TTI_DEEP"]),
                  xytext=(-35, -32), textcoords="offset points",
                  fontsize=12, fontweight="bold", color="red",
                  arrowprops=dict(arrowstyle="-", color="red", lw=1.0), zorder=6)

    ax_a.set_title("Phase-Space Trajectories Deep Baltic Sea: TTI\nClockwise trajectory: Degradation vs Recovery",
                   loc="left", fontsize=14, fontweight="bold", pad=10)
    ax_a.set_xlabel("Total Phosphorus Load (kt)", fontsize=12, fontweight="bold")
    ax_a.set_ylabel("TTI [-]", fontsize=12, fontweight="bold")
    ax_a.text(-0.07, 1.10, "A", transform=ax_a.transAxes, fontsize=18, fontweight="bold")
    style_pubr_ax(ax_a)

    cb = plt.colorbar(sc, cax=cax)
    cb.set_label("Year", fontsize=12, fontweight="bold")

    # --- PLOT B: BIOGEOCHEMISTRY FACETS ---
    gs_bot = gs[1].subgridspec(1, 3, wspace=0.22)
    facets = [
        ("CHLA_DEEP", "Chlorophyll-a (Deep)"),
        ("DIP_DEEP", "DIP (Deep)"),
        ("DIN_DEEP", "DIN (Deep)")
    ]

    for idx, (col, title_lbl) in enumerate(facets):
        ax = fig.add_subplot(gs_bot[idx])
        yv = grand_df[col].values
        ax.plot(x_tp, yv, color="#999999", linewidth=1.0, alpha=0.8, zorder=1)
        ax.scatter(x_tp, yv, c=grand_df["Year"], cmap=cmap, norm=norm,
                   s=45, edgecolors="black", linewidths=0.6, zorder=3)

        for yr in BP_YEARS:
            row = grand_df[grand_df["Year"] == yr].iloc[0]
            ax.annotate(str(yr), xy=(row["TP_load"], row[col]),
                        xytext=(4, 6), textcoords="offset points",
                        fontsize=9.5, fontweight="bold", color="black", zorder=5)

        ax.set_title(title_lbl, fontsize=11.5, fontweight="bold",
                     bbox=dict(boxstyle="square,pad=0.3", facecolor="#F2F2F2", edgecolor="none"))
        if idx == 1:
            ax.set_xlabel("Total Phosphorus Load (kt)", fontsize=12, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Concentration ($\mathrm{mg/m^3}$)", fontsize=12, fontweight="bold")
        style_pubr_ax(ax)

        if idx == 0:
            ax.text(0.0, 1.20, "Diverging Memory Effects in Deep Baltic Biogeochemistry",
                    transform=ax.transAxes, fontsize=14, fontweight="bold")
            ax.text(0.0, 1.11, "Anti-clockwise hysteresis (CHLA, DIP) vs. Clockwise feedback (DIN)",
                    transform=ax.transAxes, fontsize=11, color="#4D4D4D")
            ax.text(-0.22, 1.22, "B", transform=ax.transAxes, fontsize=18, fontweight="bold")

    plt.savefig("Figure_4_Baltic_Hysteresis_Combined.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Analysis complete. Figure 4 successfully saved.")

if __name__ == "__main__":
    main()
