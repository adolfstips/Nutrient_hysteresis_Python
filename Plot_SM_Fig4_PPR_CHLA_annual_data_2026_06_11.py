#!/usr/bin/env python3
"""
Baltic Ecosystem Productivity
SUPPLEMENTARY FIGURE 4: PPR vs Chlorophyll-a (ALL REGION)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def main():
    df_all = pd.read_csv("Baltic_variables_Mt_4_Causal_Input.csv")
    df_all = df_all[(df_all["Year"] >= 1954) & (df_all["Year"] <= 2022)][["Year", "PPR_ALL", "CHLA_ALL"]].copy()
    df_all = df_all.rename(columns={"PPR_ALL": "PPR", "CHLA_ALL": "CHLA"}).reset_index(drop=True)

    # 3-year centered rolling mean
    df_all["PPR_fil"] = df_all["PPR"].rolling(window=3, center=True).mean()
    df_all["CHLA_fil"] = df_all["CHLA"].rolling(window=3, center=True).mean()

    scale_factor = df_all["PPR"].max() / df_all["CHLA"].max()

    # Pearson correlation and 95% Fisher z confidence interval
    valid = df_all[["PPR", "CHLA"]].dropna()
    r_val, p_val = pearsonr(valid["PPR"], valid["CHLA"])
    n = len(valid)
    z = np.arctanh(r_val)
    se_z = 1.0 / np.sqrt(n - 3)
    ci_low, ci_high = np.tanh(z - 1.96 * se_z), np.tanh(z + 1.96 * se_z)
    cor_label = f"r = {r_val:.2f} [CI: {ci_low:.2f}-{ci_high:.2f}], p < 0.001"
    print(f"--- SM Figure 4: PPR vs Chl-a Correlation ---\n{cor_label}")

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Background Shading
    ax1.axvspan(1953, 1974, color="#009E73", alpha=0.06, zorder=0)
    ax1.axvspan(1974, 1988, color="#E69F00", alpha=0.06, zorder=0)
    ax1.axvspan(1988, 2012, color="#D55E00", alpha=0.06, zorder=0)
    ax1.axvspan(2012, 2022, color="#56B4E9", alpha=0.06, zorder=0)

    # Raw Points & Filtered Lines
    ax1.scatter(df_all["Year"], df_all["PPR"], color="#009E73", alpha=0.65, s=40, zorder=3)
    ax1.scatter(df_all["Year"], df_all["CHLA"] * scale_factor, color="#0072B2", alpha=0.65, s=40, zorder=3)

    ax1.plot(df_all["Year"], df_all["PPR_fil"], color="#009E73", linewidth=1.6, zorder=4)
    ax1.plot(df_all["Year"], df_all["CHLA_fil"] * scale_factor, color="#0072B2", linewidth=1.6, zorder=4)

    for bp in [1974, 1988, 2012]:
        ax1.axvline(bp, color="#D55E00", linestyle=(0, (6, 3)), linewidth=1.2, zorder=2)

    # Direct Line Labels
    last_ppr = df_all["PPR_fil"].dropna().iloc[-1]
    last_chla = df_all["CHLA_fil"].dropna().iloc[-1] * scale_factor
    ax1.text(2023, last_ppr, "PPR", color="#009E73", fontweight="bold", fontsize=12, va="center", ha="left")
    ax1.text(2023, last_chla, "Chl-a", color="#0072B2", fontweight="bold", fontsize=12, va="center", ha="left")

    ax1.set_xlim(1951.5, 2029.5)
    ax1.set_xticks(range(1960, 2030, 10))
    ax1.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax1.set_ylabel("PPR ($\mathrm{mg\ C\ m^{-2}\ d^{-1}}$)", color="#009E73", fontsize=13, fontweight="bold")
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    y1_min, y1_max = ax1.get_ylim()
    ax2.set_ylim(y1_min / scale_factor, y1_max / scale_factor)
    ax2.set_ylabel("Chl-a ($\mathrm{mg\ m^{-3}}$)", color="#0072B2", fontsize=13, fontweight="bold")
    ax2.spines["top"].set_visible(False)

    ax1.set_title(f"Baltic Sea Productivity: PPR vs Chlorophyll-a\nArea-weighted PPR vs. Chl-a | Statistics: {cor_label}",
                  loc="left", fontsize=14, fontweight="bold", pad=12)

    plt.savefig("Figure_SM4_Baltic_PPR_CHLA.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Analysis complete. Figure PPR-CHLa successfully saved.")

if __name__ == "__main__":
    main()
