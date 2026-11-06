#!/usr/bin/env python3
"""
Baltic Ecosystem Model Validation
SUPPLEMENTARY FIGURE 6: Regional Deep Water (>50m) DIP
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def clean_regional_data(df, region_name):
    out = pd.DataFrame()
    out["Year"] = df["yyyy-mm-ddThh:mm:ss.sss"].astype(str).str[:4].astype(int)
    out["DIP"] = pd.to_numeric(df["Phosphate (PHOSZZXX_UPOX) [umol/l]"], errors="coerce")
    out["Depth"] = pd.to_numeric(df["Depth (ADEPZZ01_ULAA) [m]"], errors="coerce")
    out = out[(out["Depth"] > 50) & (out["DIP"].notna())].copy()
    out["Region"] = region_name
    return out

def calc_se(s):
    valid = s.dropna()
    return valid.std(ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else np.nan

def main():
    # 1. LOAD AND PREP OBSERVATIONAL DATA
    # If full raw CSVs are present, aggregate them; otherwise fall back to pre-aggregated summary CSV
    if os.path.exists("Gotland_dip_gt50m.csv") and os.path.exists("Bornholm_dip_gt50m.csv"):
        df_c_raw = pd.read_csv("Gotland_dip_gt50m.csv", low_memory=False)
        df_b_raw = pd.read_csv("Bornholm_dip_gt50m.csv", low_memory=False)
        df_comb = pd.concat([
            clean_regional_data(df_c_raw, "Central Baltic"),
            clean_regional_data(df_b_raw, "Bornholm Basin")
        ], ignore_index=True)
        if len(df_comb) > 100:
            df_annual = df_comb.groupby(["Year", "Region"], as_index=False).agg(
                DIP_mean=("DIP", "mean"),
                DIP_se=("DIP", calc_se)
            )
            df_annual["CI_lower"] = df_annual["DIP_mean"] - 1.96 * df_annual["DIP_se"]
            df_annual["CI_upper"] = df_annual["DIP_mean"] + 1.96 * df_annual["DIP_se"]
            df_annual.to_csv("ICES_Bornholm_Gotland_DIP_annual_gt50m.csv", index=False)
        else:
            df_annual = pd.read_csv("ICES_Bornholm_Gotland_DIP_annual_gt50m.csv")
    else:
        df_annual = pd.read_csv("ICES_Bornholm_Gotland_DIP_annual_gt50m.csv")

    # 2. PLOTTING
    okabe_ito_regions = {"Central Baltic": "#0072B2", "Bornholm Basin": "#D55E00"}
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for region in ["Bornholm Basin", "Central Baltic"]:
        sub = df_annual[df_annual["Region"] == region].sort_values("Year")
        col = okabe_ito_regions[region]
        ax.fill_between(sub["Year"], sub["CI_lower"], sub["CI_upper"], color=col, alpha=0.3, zorder=1)
        ax.plot(sub["Year"], sub["DIP_mean"], color=col, linewidth=1.6, label=region, zorder=3)
        ax.scatter(sub["Year"], sub["DIP_mean"], color=col, edgecolors="black", linewidths=0.6, s=40, zorder=4)

    ax.set_ylim(0, 4.2)
    ax.set_xticks(range(1960, 2030, 10))
    ax.set_title("Regional Deep-Water (>50m) ICES Observational Data\nAnnual Mean DIP (Solid lines with 95% Confidence Intervals)",
                 loc="left", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax.set_ylabel("Phosphate ($\mu$mol / L)", fontsize=13, fontweight="bold")

    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=2, frameon=True, fontsize=11.5)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    plt.savefig("Figure_SM6_Baltic_Regional_Deep_DIP.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Analysis complete. Figure SM6 successfully saved.")

if __name__ == "__main__":
    main()
