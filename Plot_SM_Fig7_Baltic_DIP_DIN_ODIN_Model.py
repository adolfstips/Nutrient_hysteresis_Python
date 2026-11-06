#!/usr/bin/env python3
"""
Baltic Ecosystem Model Validation
SUPPLEMENTARY FIGURE 7: Arkona (Shallow) and EGB (Deep) Obs vs. Model
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_cdo(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and re.match(r"^[0-9]+$", parts[0]):
                rows.append((int(parts[0]), float(parts[1])))
    return pd.DataFrame(rows, columns=["Year", "Value"])

def calc_metrics(df, region_name):
    rows = []
    for nut, obs_col, mod_col in [("Nitrate", "Nitrate_mean", "Model_Nitrate"),
                                  ("Phosphate", "Phosphate_mean", "Model_Phosphate")]:
        r_val = df[obs_col].corr(df[mod_col])
        rmse = np.sqrt(np.mean((df[obs_col] - df[mod_col]) ** 2))
        bias = np.mean(df[mod_col] - df[obs_col])
        rows.append({
            "Region": region_name,
            "Nutrient": nut,
            "Pearson_r": round(r_val, 3),
            "RMSE": round(rmse, 3),
            "Mean_Bias": round(bias, 3)
        })
    return pd.DataFrame(rows)

def plot_basin_facets(ax_nit, ax_phos, obs_df, mod_df, basin_title, hide_x=False):
    colors = {"Nitrate": "#E69F00", "Phosphate": "#009E73"}
    for ax, nut in [(ax_nit, "Nitrate"), (ax_phos, "Phosphate")]:
        col = colors[nut]
        mean_col = f"{nut}_mean"
        rms_col = f"{nut}_rms"
        mod_col = f"Model_{nut}"

        ci_low = obs_df[mean_col] - 1.96 * obs_df[rms_col]
        ci_high = obs_df[mean_col] + 1.96 * obs_df[rms_col]

        ax.fill_between(obs_df["Year"], ci_low, ci_high, color=col, alpha=0.4, zorder=1)
        ax.plot(obs_df["Year"], obs_df[mean_col], color=col, linewidth=1.5, zorder=3)
        ax.scatter(obs_df["Year"], obs_df[mean_col], color=col, edgecolors="black", linewidths=0.6, s=38, zorder=4)

        ax.plot(mod_df["Year"], mod_df[mod_col], color="#4D4D4D", linewidth=1.4, linestyle="--", zorder=2)
        ax.scatter(mod_df["Year"], mod_df[mod_col], color="#4D4D4D", marker="^", s=28, zorder=3)

        ax.set_title(nut, fontsize=12, fontweight="bold",
                     bbox=dict(boxstyle="square,pad=0.25", facecolor="#F2F2F2", edgecolor="none"))
        ax.set_xticks(range(1970, 2030, 10))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if hide_x:
            ax.tick_params(labelbottom=False, bottom=False)

    ax_nit.text(0.0, 1.25, basin_title, transform=ax_nit.transAxes, fontsize=14, fontweight="bold")

def main():
    obs_ark = pd.read_csv("Arkona_annual_means_summary.csv")
    obs_ark = obs_ark[(obs_ark["Year"] >= 1969) & (obs_ark["Year"] <= 2022)].reset_index(drop=True)

    obs_egb = pd.read_csv("EGB_annual_means_summary.csv")
    obs_egb = obs_egb[(obs_egb["Year"] >= 1969) & (obs_egb["Year"] <= 2022)].reset_index(drop=True)

    ark_din = read_cdo("ARK_model_DIN.txt")
    ark_dip = read_cdo("ARK_model_DIP.txt")
    egb_din = read_cdo("EGB_model_DIN.txt")
    egb_dip = read_cdo("EGB_model_DIP.txt")

    model_shallow = pd.merge(ark_din.rename(columns={"Value": "Model_Nitrate"}),
                             ark_dip.rename(columns={"Value": "Model_Phosphate"}), on="Year")
    model_shallow = model_shallow[(model_shallow["Year"] >= 1969) & (model_shallow["Year"] <= 2022)].reset_index(drop=True)

    model_deep = pd.merge(egb_din.rename(columns={"Value": "Model_Nitrate"}),
                          egb_dip.rename(columns={"Value": "Model_Phosphate"}), on="Year")
    model_deep = model_deep[(model_deep["Year"] >= 1969) & (model_deep["Year"] <= 2022)].reset_index(drop=True)

    eval_ark = pd.merge(obs_ark, model_shallow, on="Year").dropna()
    eval_egb = pd.merge(obs_egb, model_deep, on="Year").dropna()

    metrics_combined = pd.concat([
        calc_metrics(eval_ark, "Arkona (Shallow)"),
        calc_metrics(eval_egb, "EGB (Deep)")
    ], ignore_index=True)

    print("--- Model vs. Observation Statistical Skill ---")
    print(metrics_combined.to_string(index=False))

    fig = plt.figure(figsize=(11, 12), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.28)

    gs_a = gs[0].subgridspec(2, 1, hspace=0.28)
    ax_a1 = fig.add_subplot(gs_a[0])
    ax_a2 = fig.add_subplot(gs_a[1])
    plot_basin_facets(ax_a1, ax_a2, obs_ark, model_shallow,
                      "A. Arkona Basin (Shallow): Observations vs. Blue2 Model", hide_x=True)

    gs_b = gs[1].subgridspec(2, 1, hspace=0.28)
    ax_b1 = fig.add_subplot(gs_b[0])
    ax_b2 = fig.add_subplot(gs_b[1])
    plot_basin_facets(ax_b1, ax_b2, obs_egb, model_deep,
                      "B. Eastern Gotland Basin (Deep): Observations vs. Blue2 Model", hide_x=False)
    ax_b2.set_xlabel("Year", fontsize=13, fontweight="bold")

    fig.text(0.04, 0.72, "Conc. ($\mathrm{mmol / m^3}$)", fontsize=12, fontweight="bold", va="center", rotation="vertical")
    fig.text(0.04, 0.27, "Conc. ($\mathrm{mmol / m^3}$)", fontsize=12, fontweight="bold", va="center", rotation="vertical")
    fig.text(0.125, 0.955, "Annual Mean 0-10m Concentration (Colored = ODIN Obs with 95% CI | Grey Triangles = Blue2 Model)",
             fontsize=11.5, color="#333333")

    plt.savefig("Figure_SM7_Baltic_ODIN_vs_Model_Combined.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Analysis complete. Composite figure SM7 successfully saved.")

if __name__ == "__main__":
    main()
