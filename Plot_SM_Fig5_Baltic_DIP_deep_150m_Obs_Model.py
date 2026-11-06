#!/usr/bin/env python3
"""
Baltic Ecosystem Model Validation
SUPPLEMENTARY FIGURE 5: Deep Water (150m) DIP - Observations vs. Model (EGB)
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    # 1. LOAD AND PREP MODEL DATA (150m)
    rows = []
    with open("EGB_183_586_model_mon_DIP.txt", "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and re.match(r"^[0-9]+$", parts[0]):
                rows.append((int(parts[0]), float(parts[2])))

    model_clean = pd.DataFrame(rows, columns=["Year", "Model_DIP"])
    model_annual = model_clean.groupby("Year", as_index=False)["Model_DIP"].mean()

    # 2. LOAD AND PREP ICES OBSERVATIONAL DATA (150m)
    obs_raw = pd.read_csv("Phosphate_EGB_ICES_18.24_58.58_150m_1960_2022.csv")
    obs_raw["Year"] = obs_raw["yyyy-mm-ddThh:mm:ss.sss"].astype(str).str[:4].astype(int)
    obs_raw["Obs_DIP"] = pd.to_numeric(obs_raw["Phosphate (PHOSZZXX_UPOX) [umol/l]"], errors="coerce")

    def calc_se(s):
        valid = s.dropna()
        return valid.std(ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else np.nan

    obs_annual = obs_raw.groupby("Year", as_index=False).agg(
        Obs_DIP_mean=("Obs_DIP", "mean"),
        Obs_DIP_se=("Obs_DIP", calc_se)
    )
    obs_annual["CI_lower"] = obs_annual["Obs_DIP_mean"] - 1.96 * obs_annual["Obs_DIP_se"]
    obs_annual["CI_upper"] = obs_annual["Obs_DIP_mean"] + 1.96 * obs_annual["Obs_DIP_se"]

    # 3. STATISTICAL EVALUATION
    df_eval = pd.merge(obs_annual, model_annual, on="Year").dropna(subset=["Obs_DIP_mean", "Model_DIP"])
    r_val = df_eval["Obs_DIP_mean"].corr(df_eval["Model_DIP"])
    rmse_val = np.sqrt(np.mean((df_eval["Obs_DIP_mean"] - df_eval["Model_DIP"]) ** 2))
    bias_val = np.mean(df_eval["Model_DIP"] - df_eval["Obs_DIP_mean"])

    eval_metrics = pd.DataFrame([{
        "Station": "EGB 26DA (150m)",
        "Variable": "Phosphate (DIP)",
        "Pearson_r": round(r_val, 3),
        "RMSE": round(rmse_val, 3),
        "Mean_Bias": round(bias_val, 3)
    }])
    print("--- Deep Station (150m) Statistical Skill ---")
    print(eval_metrics.to_string(index=False))

    # 4. PLOTTING
    color_dip = "#009E73"
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    ax.fill_between(obs_annual["Year"], obs_annual["CI_lower"], obs_annual["CI_upper"],
                    color=color_dip, alpha=0.4, zorder=1)
    ax.plot(obs_annual["Year"], obs_annual["Obs_DIP_mean"], color=color_dip, linewidth=1.6, zorder=3)
    ax.scatter(obs_annual["Year"], obs_annual["Obs_DIP_mean"], color=color_dip, edgecolors="black",
               linewidths=0.6, s=40, zorder=4)

    ax.plot(model_annual["Year"], model_annual["Model_DIP"], color="#4D4D4D", linewidth=1.5, linestyle="--", zorder=2)
    ax.scatter(model_annual["Year"], model_annual["Model_DIP"], color="#4D4D4D", marker="^", s=30, zorder=3)

    ax.set_xticks(range(1960, 2030, 10))
    ax.set_title("Eastern Gotland Basin (150m): Observations vs. Blue2 Model\nAnnual Mean Deep-Water DIP (Green = ICES Obs with 95% CI | Grey Triangles = Blue2 Model)",
                 loc="left", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax.set_ylabel("Phosphate ($\mu$mol / L)", fontsize=13, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    plt.savefig("Figure_SM5_Baltic_Deep_150m_DIP.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Analysis complete. Figure SM5 successfully saved.")

if __name__ == "__main__":
    main()
