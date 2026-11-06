#!/usr/bin/env python3
"""
Baltic Ecosystem TTI Analysis
Plot annual mean TTI (All, Deep, Shallow)
Figure 3 for hysteresis article
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

GES_THRESHOLD = 0.71
S_DATE = 1954
E_DATE = 2022

def loess_1d(x, y, span=0.18):
    """Local linear/quadratic tricube LOESS smoother replicating ggplot2 geom_smooth(method='loess')."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    k = max(5, int(np.ceil(span * n)))
    x_eval = np.linspace(x.min(), x.max(), 200)
    y_eval = np.zeros_like(x_eval)
    for idx, x0 in enumerate(x_eval):
        dists = np.abs(x - x0)
        part_idx = np.argsort(dists)[:k]
        max_d = np.max(dists[part_idx])
        if max_d == 0:
            y_eval[idx] = np.mean(y[part_idx])
            continue
        u = dists[part_idx] / max_d
        w = (1.0 - u ** 3) ** 3
        xw = x[part_idx] - x0
        yw = y[part_idx]
        X_mat = np.column_stack([np.ones_like(xw), xw, xw ** 2])
        W_mat = np.diag(w)
        try:
            beta = np.linalg.lstsq(X_mat.T @ W_mat @ X_mat, X_mat.T @ W_mat @ yw, rcond=None)[0]
            y_eval[idx] = beta[0]
        except np.linalg.LinAlgError:
            y_eval[idx] = np.sum(w * yw) / np.sum(w)
    return x_eval, y_eval

def main():
    df_all = pd.read_csv("Baltic_variables_Mt_4_Causal_Input.csv")
    df_all = df_all[(df_all["Year"] >= S_DATE) & (df_all["Year"] <= E_DATE)].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # A. Background Regime Shading
    ax.axvspan(1953, 1974, color="green", alpha=0.07, zorder=0)
    ax.axvspan(1974, 1988, color="orange", alpha=0.07, zorder=0)
    ax.axvspan(1988, 2012, color="red", alpha=0.07, zorder=0)
    ax.axvspan(2012, 2022, color="blue", alpha=0.07, zorder=0)

    # B. Reference Lines
    for bp in [1974, 1988, 2012]:
        ax.axvline(bp, color="red", linestyle=(0, (6, 3)), linewidth=1.2, zorder=2)

    ax.plot([1953, 2022], [GES_THRESHOLD, GES_THRESHOLD], linestyle="--", color="#009E73", linewidth=1.3, zorder=2)
    ax.text(1960, GES_THRESHOLD + 0.006, f"GES ({GES_THRESHOLD})", color="#009E73", fontsize=12, ha="center", va="bottom", zorder=3)

    # C. Lines and Points
    series_cfg = [
        ("TTI_ALL", "All", "#D55E00", "o", 2.2),
        ("TTI_DEEP", "Deep", "#0072B2", "^", 1.5),
        ("TTI_SHALLOW", "Shallow", "#009E73", "s", 1.5),
    ]

    max_year = df_all["Year"].max()
    for col, label_name, color, marker, lw in series_cfg:
        valid = df_all[["Year", col]].dropna()
        ax.scatter(valid["Year"], valid[col], color=color, marker=marker, s=45, alpha=0.8, zorder=3)
        x_sm, y_sm = loess_1d(valid["Year"].values, valid[col].values, span=0.18)
        ax.plot(x_sm, y_sm, color=color, linewidth=lw, zorder=4)

        # Direct Labels using last 3 years mean
        last3_mean = valid[valid["Year"] > (max_year - 3)][col].mean()
        ax.text(max_year + 1.5, last3_mean, label_name, color=color, fontsize=13, fontweight="bold", va="center", ha="left", zorder=5)

    ax.set_ylim(0.45, 0.82)
    ax.set_yticks(np.arange(0.5, 0.85, 0.1))
    ax.set_xlim(1951.5, 2029.5)
    ax.set_xticks(range(1960, 2030, 10))

    ax.set_title("Baltic Sea spatial Mean TTI (All, Shallow, Deep)", loc="left", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean TTI [-]", fontsize=13, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    plt.savefig("Figure_3_Baltic_TTI_deep_shallow_smoothed.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Analysis complete. Figure 3 successfully saved.")

if __name__ == "__main__":
    main()
