#!/usr/bin/env python3
"""
Baltic Ecosystem Time Lag Analysis
CCF of External TN Load vs Deep MSFD Indicators
Figure SM 1 for hysteresis article
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def r_ccf(x, y, lag_max=25):
    """Exact replica of R's stats::ccf(x, y, lag.max = lag_max, plot = FALSE)."""
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

def main():
    grand_df = pd.read_csv("Baltic_variables_Mt_4_Causal_Input.csv")
    grand_df = grand_df[(grand_df["Year"] >= 1954) & (grand_df["Year"] <= 2022)].reset_index(drop=True)

    # Order matching R ggplot2 facet_wrap alphabetical ordering
    target_vars = ["CHLA_DEEP", "DIN_DEEP", "DIP_DEEP", "DO2_Mt_DEEP", "Secchi_DEEP"]

    fig, axes = plt.subplots(len(target_vars), 1, figsize=(10, 8), dpi=300, sharex=True)
    fig.subplots_adjust(hspace=0.38, top=0.89, bottom=0.08, left=0.10, right=0.96)

    print("--- SM Figure 1: Dominant Causal Lags (TN Load vs Deep Indicators) ---")
    for ax, var in zip(axes, target_vars):
        lags, corrs = r_ccf(grand_df["TN_load"], grand_df[var], lag_max=25)
        clean_name = var.replace("_DEEP", "")

        # Restrict to Lag >= 0 for causal maximum
        causal_mask = lags >= 0
        causal_lags = lags[causal_mask]
        causal_corrs = corrs[causal_mask]
        dom_idx = np.argmax(np.abs(causal_corrs))
        dom_lag = int(causal_lags[dom_idx])
        dom_corr = causal_corrs[dom_idx]
        print(f"  {clean_name:8s}: Lag = {dom_lag:2d} yrs (r = {dom_corr:+.3f})")

        # Plot bars with causal regime colors
        colors = []
        for l in lags:
            if l < 0:
                colors.append("#D9D9D9")  # grey85
            elif l == dom_lag:
                colors.append("#D55E00")  # Dominant Okabe-Ito Vermillion
            else:
                colors.append("#999999")  # grey60

        ax.bar(lags, corrs, width=0.8, color=colors, zorder=2)
        ax.axhline(0, color="black", linewidth=0.8, zorder=3)
        ax.axhline(0.25, color="blue", linestyle="--", linewidth=1.0, alpha=0.5, zorder=3)
        ax.axhline(-0.25, color="blue", linestyle="--", linewidth=1.0, alpha=0.5, zorder=3)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)

        # Annotate dominant lag
        y_text = -np.sign(dom_corr) * 0.18
        ax.text(dom_lag, y_text, f"Lag: {dom_lag} yrs", color="#D55E00", fontweight="bold",
                fontsize=11, ha="center", va="center", zorder=4)

        # Facet strip header
        ax.set_title(clean_name, fontsize=11.5, fontweight="bold", pad=4,
                     bbox=dict(boxstyle="square,pad=0.25", facecolor="#F2F2F2", edgecolor="none"))
        ax.set_xlim(-11, 26)
        ax.set_xticks(range(-10, 26, 5))

        ymin, ymax = min(corrs.min(), -0.32), max(corrs.max(), 0.32)
        pad_y = 0.12 * (ymax - ymin)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=10.5)

    fig.text(0.10, 0.955, "Causally Constrained CCF: TN Load vs Deep Indicators", fontsize=16, fontweight="bold", ha="left")
    fig.text(0.10, 0.925, "Analysis restricted to Lag ≥ 0. Highlights strongest absolute correlation.", fontsize=11.5, color="#4D4D4D", ha="left")
    fig.text(0.02, 0.48, "Correlation Coefficient", fontsize=13, fontweight="bold", va="center", rotation="vertical")
    axes[-1].set_xlabel("Lag (Years)", fontsize=13, fontweight="bold")

    plt.savefig("Figure_SM1_Baltic_CCF_TN_HELCOM_deep.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Analysis complete. SM Figure 1 successfully saved.")

if __name__ == "__main__":
    main()
