#!/usr/bin/env python3
"""
Baltic Ecosystem Deep and Shallow Variables
Global System Impact (SI) and Multidimensional Information Flow Rate (IFR) Analysis
Figure 5 for hysteresis article (Hierarchical Eutrophication Cascade Layout)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# Okabe-Ito colorblind-friendly palette (separate node fill vs edge stroke for white-bg contrast)
OKABE_ITO_FILL = {
    "TN_load": "#56B4E9",  # Light Blue
    "TP_load": "#E69F00",  # Orange
    "DIP":     "#009E73",  # Bluish Green
    "DIN":     "#F0E442",  # Yellow (outlined in dark grey on nodes)
    "CHLA":    "#D55E00",  # Vermillion
    "DO2_Mt":  "#0072B2",  # Dark Blue
    "Secchi":  "#CC79A7",  # Reddish Purple
    "TTI":     "#999999"   # Grey
}

OKABE_ITO_EDGE = {
    "TN_load": "#56B4E9",
    "TP_load": "#E69F00",
    "DIP":     "#009E73",
    "DIN":     "#C8A200",  # Deeper gold so DIN arrows are clearly legible on white
    "CHLA":    "#D55E00",
    "DO2_Mt":  "#0072B2",
    "Secchi":  "#CC79A7",
    "TTI":     "#7F7F7F"
}

# Fixed 4-tier mechanistic eutrophication cascade coordinates in [0, 1] x [0, 1]
TIER_COORDS = {
    "TN_load": (0.32, 1.00),  # Tier 1: External Forcing (N pathway)
    "TP_load": (0.68, 1.00),  # Tier 1: External Forcing (P pathway)
    "DIN":     (0.04, 0.68),  # Tier 2: Ambient Nutrients (N pathway)
    "DIP":     (0.96, 0.68),  # Tier 2: Ambient Nutrients (P pathway)
    "CHLA":    (0.00, 0.32),  # Tier 3: Pelagic Response (Biomass)
    "Secchi":  (1.00, 0.32),  # Tier 3: Pelagic Response (Clarity)
    "DO2_Mt":  (0.28, 0.00),  # Tier 4: Benthic Hypoxia & Trophic State
    "TTI":     (0.72, 0.00),  # Tier 4: Benthic Hypoxia & Trophic State
}

# Outward label offsets: (lx, ly, ha, va)
NODE_LABELS = {
    "TN_load": ( 0.28,  1.095, "center", "bottom"),
    "TP_load": ( 0.72,  1.095, "center", "bottom"),
    "DIN":     (-0.08,  0.680, "right",  "center"),
    "DIP":     ( 1.08,  0.680, "left",   "center"),
    "CHLA":    (-0.09,  0.320, "right",  "center"),
    "Secchi":  ( 1.09,  0.320, "left",   "center"),
    "DO2_Mt":  ( 0.28, -0.095, "center", "top"),
    "TTI":     ( 0.72, -0.095, "center", "top"),
}

TIER_GUIDES = [
    (1.00, "External Forcing"),
    (0.68, "Ambient Nutrients"),
    (0.32, "Pelagic Response"),
    (0.00, "Hypoxia & TTI"),
]


def r_scale(df_or_arr):
    """Replicates R's scale() using sample standard deviation (ddof=1)."""
    return (df_or_arr - np.mean(df_or_arr, axis=0)) / np.std(df_or_arr, axis=0, ddof=1)


def global_impact(xxx, np_lag=1):
    """Exact translation of Liang's Global System Impact (GI / SI) function."""
    dt = 1.0
    xxx = np.asarray(xxx, dtype=float)
    nm, M = xxx.shape
    N = nm - np_lag

    dx1 = (xxx[np_lag:nm, :] - xxx[0:N, :]) / (np_lag * dt)
    x = xxx[0:N, :].copy()

    C = np.cov(x, rowvar=False, ddof=1)
    C1 = C.copy()
    C1[:, 0] = 0.0
    C1[0, :] = 0.0
    C1[0, 0] = 1.0

    xm = x - np.mean(x, axis=0)
    dxm = dx1 - np.mean(dx1, axis=0)
    dC = (xm.T @ dxm) / (N - 1)

    invC = np.linalg.inv(C)
    invC1 = np.linalg.inv(C1)

    A = (invC @ dC).T
    AC = A @ C

    T1 = np.trace(invC1[1:M, 1:M] @ AC[1:M, 1:M]) - np.trace(A[1:M, 1:M])
    return float(T1)


def causal_error_multidim_95(xxx, np_lag=1):
    """Exact translation of Liang (2021) multidimensional Information Flow Rate (T21) and 95% CI."""
    dt = 1.0
    xxx = np.asarray(xxx, dtype=float)
    nm, M = xxx.shape
    N = nm - np_lag

    dx1_col0 = xxx[np_lag:nm, 0] - xxx[0:N, 0]
    x = xxx[0:N, :].copy()

    C = np.cov(x, rowvar=False, ddof=1)
    xm = x - np.mean(x, axis=0)
    dxm0 = dx1_col0 - np.mean(dx1_col0)

    dC_col0 = (xm.T @ dxm0) / (N - 1)
    a1n = np.linalg.solve(C, dC_col0)
    a12 = a1n[1]

    T21 = (C[0, 1] / C[0, 0]) * a12

    f1 = np.mean(dx1_col0) - np.sum(a1n * np.mean(x, axis=0))
    R1 = dx1_col0 - f1 - (x @ a1n)

    Q1 = np.sum(R1 * R1)
    b1 = np.sqrt(Q1 * dt / N)

    NI = np.zeros((M + 2, M + 2))
    NI[0, 0] = N * dt / (b1 * b1)
    NI[M + 1, M + 1] = 3.0 * dt / (b1 ** 4) * np.sum(R1 * R1) - N / (b1 * b1)

    for k in range(M):
        NI[0, k + 1] = dt / (b1 * b1) * np.sum(x[:, k])
    NI[0, M + 1] = 2.0 * dt / (b1 ** 3) * np.sum(R1)

    for k in range(M):
        for j in range(M):
            NI[j + 1, k + 1] = dt / (b1 * b1) * np.sum(x[:, j] * x[:, k])

    for k in range(M):
        NI[k + 1, M + 1] = 2.0 * dt / (b1 ** 3) * np.sum(R1 * x[:, k])

    for j in range(M + 2):
        for k in range(j):
            NI[j, k] = NI[k, j]

    invNI = np.linalg.inv(NI)
    var_a12 = invNI[2, 2]
    var_T21 = ((C[0, 1] / C[0, 0]) ** 2) * var_a12
    err95 = 1.96 * np.sqrt(max(0.0, var_T21))

    return float(T21), float(err95)


def compute_region_metrics(plot_data, var_names):
    x = np.asarray(plot_data, dtype=float)
    M = x.shape[1]

    # A. Global Impact (SI)
    GG1 = np.zeros(M)
    for k in range(M):
        indi = [k] + [idx for idx in range(M) if idx != k]
        GG1[k] = global_impact(x[:, indi], 1)
    gg_ad = np.round(GG1, 2)

    # C. Multidimensional IFR
    TT = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i != j:
                cols = [i, j] + [k for k in range(M) if k != i and k != j]
                t21, _ = causal_error_multidim_95(x[:, cols], 1)
                TT[j, i] = t21
    infoflow = np.round(TT, 2)
    return GG1, gg_ad, infoflow


def plot_gi_panel(ax, var_names, gg_ad, region_name, s_date, e_date, panel_label):
    # Shared y-axis and colour limits across Deep & Shallow for direct visual comparison
    ymin, ymax = -0.35, 0.52

    cmap = LinearSegmentedColormap.from_list("si_div", ["#4E6A9E", "#FFFFFF", "#9E3B33"])
    norm = TwoSlopeNorm(vmin=ymin, vcenter=0.0, vmax=ymax)
    colors = [cmap(norm(val)) for val in gg_ad]

    x_pos = np.arange(len(var_names))
    ax.bar(x_pos, gg_ad, width=0.85, color=colors, edgecolor="black", linewidth=0.5, zorder=2)

    for idx, val in enumerate(gg_ad):
        y_txt = val + 0.032 * (1 if val >= 0 else -1)
        ax.text(idx, y_txt, f"{val:.2g}" if val in [0.2] else f"{val:.2f}",
                ha="center", va="center", fontweight="bold", fontsize=9.5, zorder=4)

    ax.axhline(0, color="black", linewidth=1.0, zorder=3)
    ax.axhline(0.1, color="#7F7F7F", linestyle="--", linewidth=0.9, zorder=3)
    ax.axhline(-0.1, color="#7F7F7F", linestyle="--", linewidth=0.9, zorder=3)

    yticks = np.round(np.arange(-0.3, 0.55, 0.1), 1)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=10)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(var_names, rotation=45, ha="right", fontweight="bold", fontsize=10)
    ax.set_ylabel("System Impact [Nat/year]", fontsize=12, fontweight="bold")
    ax.set_title(f"System Impact - {region_name}", fontsize=16, fontweight="bold", pad=18)
    ax.text(0.0, 1.02, f"Period: {s_date} - {e_date}", transform=ax.transAxes, fontsize=10.5, color="#333333")
    ax.text(-0.16, 1.08, panel_label, transform=ax.transAxes, fontsize=22, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_network_panel(ax, var_names, GG1, infoflow, region_name, panel_label):
    adj = infoflow.copy()
    adj[np.isnan(adj)] = 0.0
    adj[np.abs(adj) < 0.1] = 0.0

    # Enforce exogenous boundary condition: external loads cannot receive causal inflows
    exog_indices = [i for i, name in enumerate(var_names) if name in ("TN_load", "TP_load")]
    adj[:, exog_indices] = 0.0

    # Subtle horizontal functional tier guide lines & left-margin labels
    for ty, tlabel in TIER_GUIDES:
        ax.plot([-0.06, 1.06], [ty, ty], color="#E5E5E5", linestyle=":", linewidth=1.0, zorder=0)
        ax.text(-0.24, ty, tlabel, fontsize=10, color="#666666", fontstyle="italic",
                ha="right", va="center", zorder=1)

    # Fixed global SI node-size scaling [0, 0.50] across Deep and Shallow
    gi_abs = np.clip(np.abs(GG1), 0.0, 0.50)
    node_sizes = 240 + (gi_abs / 0.50) * 2150

    for idx, n in enumerate(var_names):
        px, py = TIER_COORDS[n]
        marker = "s" if GG1[idx] < 0 else "o"
        ax.scatter(px, py, s=node_sizes[idx], facecolor=OKABE_ITO_FILL[n],
                   marker=marker, alpha=0.90, edgecolors="#333333", linewidths=0.9, zorder=3)
        lx, ly, ha, va = NODE_LABELS[n]
        ax.text(lx, ly, n, fontsize=11.5, fontweight="bold",
                ha=ha, va=va, color="black", zorder=5)

    # Directed causal edges with fixed global IFR scaling [0.10, 0.65]
    M = len(var_names)
    for i in range(M):
        for j in range(M):
            w = adj[i, j]
            if w != 0.0:
                u, v = var_names[i], var_names[j]
                s_abs = abs(w)
                norm_s = np.clip((s_abs - 0.10) / (0.65 - 0.10), 0.0, 1.0)
                lw = 1.1 + norm_s * 4.5
                alpha = 0.52 + norm_s * 0.43
                ls = "solid" if w > 0 else (0, (4, 2))

                # rad=-0.15 bends clockwise (right-hand rule), matching ggplot2 curvature=+0.15
                patch = FancyArrowPatch(
                    posA=TIER_COORDS[u], posB=TIER_COORDS[v],
                    connectionstyle="arc3,rad=-0.15",
                    arrowstyle="-|>,head_length=8.5,head_width=5.2",
                    color=OKABE_ITO_EDGE[u], linewidth=lw, linestyle=ls, alpha=alpha,
                    shrinkA=18, shrinkB=21, zorder=2
                )
                ax.add_patch(patch)

    ax.set_xlim(-0.46, 1.24)
    ax.set_ylim(-0.16, 1.16)
    ax.set_title(f"IFR Causality Network - {region_name}", fontsize=16, fontweight="bold", pad=18)
    ax.text(-0.04, 1.08, panel_label, transform=ax.transAxes, fontsize=22, fontweight="bold")
    ax.axis("off")

    # Compact bottom legend matching R version
    legend_elements = [
        Line2D([0], [0], color="#555555", lw=1.8, ls="--", label="Causal: Negative"),
        Line2D([0], [0], color="#555555", lw=1.8, ls="-", label="Causal: Positive"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#737373", markeredgecolor="#333333", markersize=8.5, label="SI: Negative"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#737373", markeredgecolor="#333333", markersize=8.5, label="SI: Positive"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.52, -0.06),
              ncol=4, frameon=False, fontsize=10)


def main():
    grand_df = pd.read_csv("Baltic_variables_Mt_4_Causal_Input.csv")
    s_date = int(grand_df["Year"].iloc[0])
    e_date = int(grand_df["Year"].iloc[-1])

    deep_vars = ["TN_load", "TP_load", "DIP_DEEP", "DIN_DEEP", "CHLA_DEEP", "DO2_Mt_DEEP", "Secchi_DEEP", "TTI_DEEP"]
    shallow_vars = ["TN_load", "TP_load", "DIP_SHALLOW", "DIN_SHALLOW", "CHLA_SHALLOW", "DO2_Mt_SHALLOW", "Secchi_SHALLOW", "TTI_SHALLOW"]
    clean_names = [v.replace("_DEEP", "") for v in deep_vars]

    deep_df = r_scale(grand_df[deep_vars].values)
    shallow_df = r_scale(grand_df[shallow_vars].values)

    GG_deep, gg_ad_deep, ifr_deep = compute_region_metrics(deep_df, clean_names)
    GG_shal, gg_ad_shal, ifr_shal = compute_region_metrics(shallow_df, clean_names)

    print("Deep SI:", {k: float(v) for k, v in zip(clean_names, gg_ad_deep)})
    print("Shallow SI:", {k: float(v) for k, v in zip(clean_names, gg_ad_shal)})

    fig = plt.figure(figsize=(18, 14), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.38, 0.62], wspace=0.14, hspace=0.28)

    plot_gi_panel(fig.add_subplot(gs[0, 0]), clean_names, gg_ad_deep, "Deep Baltic", s_date, e_date, "A")
    plot_network_panel(fig.add_subplot(gs[0, 1]), clean_names, GG_deep, ifr_deep, "Deep Baltic", "B")
    plot_gi_panel(fig.add_subplot(gs[1, 0]), clean_names, gg_ad_shal, "Shallow Baltic", s_date, e_date, "C")
    plot_network_panel(fig.add_subplot(gs[1, 1]), clean_names, GG_shal, ifr_shal, "Shallow Baltic", "D")

    plt.savefig("Figure_5_Baltic_Causality_Combined.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Analysis complete. Figure 5 successfully saved.")


if __name__ == "__main__":
    main()
