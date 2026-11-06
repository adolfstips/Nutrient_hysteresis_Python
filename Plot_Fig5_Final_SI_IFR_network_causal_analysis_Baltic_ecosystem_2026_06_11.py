#!/usr/bin/env python3
"""
Baltic Ecosystem Deep and Shallow Variables
Global System Impact (SI) and Multidimensional Information Flow Rate (IFR) Analysis
Figure 5 for hysteresis article
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import networkx as nx

OKABE_ITO_NET = {
    "TN_load": "#56B4E9",  # Light Blue
    "TP_load": "#E69F00",  # Orange
    "DIP":     "#009E73",  # Bluish Green
    "DIN":     "#F0E442",  # Yellow
    "CHLA":    "#D55E00",  # Vermillion
    "DO2_Mt":  "#0072B2",  # Dark Blue
    "Secchi":  "#CC79A7",  # Reddish Purple
    "TTI":     "#999999"   # Grey
}

def r_scale(df_or_arr):
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
    ymin = np.floor(np.min(gg_ad) * 10.0 - 0.4) / 10.0
    ymax = np.ceil(np.max(gg_ad) * 10.0 + 0.35) / 10.0

    # Muted Blue - White - Muted Red diverging colormap
    cmap = LinearSegmentedColormap.from_list("si_div", ["#4E6A9E", "#FFFFFF", "#9E3B33"])
    norm = TwoSlopeNorm(vmin=min(ymin, -0.05), vcenter=0.0, vmax=max(ymax, 0.05))
    colors = [cmap(norm(val)) for val in gg_ad]

    x_pos = np.arange(len(var_names))
    ax.bar(x_pos, gg_ad, width=0.92, color=colors, edgecolor="black", linewidth=0.5, zorder=2)

    for idx, val in enumerate(gg_ad):
        y_txt = val + 0.03 * (1 if val >= 0 else -1)
        ax.text(idx, y_txt, f"{val:.2g}" if val in [0.2] else f"{val:.2f}",
                ha="center", va="center", fontweight="bold", fontsize=9.5, zorder=4)

    ax.axhline(0, color="black", linewidth=1.0, zorder=3)
    ax.axhline(0.1, color="#7F7F7F", linestyle="--", linewidth=1.0, zorder=3)
    ax.axhline(-0.1, color="#7F7F7F", linestyle="--", linewidth=1.0, zorder=3)

    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(var_names, rotation=45, ha="right", fontweight="bold", fontsize=10)
    ax.set_ylabel("System Impact [Nat/year]", fontsize=12, fontweight="bold")
    ax.set_title(f"System Impact - {region_name}", fontsize=15, fontweight="bold", pad=16)
    ax.text(0.0, 1.02, f"Period: {s_date} - {e_date}", transform=ax.transAxes, fontsize=10, color="#333333")
    ax.text(-0.12, 1.08, panel_label, transform=ax.transAxes, fontsize=20, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def plot_network_panel(ax, var_names, GG1, infoflow, region_name, panel_label):
    adj = infoflow.copy()
    adj[np.isnan(adj)] = 0.0
    adj[np.abs(adj) < 0.1] = 0.0

    G = nx.DiGraph()
    for idx, name in enumerate(var_names):
        G.add_node(name, gi_abs=abs(GG1[idx]), gi_sign="Negative" if GG1[idx] < 0 else "Positive")

    M = len(var_names)
    for i in range(M):
        for j in range(M):
            if adj[i, j] != 0:
                G.add_edge(var_names[i], var_names[j], weight=adj[i, j],
                           flow_strength=abs(adj[i, j]),
                           flow_sign="Positive" if adj[i, j] > 0 else "Negative")

    pos = nx.kamada_kawai_layout(G, weight=None)

    # Scale node sizes and edge widths
    gi_vals = np.array([G.nodes[n]["gi_abs"] for n in var_names])
    g_min, g_max = gi_vals.min(), max(gi_vals.max(), gi_vals.min() + 1e-6)
    node_sizes = 350 + (gi_vals - g_min) / (g_max - g_min) * 1850

    for idx, n in enumerate(var_names):
        marker = "s" if G.nodes[n]["gi_sign"] == "Negative" else "o"
        ax.scatter(pos[n][0], pos[n][1], s=node_sizes[idx], color=OKABE_ITO_NET[n],
                   marker=marker, alpha=0.85, edgecolors="none", zorder=3)
        ax.text(pos[n][0], pos[n][1] - 0.09, n, fontsize=11, fontweight="bold", ha="center", va="top", zorder=5)

    edges = list(G.edges(data=True))
    if edges:
        strengths = np.array([d["flow_strength"] for _, _, d in edges])
        s_min, s_max = strengths.min(), max(strengths.max(), strengths.min() + 1e-6)

        for u, v, d in edges:
            norm_s = (d["flow_strength"] - s_min) / (s_max - s_min) if s_max > s_min else 0.5
            lw = 1.0 + norm_s * 4.2
            alpha = 0.40 + norm_s * 0.55
            ls = "solid" if d["flow_sign"] == "Positive" else (0, (4, 2))
            edge_col = OKABE_ITO_NET[u]

            patch = FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                connectionstyle="arc3,rad=0.18",
                arrowstyle="-|>,head_length=8,head_width=5",
                color=edge_col, linewidth=lw, linestyle=ls, alpha=alpha,
                shrinkA=18, shrinkB=18, zorder=2
            )
            ax.add_patch(patch)

    ax.set_title(f"IFR Causality Network - {region_name}", fontsize=15, fontweight="bold", pad=16)
    ax.text(-0.08, 1.08, panel_label, transform=ax.transAxes, fontsize=20, fontweight="bold")
    ax.axis("off")

    # Add compact bottom legend
    legend_elements = [
        Line2D([0], [0], color="#666666", lw=1.8, ls="--", label="Causal: Negative"),
        Line2D([0], [0], color="#666666", lw=1.8, ls="-", label="Causal: Positive"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#666666", markersize=8, label="SI: Negative"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666", markersize=8, label="SI: Positive")
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.08),
              ncol=4, frameon=False, fontsize=9.5)

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

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)
    fig.subplots_adjust(wspace=0.22, hspace=0.32)

    plot_gi_panel(axes[0, 0], clean_names, gg_ad_deep, "Deep Baltic", s_date, e_date, "A")
    plot_network_panel(axes[0, 1], clean_names, GG_deep, ifr_deep, "Deep Baltic", "B")
    plot_gi_panel(axes[1, 0], clean_names, gg_ad_shal, "Shallow Baltic", s_date, e_date, "C")
    plot_network_panel(axes[1, 1], clean_names, GG_shal, ifr_shal, "Shallow Baltic", "D")

    plt.savefig("Figure_5_Baltic_Causality_Combined.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Analysis complete. Figure 5 successfully saved.")

if __name__ == "__main__":
    main()
