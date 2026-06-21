"""
Final research question visualizations with ANOVA significance annotations.
Three features: chunking strategy (method + size), domain, query type.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy import stats
import os
import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.edgecolor": "0.2",
    "grid.color": "0.85",
    "grid.linewidth": 0.5,
})

OUTPUT_DIR = "figures"
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
METHOD_PALETTE = {"Fixed": "#0072B2", "Semantic": "#D55E00"}
DOMAIN_PALETTE = {"General": "#0072B2", "Medical": "#D55E00", "Finance": "#009E73"}
QT_PALETTE = {"Single-Hop": "#0072B2", "Multi-Hop": "#D55E00",
              "Comparative": "#009E73", "Unanswerable": "#CC79A7"}

METRIC_LABELS = {
    "f1_score": "Answer Correctness",
    "faithfulness": "Faithfulness",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}


def load_data():
    df = pd.read_csv("results/evaluation_results_judged.csv")
    df["chunk_method"] = df["strategy"].apply(lambda s: s.rsplit("_", 1)[0])
    df["chunk_method_label"] = df["chunk_method"].map({
        "fixed": "Fixed", "semantic": "Semantic"
    })
    df["chunk_size"] = df["strategy"].apply(lambda s: int(s.rsplit("_", 1)[1]))
    df["domain_label"] = df["domain"].str.title()
    df["qt_label"] = df["query_type"].str.replace("-", "-").str.title()
    return df


# ── Fig A: Eta-squared bar chart — what explains variance? ──────────────────
def figA_eta_squared(df):
    """Shows relative importance of each factor via eta-squared from ANOVA."""
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    metrics = list(METRIC_LABELS.keys())

    # Run ANOVA on general domain (has all query types)
    general = df[df["domain"] == "general"]
    results = []
    for metric in metrics:
        model = ols(f'{metric} ~ C(chunk_method) * C(chunk_size) * C(query_type)',
                    data=general).fit()
        table = anova_lm(model, typ=2)
        ss_total = table["sum_sq"].sum()

        factors = {
            "Chunk Method": "C(chunk_method)",
            "Chunk Size": "C(chunk_size)",
            "Query Type": "C(query_type)",
        }
        for label, key in factors.items():
            eta2 = table.loc[key, "sum_sq"] / ss_total * 100
            p = table.loc[key, "PR(>F)"]
            results.append({"Metric": METRIC_LABELS[metric], "Factor": label,
                            "Eta-squared (%)": eta2, "p": p})

    res_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.25
    factors = ["Query Type", "Chunk Size", "Chunk Method"]
    colors = ["#CC79A7", "#0072B2", "#D55E00"]

    for i, (factor, color) in enumerate(zip(factors, colors)):
        vals = res_df[res_df["Factor"] == factor]["Eta-squared (%)"].values
        ps = res_df[res_df["Factor"] == factor]["p"].values
        bars = ax.bar(x + i * width, vals, width, label=factor, color=color,
                      edgecolor="0.2", linewidth=0.5)
        for j, (v, p) in enumerate(zip(vals, ps)):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(x[j] + i * width, v + 0.3, f"{v:.1f}%\n{sig}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
    ax.set_ylabel("Variance Explained (Eta-squared %)")
    ax.set_title("How Much Does Each Factor Explain?\n(Three-Way ANOVA, General Domain, N=8,595)",
                 fontweight="bold")
    ax.legend(frameon=True)
    ax.set_ylim(0, max(res_df["Eta-squared (%)"]) * 1.4)

    fig.text(0.5, -0.03, "* p<.05  ** p<.01  *** p<.001  ns = not significant",
             ha="center", fontsize=9, color="0.4", style="italic")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/figA_eta_squared_factors.png")
    plt.close(fig)
    print("Saved figA_eta_squared_factors.png")


# ── Fig B: Faceted heatmap — Strategy x Query Type, per Domain ───────────────
def figB_faceted_heatmap_all_features(df):
    """The 'everything in one figure' chart: 3 heatmaps (domain) showing
    strategy x query_type for answer correctness."""

    # Only general has all query types; medical/finance only have single-hop
    # So show general as full heatmap, medical/finance as single column
    domains = ["general", "medical", "finance"]
    qt_order = ["single-hop", "multi-hop", "comparative", "unanswerable"]
    strategies = sorted(df["strategy"].unique())

    strategy_labels = {}
    for s in strategies:
        method, size = s.rsplit("_", 1)
        strategy_labels[s] = f"{method.replace('_', ' ').title()} {size}"

    fig, axes = plt.subplots(1, 3, figsize=(20, 8),
                             gridspec_kw={"width_ratios": [4, 1, 1]})

    for ax, dom in zip(axes, domains):
        sub = df[df["domain"] == dom]
        qts = sorted(sub["query_type"].unique())

        pivot = sub.pivot_table(
            values="f1_score", index="strategy", columns="query_type", aggfunc="mean"
        ).reindex(index=strategies, columns=[q for q in qt_order if q in qts])

        annot = pivot.copy().astype(str)
        for col in pivot.columns:
            for idx in pivot.index:
                v = pivot.loc[idx, col]
                annot.loc[idx, col] = f"{v:.3f}" if not pd.isna(v) else ""

        sns.heatmap(
            pivot, annot=annot, fmt="", cmap="YlGnBu", linewidths=0.8,
            cbar=ax == axes[-1],
            cbar_kws={"label": "Answer Correctness"} if ax == axes[-1] else {},
            ax=ax, vmin=0.4, vmax=1.0,
            annot_kws={"fontsize": 9},
        )
        n = len(sub)
        ax.set_title(f"{dom.title()} Domain\n(n={n})", fontweight="bold")
        ax.set_yticklabels([strategy_labels[s] for s in strategies], rotation=0,
                           fontsize=9)
        ax.set_xticklabels([q.title() for q in pivot.columns], rotation=30,
                           ha="right", fontsize=9)
        ax.set_ylabel("Chunking Strategy" if ax == axes[0] else "")
        ax.set_xlabel("")

    fig.suptitle(
        "Answer Correctness: Chunking Strategy x Query Type x Domain",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/figB_faceted_heatmap_all.png")
    plt.close(fig)
    print("Saved figB_faceted_heatmap_all.png")


# ── Fig C: Interaction plot — Chunk Size x Domain ────────────────────────────
def figC_interaction_chunksize_domain(df):
    """Interaction plots showing how chunk size effect varies by domain."""
    single = df[df["query_type"] == "single-hop"]
    metrics = ["f1_score", "faithfulness"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric in zip(axes, metrics):
        for dom, color in DOMAIN_PALETTE.items():
            sub = single[single["domain_label"] == dom]
            means = sub.groupby("chunk_size")[metric].mean()
            sems = sub.groupby("chunk_size")[metric].sem()
            ax.errorbar(means.index, means.values, yerr=1.96 * sems.values,
                        marker="o", color=color, linewidth=2.5, markersize=9,
                        capsize=5, label=f"{dom} (n={len(sub)})")

            # Annotate values
            for cs, m in means.items():
                ax.annotate(f"{m:.3f}", xy=(cs, m), xytext=(8, 8),
                            textcoords="offset points", fontsize=8, color=color)

        # Add significance bracket for 128 vs 512
        ax.set_title(METRIC_LABELS[metric], fontweight="bold")
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Mean Score")
        ax.set_xticks([128, 256, 512])
        ax.legend(frameon=True)

    fig.suptitle(
        "Chunk Size x Domain Interaction (Single-Hop Only, 95% CI)\n"
        "ANOVA: Chunk Size x Domain interaction significant for correctness (p<.001) and faithfulness (p<.001)",
        fontsize=12, fontweight="bold", y=1.05,
    )
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/figC_interaction_chunksize_domain.png")
    plt.close(fig)
    print("Saved figC_interaction_chunksize_domain.png")


# ── Fig D: Interaction plot — Chunk Size x Query Type ────────────────────────
def figD_interaction_chunksize_querytype(df):
    """Interaction plots showing how chunk size effect varies by query type."""
    general = df[df["domain"] == "general"]
    metrics = ["f1_score", "faithfulness"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric in zip(axes, metrics):
        for qt, color in QT_PALETTE.items():
            qt_val = qt.lower()
            sub = general[general["query_type"] == qt_val]
            if len(sub) == 0:
                continue
            means = sub.groupby("chunk_size")[metric].mean()
            sems = sub.groupby("chunk_size")[metric].sem()
            ax.errorbar(means.index, means.values, yerr=1.96 * sems.values,
                        marker="o", color=color, linewidth=2.5, markersize=9,
                        capsize=5, label=f"{qt} (n={len(sub)})")

            for cs, m in means.items():
                ax.annotate(f"{m:.3f}", xy=(cs, m), xytext=(8, -12),
                            textcoords="offset points", fontsize=7.5, color=color)

        ax.set_title(METRIC_LABELS[metric], fontweight="bold")
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Mean Score")
        ax.set_xticks([128, 256, 512])
        ax.legend(frameon=True, fontsize=8)

    fig.suptitle(
        "Chunk Size x Query Type Interaction (General Domain, 95% CI)\n"
        "ANOVA: Chunk Size x Query Type interaction significant for faithfulness (p<.001)",
        fontsize=12, fontweight="bold", y=1.05,
    )
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/figD_interaction_chunksize_querytype.png")
    plt.close(fig)
    print("Saved figD_interaction_chunksize_querytype.png")


# ── Fig E: Chunk Method comparison by Domain (grouped bar) ───────────────────
def figE_method_by_domain(df):
    single = df[df["query_type"] == "single-hop"]
    metrics = ["f1_score", "faithfulness", "context_recall"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=single, x="domain_label", y=metric, hue="chunk_method_label",
            palette=METHOD_PALETTE, edgecolor="0.2", ax=ax, ci=95, capsize=0.04,
            order=["General", "Medical", "Finance"],
            hue_order=["Fixed", "Semantic"], errwidth=1,
        )
        ax.set_title(METRIC_LABELS[metric], fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Mean Score" if ax == axes[0] else "")
        ax.set_ylim(0, 1.05)

        if ax != axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Chunk Method", bbox_to_anchor=(1.02, 1),
                      loc="upper left", frameon=True)

    fig.suptitle(
        "Chunking Method x Domain (Single-Hop, 95% CI)\n"
        "ANOVA: Chunk Method main effect NOT significant for correctness (p=.060)",
        fontsize=12, fontweight="bold", y=1.05,
    )
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/figE_method_by_domain.png")
    plt.close(fig)
    print("Saved figE_method_by_domain.png")


# ── Fig F: Chunk Method comparison by Query Type (grouped bar) ───────────────
def figF_method_by_querytype(df):
    general = df[df["domain"] == "general"]
    metrics = ["f1_score", "faithfulness", "context_recall"]
    qt_order = ["Single-Hop", "Multi-Hop", "Comparative", "Unanswerable"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=general, x="qt_label", y=metric, hue="chunk_method_label",
            palette=METHOD_PALETTE, edgecolor="0.2", ax=ax, ci=95, capsize=0.04,
            order=qt_order,
            hue_order=["Fixed", "Semantic"], errwidth=1,
        )
        ax.set_title(METRIC_LABELS[metric], fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Mean Score" if ax == axes[0] else "")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(qt_order, rotation=20, ha="right")

        if ax != axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Chunk Method", bbox_to_anchor=(1.02, 1),
                      loc="upper left", frameon=True)

    fig.suptitle(
        "Chunking Method x Query Type (General Domain, 95% CI)\n"
        "ANOVA: Method x Query Type interaction significant for correctness (p=.004)",
        fontsize=12, fontweight="bold", y=1.05,
    )
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/figF_method_by_querytype.png")
    plt.close(fig)
    print("Saved figF_method_by_querytype.png")


# ── Fig G: Bubble chart — Chunk Size x Correctness x Faithfulness x Domain ──
def figG_bubble_chart(df):
    single = df[df["query_type"] == "single-hop"]
    combo = single.groupby(["chunk_method_label", "chunk_size", "domain_label"]).agg(
        correctness=("f1_score", "mean"),
        faithfulness=("faithfulness", "mean"),
        ctx_recall=("context_recall", "mean"),
        n=("f1_score", "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 8))

    for dom, marker in zip(["General", "Medical", "Finance"], ["o", "s", "D"]):
        sub = combo[combo["domain_label"] == dom]
        for _, row in sub.iterrows():
            color = METHOD_PALETTE[row["chunk_method_label"]]
            size = row["chunk_size"] * 0.8  # Scale bubble by chunk size
            ax.scatter(row["correctness"], row["faithfulness"],
                       s=size, c=color, marker=marker, alpha=0.7,
                       edgecolors="0.2", linewidths=0.8)
            ax.annotate(f"{int(row['chunk_size'])}",
                        xy=(row["correctness"], row["faithfulness"]),
                        fontsize=7, ha="center", va="center", color="0.2")

    # Custom legend
    method_handles = [mpatches.Patch(color=c, label=l)
                      for l, c in METHOD_PALETTE.items()]
    domain_handles = [plt.scatter([], [], marker=m, c="gray", s=60, label=l)
                      for l, m in zip(["General", "Medical", "Finance"],
                                      ["o", "s", "D"])]
    size_handles = [plt.scatter([], [], marker="o", c="gray", s=s * 0.8, label=f"{s} tokens")
                    for s in [128, 256, 512]]

    leg1 = ax.legend(handles=method_handles, title="Chunk Method",
                     loc="lower left", frameon=True, fontsize=8)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=domain_handles, title="Domain",
                     loc="lower right", frameon=True, fontsize=8)
    ax.add_artist(leg2)

    ax.set_xlabel("Answer Correctness", fontsize=12)
    ax.set_ylabel("Faithfulness", fontsize=12)
    ax.set_title(
        "Correctness vs. Faithfulness by Chunking Config and Domain\n"
        "(Single-Hop Only; Bubble Size = Chunk Size; Number = Token Count)",
        fontweight="bold",
    )
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.3, 0.85)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/figG_bubble_correctness_faithfulness.png")
    plt.close(fig)
    print("Saved figG_bubble_correctness_faithfulness.png")


# ── Fig H: Comprehensive faceted grid — all metrics x query type x chunk size
def figH_faceted_grid(df):
    general = df[df["domain"] == "general"]
    metrics = ["f1_score", "faithfulness", "context_precision", "context_recall"]
    qt_order = ["single-hop", "multi-hop", "comparative", "unanswerable"]

    fig, axes = plt.subplots(4, 4, figsize=(20, 18), sharey="row")

    for row, metric in enumerate(metrics):
        for col, qt in enumerate(qt_order):
            ax = axes[row][col]
            sub = general[general["query_type"] == qt]

            for method, color in METHOD_PALETTE.items():
                msub = sub[sub["chunk_method_label"] == method]
                means = msub.groupby("chunk_size")[metric].mean()
                sems = msub.groupby("chunk_size")[metric].sem()
                ax.errorbar(means.index, means.values, yerr=1.96 * sems.values,
                            marker="o", color=color, linewidth=2, markersize=7,
                            capsize=3, label=method if row == 0 and col == 0 else "")

            if row == 0:
                ax.set_title(qt.title(), fontweight="bold")
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
            if row == 3:
                ax.set_xlabel("Chunk Size")
            ax.set_xticks([128, 256, 512])
            ax.set_ylim(0, 1.05)

    # Single legend at top
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True,
               fontsize=11, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        "All Metrics x Query Type x Chunk Size (General Domain)\nChunking Method Comparison with 95% CI",
        fontsize=14, fontweight="bold", y=1.06,
    )
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/figH_faceted_grid_all.png")
    plt.close(fig)
    print("Saved figH_faceted_grid_all.png")


def main():
    df = load_data()
    print(f"Loaded {len(df)} records\n")

    figA_eta_squared(df)
    figB_faceted_heatmap_all_features(df)
    figC_interaction_chunksize_domain(df)
    figD_interaction_chunksize_querytype(df)
    figE_method_by_domain(df)
    figF_method_by_querytype(df)
    figG_bubble_chart(df)
    figH_faceted_grid(df)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
