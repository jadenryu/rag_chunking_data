"""
Research question-focused visualizations:
How can query-content chunking strategies be optimized for LLM response
performance in RAG for closed-domain question answering?

Figures focus on chunking strategy interactions with query type and domain.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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
os.makedirs(OUTPUT_DIR, exist_ok=True)

CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
PAIR_PALETTE = ["#0072B2", "#D55E00"]

METRIC_LABELS = {
    "f1_score": "Answer Correctness",
    "faithfulness": "Faithfulness",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}

KEY_METRICS = ["f1_score", "faithfulness", "context_precision", "context_recall"]


def load_data():
    df = pd.read_csv("results/evaluation_results_judged.csv")
    df["chunk_method"] = df["strategy"].apply(lambda s: s.rsplit("_", 1)[0])
    df["chunk_method_label"] = df["chunk_method"].map({
        "fixed": "Fixed", "semantic": "Semantic"
    })
    df["chunk_size"] = df["strategy"].apply(lambda s: int(s.rsplit("_", 1)[1]))
    return df


# ── Fig 10: Answer Correctness by Chunk Size x Query Type ────────────────────
def fig10_correctness_by_chunksize_querytype(df):
    qt_order = ["single-hop", "multi-hop", "comparative", "unanswerable"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    for ax, qt in zip(axes, qt_order):
        sub = df[df["query_type"] == qt]
        sns.barplot(
            data=sub, x="chunk_size", y="f1_score", hue="chunk_method_label",
            palette=PAIR_PALETTE, edgecolor="0.2", ax=ax, ci=95, capsize=0.04,
            order=[128, 256, 512], hue_order=["Fixed", "Semantic"],
            errwidth=1,
        )
        n = len(sub)
        ax.set_title(f"{qt.title()}\n(n={n})", fontweight="bold")
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Answer Correctness" if ax == axes[0] else "")
        ax.set_ylim(0, 1.05)

        if ax != axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Method", frameon=True, fontsize=8)

    fig.suptitle(
        "Answer Correctness by Chunk Size and Query Type",
        y=1.03, fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig10_correctness_chunksize_querytype.png")
    plt.close(fig)
    print("Saved fig10_correctness_chunksize_querytype.png")


# ── Fig 11: Answer Correctness by Chunk Size x Domain ────────────────────────
def fig11_correctness_by_chunksize_domain(df):
    domain_order = ["general", "medical", "finance"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, dom in zip(axes, domain_order):
        sub = df[df["domain"] == dom]
        sns.barplot(
            data=sub, x="chunk_size", y="f1_score", hue="chunk_method_label",
            palette=PAIR_PALETTE, edgecolor="0.2", ax=ax, ci=95, capsize=0.04,
            order=[128, 256, 512], hue_order=["Fixed", "Semantic"],
            errwidth=1,
        )
        n = len(sub)
        ax.set_title(f"{dom.title()} Domain\n(n={n})", fontweight="bold")
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Answer Correctness" if ax == axes[0] else "")
        ax.set_ylim(0, 1.05)

        if ax != axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Method", frameon=True, fontsize=8)

    fig.suptitle(
        "Answer Correctness by Chunk Size and Domain",
        y=1.03, fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig11_correctness_chunksize_domain.png")
    plt.close(fig)
    print("Saved fig11_correctness_chunksize_domain.png")


# ── Fig 12: Faithfulness by Chunk Size x Query Type ──────────────────────────
def fig12_faithfulness_by_chunksize_querytype(df):
    qt_order = ["single-hop", "multi-hop", "comparative", "unanswerable"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    for ax, qt in zip(axes, qt_order):
        sub = df[df["query_type"] == qt]
        sns.barplot(
            data=sub, x="chunk_size", y="faithfulness", hue="chunk_method_label",
            palette=PAIR_PALETTE, edgecolor="0.2", ax=ax, ci=95, capsize=0.04,
            order=[128, 256, 512], hue_order=["Fixed", "Semantic"],
            errwidth=1,
        )
        n = len(sub)
        ax.set_title(f"{qt.title()}\n(n={n})", fontweight="bold")
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Faithfulness" if ax == axes[0] else "")
        ax.set_ylim(0, 1.0)

        if ax != axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Method", frameon=True, fontsize=8)

    fig.suptitle(
        "Faithfulness by Chunk Size and Query Type",
        y=1.03, fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig12_faithfulness_chunksize_querytype.png")
    plt.close(fig)
    print("Saved fig12_faithfulness_chunksize_querytype.png")


# ── Fig 13: Faithfulness by Chunk Size x Domain ─────────────────────────────
def fig13_faithfulness_by_chunksize_domain(df):
    domain_order = ["general", "medical", "finance"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, dom in zip(axes, domain_order):
        sub = df[df["domain"] == dom]
        sns.barplot(
            data=sub, x="chunk_size", y="faithfulness", hue="chunk_method_label",
            palette=PAIR_PALETTE, edgecolor="0.2", ax=ax, ci=95, capsize=0.04,
            order=[128, 256, 512], hue_order=["Fixed", "Semantic"],
            errwidth=1,
        )
        n = len(sub)
        ax.set_title(f"{dom.title()} Domain\n(n={n})", fontweight="bold")
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Faithfulness" if ax == axes[0] else "")
        ax.set_ylim(0, 1.0)

        if ax != axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Method", frameon=True, fontsize=8)

    fig.suptitle(
        "Faithfulness by Chunk Size and Domain",
        y=1.03, fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig13_faithfulness_chunksize_domain.png")
    plt.close(fig)
    print("Saved fig13_faithfulness_chunksize_domain.png")


# ── Fig 14: Heatmap – Chunking Strategy x Query Type (Answer Correctness) ───
def fig14_heatmap_strategy_querytype_correctness(df):
    strategies = sorted(df["strategy"].unique())
    qt_order = ["single-hop", "multi-hop", "comparative", "unanswerable"]

    strategy_labels = {}
    for s in strategies:
        method, size = s.rsplit("_", 1)
        strategy_labels[s] = f"{method.replace('_', ' ').title()} {size}"

    pivot = df.pivot_table(
        values="f1_score", index="strategy", columns="query_type", aggfunc="mean"
    ).reindex(index=strategies, columns=qt_order)

    annot_df = pivot.copy().astype(str)
    for col in pivot.columns:
        for idx in pivot.index:
            annot_df.loc[idx, col] = f"{pivot.loc[idx, col]:.3f}"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        pivot, annot=annot_df, fmt="", cmap="YlGnBu", linewidths=0.8,
        cbar_kws={"label": "Answer Correctness"}, ax=ax, vmin=0, vmax=1,
        annot_kws={"fontsize": 10},
    )
    ax.set_yticklabels([strategy_labels[s] for s in strategies], rotation=0)
    ax.set_xticklabels([qt.title() for qt in qt_order], rotation=0)
    ax.set_title("Answer Correctness: Chunking Strategy x Query Type", fontweight="bold", pad=15)
    ax.set_ylabel("Chunking Strategy")
    ax.set_xlabel("Query Type")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig14_heatmap_strategy_querytype.png")
    plt.close(fig)
    print("Saved fig14_heatmap_strategy_querytype.png")


# ── Fig 15: Heatmap – Chunking Strategy x Domain (Answer Correctness) ───────
def fig15_heatmap_strategy_domain_correctness(df):
    strategies = sorted(df["strategy"].unique())
    domain_order = ["general", "medical", "finance"]

    strategy_labels = {}
    for s in strategies:
        method, size = s.rsplit("_", 1)
        strategy_labels[s] = f"{method.replace('_', ' ').title()} {size}"

    pivot = df.pivot_table(
        values="f1_score", index="strategy", columns="domain", aggfunc="mean"
    ).reindex(index=strategies, columns=domain_order)

    annot_df = pivot.copy().astype(str)
    for col in pivot.columns:
        for idx in pivot.index:
            annot_df.loc[idx, col] = f"{pivot.loc[idx, col]:.3f}"

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        pivot, annot=annot_df, fmt="", cmap="YlOrRd", linewidths=0.8,
        cbar_kws={"label": "Answer Correctness"}, ax=ax, vmin=0, vmax=1,
        annot_kws={"fontsize": 11},
    )
    ax.set_yticklabels([strategy_labels[s] for s in strategies], rotation=0)
    ax.set_xticklabels([d.title() for d in domain_order], rotation=0)
    ax.set_title("Answer Correctness: Chunking Strategy x Domain", fontweight="bold", pad=15)
    ax.set_ylabel("Chunking Strategy")
    ax.set_xlabel("Domain")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig15_heatmap_strategy_domain.png")
    plt.close(fig)
    print("Saved fig15_heatmap_strategy_domain.png")


# ── Fig 16: Chunk Size Effect with Significance – by Query Type ─────────────
def fig16_chunksize_effect_querytype(df):
    qt_order = ["single-hop", "multi-hop", "comparative", "unanswerable"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for ax, qt in zip(axes, qt_order):
        sub = df[df["query_type"] == qt]

        for metric, color, marker in zip(
            ["f1_score", "faithfulness", "context_precision", "context_recall"],
            CB_PALETTE[:4],
            ["o", "s", "D", "^"],
        ):
            means = sub.groupby("chunk_size")[metric].mean()
            sems = sub.groupby("chunk_size")[metric].sem()
            ax.errorbar(
                means.index, means.values, yerr=1.96 * sems.values,
                marker=marker, color=color, linewidth=2, markersize=8,
                capsize=4, label=METRIC_LABELS[metric],
            )

        # Significance test: 128 vs 512
        for metric in ["f1_score", "faithfulness"]:
            v128 = sub[sub["chunk_size"] == 128][metric]
            v512 = sub[sub["chunk_size"] == 512][metric]
            if len(v128) > 1 and len(v512) > 1:
                _, p = stats.ttest_ind(v128, v512, equal_var=False)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                label = "Correctness" if metric == "f1_score" else "Faithfulness"
                ax.text(0.98, 0.02 if metric == "f1_score" else 0.08,
                        f"128 vs 512 {label}: {sig} (p={p:.3f})",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, color="0.3", style="italic")

        n = len(sub)
        ax.set_title(f"{qt.title()} (n={n})", fontweight="bold")
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Score")
        ax.set_xticks([128, 256, 512])
        ax.set_ylim(0, 1.05)

        if ax == axes[0]:
            ax.legend(loc="lower right", frameon=True, fontsize=8)

    fig.suptitle(
        "Effect of Chunk Size on RAG Metrics by Query Type\n(95% CI Error Bars, Welch's t-test 128 vs 512)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.text(0.5, -0.01, "* p<.05  ** p<.01  *** p<.001  ns = not significant",
             ha="center", fontsize=9, color="0.4", style="italic")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig16_chunksize_effect_querytype.png")
    plt.close(fig)
    print("Saved fig16_chunksize_effect_querytype.png")


# ── Fig 17: Chunk Size Effect with Significance – by Domain ─────────────────
def fig17_chunksize_effect_domain(df):
    domain_order = ["general", "medical", "finance"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, dom in zip(axes, domain_order):
        sub = df[df["domain"] == dom]

        for metric, color, marker in zip(
            ["f1_score", "faithfulness", "context_precision", "context_recall"],
            CB_PALETTE[:4],
            ["o", "s", "D", "^"],
        ):
            means = sub.groupby("chunk_size")[metric].mean()
            sems = sub.groupby("chunk_size")[metric].sem()
            ax.errorbar(
                means.index, means.values, yerr=1.96 * sems.values,
                marker=marker, color=color, linewidth=2, markersize=8,
                capsize=4, label=METRIC_LABELS[metric],
            )

        # Significance test: 128 vs 512
        for i, metric in enumerate(["f1_score", "faithfulness"]):
            v128 = sub[sub["chunk_size"] == 128][metric]
            v512 = sub[sub["chunk_size"] == 512][metric]
            if len(v128) > 1 and len(v512) > 1:
                _, p = stats.ttest_ind(v128, v512, equal_var=False)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                label = "Correctness" if metric == "f1_score" else "Faithfulness"
                ax.text(0.98, 0.02 + i * 0.06,
                        f"128 vs 512 {label}: {sig} (p={p:.3f})",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, color="0.3", style="italic")

        n = len(sub)
        ax.set_title(f"{dom.title()} Domain (n={n})", fontweight="bold")
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Score" if ax == axes[0] else "")
        ax.set_xticks([128, 256, 512])
        ax.set_ylim(0, 1.05)

        if ax == axes[0]:
            ax.legend(loc="lower right", frameon=True, fontsize=8)

    fig.suptitle(
        "Effect of Chunk Size on RAG Metrics by Domain\n(95% CI Error Bars, Welch's t-test 128 vs 512)",
        fontsize=14, fontweight="bold", y=1.05,
    )
    fig.text(0.5, -0.01, "* p<.05  ** p<.01  *** p<.001  ns = not significant",
             ha="center", fontsize=9, color="0.4", style="italic")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig17_chunksize_effect_domain.png")
    plt.close(fig)
    print("Saved fig17_chunksize_effect_domain.png")


# ── Fig 18: Domain x Query Type Heatmap (Answer Correctness) ────────────────
def fig18_domain_querytype_heatmap(df):
    domain_order = ["general", "medical", "finance"]
    qt_order = ["single-hop", "multi-hop", "comparative", "unanswerable"]

    pivot_mean = df.pivot_table(
        values="f1_score", index="domain", columns="query_type", aggfunc="mean"
    ).reindex(index=domain_order, columns=qt_order)

    pivot_n = df.pivot_table(
        values="f1_score", index="domain", columns="query_type", aggfunc="count"
    ).reindex(index=domain_order, columns=qt_order).fillna(0).astype(int)

    annot_df = pivot_mean.copy().astype(str)
    for col in pivot_mean.columns:
        for idx in pivot_mean.index:
            m = pivot_mean.loc[idx, col]
            n = pivot_n.loc[idx, col]
            if pd.isna(m):
                annot_df.loc[idx, col] = "N/A"
            else:
                annot_df.loc[idx, col] = f"{m:.3f}\n(n={n})"

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot_mean, annot=annot_df, fmt="", cmap="RdYlGn", linewidths=1,
        cbar_kws={"label": "Answer Correctness"}, ax=ax, vmin=0, vmax=1,
        annot_kws={"fontsize": 11},
    )
    ax.set_yticklabels([d.title() for d in domain_order], rotation=0)
    ax.set_xticklabels([qt.title() for qt in qt_order], rotation=0)
    ax.set_title("Answer Correctness: Domain x Query Type\n(Averaged Across All Strategies and LLMs)",
                 fontweight="bold", pad=15)
    ax.set_ylabel("Domain")
    ax.set_xlabel("Query Type")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig18_domain_querytype_heatmap.png")
    plt.close(fig)
    print("Saved fig18_domain_querytype_heatmap.png")


# ── Fig 19: Optimal Config Summary – grouped bar ────────────────────────────
def fig19_optimal_config(df):
    combo = df.groupby(["chunk_method_label", "chunk_size"])[KEY_METRICS].mean().reset_index()
    combo["config"] = combo["chunk_method_label"] + " " + combo["chunk_size"].astype(str)

    config_order = [
        "Fixed 128", "Fixed 256", "Fixed 512",
        "Semantic 128", "Semantic 256", "Semantic 512",
        "LC Semantic 128", "LC Semantic 256", "LC Semantic 512",
    ]
    combo = combo.set_index("config").reindex(config_order)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(config_order))
    width = 0.18

    for i, metric in enumerate(KEY_METRICS):
        vals = combo[metric].values
        ax.bar(x + i * width, vals, width, label=METRIC_LABELS[metric],
               color=CB_PALETTE[i], edgecolor="0.2", linewidth=0.5)
        # Annotate correctness values on top
        if metric == "f1_score":
            for j, v in enumerate(vals):
                ax.text(x[j] + i * width, v + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(config_order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("All Metrics by Chunking Configuration\n(Averaged Across LLMs, Domains, and Query Types)",
                 fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig19_optimal_config.png")
    plt.close(fig)
    print("Saved fig19_optimal_config.png")


def main():
    df = load_data()
    print(f"Loaded {len(df)} records\n")

    fig10_correctness_by_chunksize_querytype(df)
    fig11_correctness_by_chunksize_domain(df)
    fig12_faithfulness_by_chunksize_querytype(df)
    fig13_faithfulness_by_chunksize_domain(df)
    fig14_heatmap_strategy_querytype_correctness(df)
    fig15_heatmap_strategy_domain_correctness(df)
    fig16_chunksize_effect_querytype(df)
    fig17_chunksize_effect_domain(df)
    fig18_domain_querytype_heatmap(df)
    fig19_optimal_config(df)

    print(f"\nAll RQ figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
