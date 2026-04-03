"""
Research paper visualizations for RAG pipeline evaluation.
Publication-quality figures with statistical annotations using seaborn.
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

# ── Style: standard academic grayscale-friendly palette ─────────────────────
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

# Colorblind-safe palette (Okabe-Ito)
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
# Two-color for fixed vs semantic
PAIR_PALETTE = ["#0072B2", "#D55E00"]

METRIC_LABELS = {
    "f1_score": "LLM Judge Score",
    "answer_relevancy": "Answer Relevancy",
    "faithfulness": "Faithfulness",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "token_f1": "Token F1",
}

LLM_SHORT = {
    "claude-3.5-sonnet": "Claude 3.5",
    "gemini-2.5-pro": "Gemini 2.5",
    "gpt-4o-mini": "GPT-4o Mini",
    "llama-3.1-70b": "Llama 3.1",
    "mistral-large": "Mistral Large",
}

STRATEGY_LABELS = {
    "fixed_128": "Fixed 128",
    "fixed_256": "Fixed 256",
    "fixed_512": "Fixed 512",
    "semantic_128": "Semantic 128",
    "semantic_256": "Semantic 256",
    "semantic_512": "Semantic 512",
}

METRICS = list(METRIC_LABELS.keys())


def load_summary():
    df = pd.read_csv("results/summary_statistics_judged.csv")
    df["llm_short"] = df["llm"].map(LLM_SHORT)
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS)
    df["chunk_method"] = df["strategy"].apply(lambda s: s.rsplit("_", 1)[0].title())
    df["chunk_size"] = df["strategy"].apply(lambda s: int(s.rsplit("_", 1)[1]))
    return df


def load_raw():
    df = pd.read_csv("results/full_evaluation_results_judged.csv")
    df["llm_short"] = df["llm"].map(LLM_SHORT)
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS)
    df["chunk_method"] = df["strategy"].apply(lambda s: s.rsplit("_", 1)[0].title())
    df["chunk_size"] = df["strategy"].apply(lambda s: int(s.rsplit("_", 1)[1]))
    return df


# ── Figure 1: Heatmap – LLM x Metric with SD ──────────────────────────────
def fig1_llm_metric_heatmap(raw):
    means = raw.groupby("llm_short")[METRICS].mean()
    sds = raw.groupby("llm_short")[METRICS].std()
    ns = raw.groupby("llm_short")[METRICS].count()

    means = means.rename(columns=METRIC_LABELS)
    sds = sds.rename(columns=METRIC_LABELS)
    ns = ns.rename(columns=METRIC_LABELS)

    means = means.loc[means.mean(axis=1).sort_values(ascending=False).index]
    sds = sds.loc[means.index]
    ns = ns.loc[means.index]

    # Two-line annotation: mean on top, SD below (no n to reduce clutter)
    annot = means.copy().astype(str)
    for col in means.columns:
        for idx in means.index:
            m = means.loc[idx, col]
            s = sds.loc[idx, col]
            annot.loc[idx, col] = f"{m:.3f}\n\u00b1{s:.3f}"

    fig, ax = plt.subplots(figsize=(8.5, 4))
    sns.heatmap(
        means, annot=annot, fmt="", cmap="Blues", linewidths=0.8,
        cbar_kws={"label": "Mean Score"}, ax=ax, vmin=0, vmax=1,
        annot_kws={"fontsize": 9},
    )
    n_per_llm = int(ns.iloc[0, 0])
    ax.set_title(f"Mean Metric Scores by LLM (n={n_per_llm} per LLM)", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig1_llm_metric_heatmap.png")
    plt.close(fig)
    print("Saved fig1_llm_metric_heatmap.png")


# ── Figure 2: Heatmap – Strategy x Metric with SD ─────────────────────────
def fig2_strategy_metric_heatmap(raw):
    means = raw.groupby("strategy_label")[METRICS].mean()
    sds = raw.groupby("strategy_label")[METRICS].std()
    ns = raw.groupby("strategy_label")[METRICS].count()

    means = means.rename(columns=METRIC_LABELS).sort_index()
    sds = sds.rename(columns=METRIC_LABELS).loc[means.index]
    ns = ns.rename(columns=METRIC_LABELS).loc[means.index]

    annot = means.copy().astype(str)
    for col in means.columns:
        for idx in means.index:
            m = means.loc[idx, col]
            s = sds.loc[idx, col]
            annot.loc[idx, col] = f"{m:.3f}\n\u00b1{s:.3f}"

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.heatmap(
        means, annot=annot, fmt="", cmap="OrRd", linewidths=0.8,
        cbar_kws={"label": "Mean Score"}, ax=ax, vmin=0, vmax=1,
        annot_kws={"fontsize": 9},
    )
    n_per = int(ns.iloc[0, 0])
    ax.set_title(f"Mean Metric Scores by Chunking Strategy (n={n_per} per strategy)", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig2_strategy_metric_heatmap.png")
    plt.close(fig)
    print("Saved fig2_strategy_metric_heatmap.png")


# ── Figure 3: Faithfulness by LLM and strategy with 95% CI ─────────────────
def fig3_faithfulness_by_llm_strategy(raw):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    order = ["Claude 3.5", "GPT-4o Mini", "Gemini 2.5", "Llama 3.1", "Mistral Large"]
    hue_order = sorted(STRATEGY_LABELS.values())

    sns.barplot(
        data=raw, x="llm_short", y="faithfulness", hue="strategy_label",
        palette="Greys", edgecolor="0.2", ax=ax, ci=95, capsize=0.03,
        order=order, hue_order=hue_order, errwidth=1,
    )
    ax.set_title("Faithfulness by LLM and Chunking Strategy (95% CI)", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Faithfulness")
    ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Show n per LLM in the x-tick labels themselves
    new_labels = []
    for llm in order:
        n = len(raw[raw["llm_short"] == llm])
        new_labels.append(f"{llm}\n(n={n})")
    ax.set_xticklabels(new_labels)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig3_faithfulness_llm_strategy.png")
    plt.close(fig)
    print("Saved fig3_faithfulness_llm_strategy.png")


# ── Figure 4: Key metrics by query type with 95% CI ────────────────────────
def fig4_metrics_by_query_type(raw):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    key_metrics = ["f1_score", "faithfulness", "context_recall"]
    qt_order = ["single-hop", "multi-hop", "comparative", "unanswerable"]
    llm_order = ["Claude 3.5", "GPT-4o Mini", "Gemini 2.5", "Llama 3.1", "Mistral Large"]

    # Pre-compute n per query type for axis labels
    qt_labels = {}
    for qt in qt_order:
        n = len(raw[raw["query_type"] == qt]) // len(llm_order)
        qt_labels[qt] = f"{qt}\n(n={n})"

    for ax, metric in zip(axes, key_metrics):
        sns.barplot(
            data=raw, x="query_type", y=metric, hue="llm_short",
            palette=CB_PALETTE, edgecolor="0.2", ax=ax, ci=95, capsize=0.02,
            order=qt_order, hue_order=llm_order, errwidth=1,
        )
        ax.set_title(METRIC_LABELS[metric], fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Mean Score" if ax == axes[0] else "")

        # Embed n into the tick labels instead of floating text
        ax.set_xticklabels([qt_labels[qt] for qt in qt_order], fontsize=9)

        if ax != axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="LLM", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    fig.suptitle("Key Metrics by Query Type (95% CI Error Bars)", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig4_metrics_by_query_type.png")
    plt.close(fig)
    print("Saved fig4_metrics_by_query_type.png")


# ── Figure 5: Domain comparison with 95% CI ────────────────────────────────
def fig5_domain_comparison(raw):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    key_metrics = ["f1_score", "faithfulness", "context_precision"]
    domain_order = ["general", "medical", "finance"]
    llm_order = ["Claude 3.5", "GPT-4o Mini", "Gemini 2.5", "Llama 3.1", "Mistral Large"]

    # Pre-compute n per domain for axis labels
    dom_labels = {}
    for dom in domain_order:
        n = len(raw[raw["domain"] == dom])
        dom_labels[dom] = f"{dom.title()}\n(N={n})"

    for ax, metric in zip(axes, key_metrics):
        sns.barplot(
            data=raw, x="domain", y=metric, hue="llm_short",
            palette=CB_PALETTE, edgecolor="0.2", ax=ax, ci=95, capsize=0.02,
            order=domain_order, hue_order=llm_order, errwidth=1,
        )
        ax.set_title(METRIC_LABELS[metric], fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Mean Score" if ax == axes[0] else "")

        ax.set_xticklabels([dom_labels[dom] for dom in domain_order], fontsize=9)

        if ax != axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="LLM", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    fig.suptitle("Key Metrics by Domain (95% CI Error Bars)", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig5_domain_comparison.png")
    plt.close(fig)
    print("Saved fig5_domain_comparison.png")


# ── Figure 6: Fixed vs Semantic with 95% CI + significance brackets ────────
def fig6_fixed_vs_semantic(raw):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)

    for ax, metric in zip(axes, METRICS):
        sns.barplot(
            data=raw, x="chunk_size", y=metric, hue="chunk_method",
            palette=PAIR_PALETTE, edgecolor="0.2", ax=ax, ci=95, capsize=0.04,
            order=[128, 256, 512], hue_order=["Fixed", "Semantic"], errwidth=1,
        )
        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        # Get y-axis top for placing significance text consistently
        ymin, ymax = ax.get_ylim()
        headroom = (ymax - ymin) * 0.15
        ax.set_ylim(ymin, ymax + headroom)

        # Significance annotations placed above the bars using bracket style
        for j, cs in enumerate([128, 256, 512]):
            fixed_vals = raw[(raw["chunk_method"] == "Fixed") & (raw["chunk_size"] == cs)][metric]
            sem_vals = raw[(raw["chunk_method"] == "Semantic") & (raw["chunk_size"] == cs)][metric]
            if len(fixed_vals) > 1 and len(sem_vals) > 1:
                _, p_val = stats.ttest_ind(fixed_vals, sem_vals, equal_var=False)
                pooled_std = np.sqrt((fixed_vals.std()**2 + sem_vals.std()**2) / 2)
                d = (fixed_vals.mean() - sem_vals.mean()) / pooled_std if pooled_std > 0 else 0
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

                # Place bracket above the two bars
                bar_top = max(fixed_vals.mean(), sem_vals.mean())
                bracket_y = ymax + headroom * 0.15
                bar_width = 0.4
                x_left = j - bar_width / 2
                x_right = j + bar_width / 2

                ax.plot([x_left, x_left, x_right, x_right],
                        [bracket_y - headroom * 0.05, bracket_y, bracket_y, bracket_y - headroom * 0.05],
                        color="0.3", linewidth=0.8)
                ax.text(j, bracket_y + headroom * 0.02, f"{sig}, d={d:.2f}",
                        ha="center", va="bottom", fontsize=6.5, color="0.2")

        if ax != axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Method", frameon=True, fontsize=8)

    n_fixed = len(raw[raw["chunk_method"] == "Fixed"])
    n_sem = len(raw[raw["chunk_method"] == "Semantic"])
    fig.suptitle(
        f"Fixed vs. Semantic Chunking (95% CI; n={n_fixed} each)",
        y=1.05, fontsize=13, fontweight="bold",
    )
    fig.text(0.5, -0.01, "* p<.05  ** p<.01  *** p<.001  ns = not significant  d = Cohen\u2019s d",
             ha="center", fontsize=8, color="0.4", style="italic")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig6_fixed_vs_semantic.png")
    plt.close(fig)
    print("Saved fig6_fixed_vs_semantic.png")


# ── Figure 7: Radar chart with companion table (no overlapping labels) ─────
def fig7_llm_radar(raw):
    llm_means = raw.groupby("llm_short")[METRICS].mean()
    labels = [METRIC_LABELS[m] for m in METRICS]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 6))
    # Radar on the left, table on the right
    ax_radar = fig.add_axes([0.05, 0.08, 0.52, 0.82], polar=True)
    ax_table = fig.add_axes([0.60, 0.08, 0.38, 0.82])
    ax_table.axis("off")

    llm_order = llm_means.mean(axis=1).sort_values(ascending=False).index
    llm_means = llm_means.loc[llm_order]

    for idx, (llm, row) in enumerate(llm_means.iterrows()):
        values = row.tolist() + [row.tolist()[0]]
        ax_radar.plot(angles, values, "o-", linewidth=2, label=llm,
                      color=CB_PALETTE[idx], markersize=5)
        ax_radar.fill(angles, values, alpha=0.06, color=CB_PALETTE[idx])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=9)
    ax_radar.set_ylim(0, 1.05)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="0.5")
    ax_radar.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.15), frameon=True, fontsize=8)

    # Build a table with exact values
    col_labels = [METRIC_LABELS[m] for m in METRICS]
    cell_text = []
    row_colors = []
    for idx, (llm, row) in enumerate(llm_means.iterrows()):
        cell_text.append([f"{v:.3f}" for v in row])
        row_colors.append(CB_PALETTE[idx])

    table = ax_table.table(
        cellText=cell_text,
        rowLabels=list(llm_means.index),
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)

    # Color row labels to match radar lines
    for idx, key in enumerate(table.get_celld()):
        row, col = key
        cell = table[key]
        cell.set_edgecolor("0.7")
        if col == -1 and row > 0:  # row label cells
            cell.set_text_props(color=row_colors[row - 1], fontweight="bold")
        if row == 0:  # header
            cell.set_text_props(fontweight="bold", fontsize=7.5)
            cell.set_facecolor("0.92")

    n_total = len(raw)
    fig.suptitle(f"LLM Performance Profiles (N={n_total} total observations)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.savefig(f"{OUTPUT_DIR}/fig7_llm_radar.png")
    plt.close(fig)
    print("Saved fig7_llm_radar.png")


# ── Figure 8: Context metrics by chunk size with CI + clean labels ─────────
def fig8_context_by_chunk_size(raw):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    context_metrics = ["context_precision", "context_recall"]

    for ax, metric in zip(axes, context_metrics):
        sns.lineplot(
            data=raw, x="chunk_size", y=metric, hue="chunk_method",
            style="chunk_method", markers=True, markersize=10, linewidth=2.2,
            palette=PAIR_PALETTE, ax=ax, ci=95, err_style="band",
            hue_order=["Fixed", "Semantic"],
        )
        ax.set_title(METRIC_LABELS[metric], fontweight="bold")
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Mean Score" if ax == axes[0] else "")
        ax.set_xticks([128, 256, 512])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        # Annotate only mean values, stagger vertically to avoid overlap
        offsets = {"Fixed": (10, 12), "Semantic": (10, -18)}
        for method, color in zip(["Fixed", "Semantic"], PAIR_PALETTE):
            for cs in [128, 256, 512]:
                subset = raw[(raw["chunk_method"] == method) & (raw["chunk_size"] == cs)]
                m = subset[metric].mean()
                n = len(subset)
                ax.annotate(
                    f"{m:.3f} (n={n})",
                    xy=(cs, m),
                    xytext=offsets[method],
                    textcoords="offset points",
                    ha="left", fontsize=7.5, color=color, fontweight="bold",
                )

        if ax != axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Method", frameon=True)

    fig.suptitle("Context Quality vs. Chunk Size (95% CI Bands)", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig8_context_by_chunk_size.png")
    plt.close(fig)
    print("Saved fig8_context_by_chunk_size.png")


# ── Figure 9: Strategy ranking – clean table-based approach ─────────────────
def fig9_strategy_ranking(raw):
    means = raw.groupby("strategy_label")[METRICS].mean()
    means["overall"] = means.mean(axis=1)
    means = means.sort_values("overall", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Plot dots only – no text labels on the chart to avoid overlap
    for i, metric in enumerate(METRICS):
        for j, strategy in enumerate(means.index):
            val = means.loc[strategy, metric]
            ax.scatter(val, j, color=CB_PALETTE[i], s=80, zorder=3,
                       edgecolors="0.2", linewidths=0.5,
                       label=METRIC_LABELS[metric] if j == 0 else "")

    # Dumbbell connecting lines
    for j, strategy in enumerate(means.index):
        vals = means.loc[strategy, METRICS]
        ax.plot([vals.min(), vals.max()], [j, j], color="0.75", linewidth=1, zorder=1)
        # Annotate only the min and max values at the ends
        ax.text(vals.min() - 0.02, j, f"{vals.min():.3f}", ha="right", va="center",
                fontsize=7.5, color="0.3")
        ax.text(vals.max() + 0.02, j, f"{vals.max():.3f}", ha="left", va="center",
                fontsize=7.5, color="0.3")

    ax.set_yticks(range(len(means.index)))
    ax.set_yticklabels(means.index)
    ax.set_xlabel("Score")
    ax.set_title("Metric Spread by Chunking Strategy (Range Annotated)", fontweight="bold")
    ax.set_xlim(-0.08, 1.08)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:5], labels[:5], loc="lower right", frameon=True, fontsize=8)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig9_strategy_ranking.png")
    plt.close(fig)
    print("Saved fig9_strategy_ranking.png")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    summary = load_summary()
    raw = load_raw()
    print(f"Summary: {len(summary)} rows | Raw: {len(raw)} observations")
    print(f"Strategies: {raw['strategy'].nunique()} | LLMs: {raw['llm'].nunique()}")
    print(f"Domains: {sorted(raw['domain'].unique())} | Query types: {sorted(raw['query_type'].unique())}\n")

    fig1_llm_metric_heatmap(raw)
    fig2_strategy_metric_heatmap(raw)
    fig3_faithfulness_by_llm_strategy(raw)
    fig4_metrics_by_query_type(raw)
    fig5_domain_comparison(raw)
    fig6_fixed_vs_semantic(raw)
    fig7_llm_radar(raw)
    fig8_context_by_chunk_size(raw)
    fig9_strategy_ranking(raw)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
