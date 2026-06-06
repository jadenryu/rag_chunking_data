"""
Regenerate fig2_domain_comparison.png and fig3_querytype_comparison.png
with wider, more horizontal proportions for paper layout.
"""

import os
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = "figures/paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.edgecolor": "0.2",
    "grid.color": "0.85",
    "grid.linewidth": 0.5,
})

CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]

LLM_SHORT = {
    "claude-3.5-sonnet": "Claude 3.5",
    "gpt-4o-mini": "GPT-4o Mini",
    "gemini-2.5-pro": "Gemini 2.5",
    "llama-3.1-70b": "Llama 3.1",
    "mistral-large": "Mistral Large",
}
LLM_ORDER = ["Claude 3.5", "GPT-4o Mini", "Gemini 2.5", "Llama 3.1", "Mistral Large"]


def load_data():
    df = pd.read_csv("results/full_evaluation_results_judged.csv")
    df["llm_short"] = df["llm"].map(LLM_SHORT)
    return df


def fig2_domain_comparison(df):
    """3-panel grouped bar chart: LLM Judge Score, Context Precision, Context Recall by domain."""
    metrics = ["f1_score", "context_precision", "context_recall"]
    metric_titles = ["LLM Judge Score", "Context Precision", "Context Recall"]
    domain_order = ["general", "medical", "finance"]
    domain_labels = ["General", "Medical", "Finance"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.0), sharey=False)

    for i, (ax, metric, title) in enumerate(zip(axes, metrics, metric_titles)):
        sns.barplot(
            data=df,
            x="domain", y=metric, hue="llm_short",
            palette=CB_PALETTE,
            edgecolor="0.2",
            ax=ax,
            errorbar=("ci", 95),
            capsize=0.03,
            order=domain_order,
            hue_order=LLM_ORDER,
            err_kws={"linewidth": 1},
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Mean Score" if i == 0 else "")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(domain_labels)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

        if i < len(axes) - 1:
            ax.get_legend().remove()
        else:
            leg = ax.legend(
                title="LLM",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=True,
                framealpha=0.9,
                edgecolor="0.7",
            )
            leg.get_title().set_fontweight("bold")

    fig.suptitle(
        "Key Metrics by Domain (95% CI error bars)",
        y=1.02, fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/fig2_domain_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def fig3_querytype_comparison(df):
    """Single grouped bar chart: LLM Judge Score by query type."""
    qt_order = ["single-hop", "multi-hop", "comparative", "unanswerable"]
    qt_labels = ["Single-Hop", "Multi-Hop", "Comparative", "Unanswerable"]

    fig, ax = plt.subplots(figsize=(14, 4.0))

    sns.barplot(
        data=df,
        x="query_type", y="f1_score", hue="llm_short",
        palette=CB_PALETTE,
        edgecolor="0.2",
        ax=ax,
        errorbar=("ci", 95),
        capsize=0.03,
        order=qt_order,
        hue_order=LLM_ORDER,
        err_kws={"linewidth": 1},
    )

    ax.set_xlabel("")
    ax.set_ylabel("Mean LLM Judge Score (0–1)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(qt_labels)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    leg = ax.legend(
        title="LLM",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="0.7",
    )
    leg.get_title().set_fontweight("bold")

    fig.suptitle(
        "LLM Judge Score by Query Type (95% CI error bars)",
        y=1.02, fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/fig3_querytype_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    df = load_data()
    fig2_domain_comparison(df)
    fig3_querytype_comparison(df)
    print("Done.")
