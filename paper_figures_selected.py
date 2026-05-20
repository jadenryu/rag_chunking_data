"""
Generates the 4 paper-selected figures in a unified academic style.
Output: figures/paper_selected/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE   = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66"]
GREY      = "#444444"
LIGHT     = "#F5F5F5"

plt.rcParams.update({
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "font.family":       "serif",
    "axes.titlesize":    16,
    "axes.labelsize":    13,
    "xtick.labelsize":   14,
    "ytick.labelsize":   12,
    "legend.fontsize":   11,
    "axes.edgecolor":    "0.3",
    "axes.linewidth":    0.8,
    "grid.color":        "0.88",
    "grid.linewidth":    0.6,
    "axes.axisbelow":    True,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    LIGHT,
})

OUT = "figures/paper_selected"

METRICS = ["f1_score", "faithfulness", "context_precision"]
METRIC_LABELS = {
    "f1_score":          "LLM Judge Score",
    "answer_relevancy":  "Answer Relevancy",
    "faithfulness":      "Faithfulness",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
}

BALANCED = {"hotpotqa","squad2","pubmedqa","pubmedqa_artificial","financeqa","financebench"}

def _bar_labels(ax, means, cis, fontsize=10):
    """Add value labels inside bar top when close to axis limit, else above error cap."""
    ymax = ax.get_ylim()[1]
    for i, (m, ci) in enumerate(zip(means, cis)):
        if m is None or np.isnan(m):
            continue
        if m + ci + 0.02 >= ymax - 0.02:
            ax.text(i, m - 0.04, f"{m:.2f}", ha="center", va="top",
                    fontsize=fontsize, fontfamily="serif", color="white", fontweight="bold")
        else:
            ax.text(i, m + ci + 0.02, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=fontsize, fontfamily="serif", color=GREY, fontweight="bold")


def load():
    df = pd.read_csv("results/evaluation_results_judged.csv")
    df = df.rename(columns={"strategy": "chunking_strategy"})
    df = df[df["dataset"].isin(BALANCED)]
    return df

df = load()

# ── Fig 1: Performance by Chunking Strategy ───────────────────────────────────
def fig_strategy():
    strategies = ["fixed_128","fixed_256","fixed_512","semantic_128","semantic_256","semantic_512"]
    n_metrics  = len(METRICS)
    fig, axes  = plt.subplots(1, n_metrics, figsize=(5.5 * n_metrics, 6), sharey=False)

    for ax, metric in zip(axes, METRICS):
        g      = df.groupby("chunking_strategy")[metric]
        means  = g.mean().reindex(strategies)
        cis    = g.sem().reindex(strategies) * 1.96
        x      = np.arange(len(strategies))
        bars   = ax.bar(x, means, yerr=cis, capsize=4, color=PALETTE[0],
                        alpha=0.85, width=0.6, error_kw={"linewidth": 1.2, "ecolor": GREY})
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.set_title(METRIC_LABELS[metric], pad=8)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.set_ylabel("Mean Score" if ax == axes[0] else "")
        ax.grid(axis="y")
        _bar_labels(ax, means.values, cis.values, fontsize=9)

    fig.suptitle("Performance by Chunking Strategy", fontsize=20, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_strategy.png")
    plt.close(fig)
    print("Saved fig1_strategy.png")


# ── Fig 2: Strategy × Domain Heatmap (F1 Score) ───────────────────────────────
def fig_heatmap():
    strategies = ["fixed_128","fixed_256","fixed_512","semantic_128","semantic_256","semantic_512"]
    domains    = ["finance","general","medical"]
    metric     = "f1_score"

    pivot = (df.groupby(["chunking_strategy","domain"])[metric]
               .mean()
               .unstack("domain")
               .reindex(index=strategies, columns=domains))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([d.capitalize() for d in domains], fontsize=14)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=11)
    ax.set_xlabel("Domain", labelpad=8)
    ax.set_ylabel("Chunking Strategy", labelpad=8)
    ax.set_title("LLM Judge Score: Chunking Strategy × Domain", fontsize=14,
                 fontweight="bold", pad=10)

    for i in range(len(strategies)):
        for j in range(len(domains)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.75 else GREY
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label("Mean Score", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_heatmap_strategy_domain.png")
    plt.close(fig)
    print("Saved fig2_heatmap_strategy_domain.png")


# ── Fig 3: Key Metrics by Domain ─────────────────────────────────────────────
def fig_domain():
    domain_order  = ["general","medical","finance"]
    domain_labels = {"general": "General", "medical": "Medical", "finance": "Finance"}
    plot_metrics  = ["f1_score","faithfulness","context_precision"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)

    for ax, metric in zip(axes, plot_metrics):
        x = np.arange(len(domain_order))
        means, cis = [], []
        for dom in domain_order:
            sub = df[df["domain"]==dom][metric].dropna()
            means.append(sub.mean() if len(sub) else np.nan)
            cis.append(sub.sem() * 1.96 if len(sub) > 1 else 0)
        ax.bar(x, means, yerr=cis, capsize=5, color=PALETTE[0], alpha=0.85,
               width=0.5, error_kw={"linewidth": 1.2, "ecolor": GREY})
        ax.set_xticks(x)
        ax.set_xticklabels([domain_labels[d] for d in domain_order])
        ax.set_title(METRIC_LABELS[metric], pad=8)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.set_ylabel("Mean Score" if ax == axes[0] else "")
        ax.grid(axis="y")
        _bar_labels(ax, means, cis)

    fig.suptitle("Key Metrics by Domain (95% CI Error Bars)", fontsize=20,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_domain.png")
    plt.close(fig)
    print("Saved fig3_domain.png")


# ── Fig 4: Key Metrics by Query Type ─────────────────────────────────────────
def fig_querytype():
    qt_order  = ["single-hop","multi-hop","comparative","unanswerable"]
    qt_labels = {"single-hop": "Single-hop", "multi-hop": "Multi-hop",
                 "comparative": "Comparative", "unanswerable": "Unanswerable"}
    plot_metrics = ["f1_score","faithfulness","context_precision"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)

    for ax, metric in zip(axes, plot_metrics):
        x = np.arange(len(qt_order))
        means, cis = [], []
        for qt in qt_order:
            sub = df[df["query_type"]==qt][metric].dropna()
            means.append(sub.mean() if len(sub) else np.nan)
            cis.append(sub.sem() * 1.96 if len(sub) > 1 else 0)
        ax.bar(x, means, yerr=cis, capsize=5, color=PALETTE[0], alpha=0.85,
               width=0.5, error_kw={"linewidth": 1.2, "ecolor": GREY})
        ax.set_xticks(x)
        ax.set_xticklabels([qt_labels[qt] for qt in qt_order], rotation=20, ha="right")
        ax.set_title(METRIC_LABELS[metric], pad=8)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.set_ylabel("Mean Score" if ax == axes[0] else "")
        ax.grid(axis="y")
        _bar_labels(ax, means, cis)

    fig.suptitle("Key Metrics by Query Type (95% CI Error Bars)", fontsize=20,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_querytype.png")
    plt.close(fig)
    print("Saved fig4_querytype.png")


if __name__ == "__main__":
    fig_strategy()
    fig_heatmap()
    fig_domain()
    fig_querytype()
    print("\nAll 4 figures saved to figures/paper_selected/")
