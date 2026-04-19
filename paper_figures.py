"""
Unified publication-quality figures for paper_figures/.

Data scope: Fixed chunking (128/256/512) + LC Semantic chunking (128/256/512) only.
Primary metric: f1_score (LLM Judge Score, LLM-as-judge).
Context precision/recall retained where they explain the retrieval mechanism.

Style: clean white background, Okabe-Ito palette, consistent fonts, 300 DPI.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Output directory ─────────────────────────────────────────────────────────
OUT = "figures/paper_figures"
os.makedirs(OUT, exist_ok=True)

# ── Unified academic style ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.titlepad": 8,
    "axes.labelsize": 10,
    "axes.labelweight": "normal",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 9,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.75,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#ebebeb",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ── Color palettes (Okabe-Ito, colorblind-safe) ───────────────────────────────
OI = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "pink":   "#CC79A7",
    "red":    "#D55E00",
    "sky":    "#56B4E9",
}

METHOD_PAL = {"Fixed": OI["blue"], "LC Semantic": OI["orange"]}
METHOD_ORDER = ["Fixed", "LC Semantic"]

DOMAIN_PAL   = {"General": OI["blue"], "Medical": OI["orange"], "Finance": OI["red"]}
DOMAIN_ORDER = ["general", "medical", "finance"]

QT_PAL = {
    "Single-Hop":   OI["blue"],
    "Multi-Hop":    OI["red"],
    "Comparative":  OI["green"],
    "Unanswerable": OI["pink"],
}
QT_ORDER = ["single-hop", "multi-hop", "comparative", "unanswerable"]

LLM_PAL = {
    "Claude 3.5":    OI["blue"],
    "GPT-4o Mini":   OI["red"],
    "Gemini 2.5":    OI["green"],
    "Llama 3.1":     OI["pink"],
    "Mistral Large": OI["orange"],
}
LLM_ORDER = ["Claude 3.5", "GPT-4o Mini", "Gemini 2.5", "Llama 3.1", "Mistral Large"]

FACTOR_PAL = {
    "Chunk Size":   OI["orange"],
    "Domain":       OI["blue"],
    "Query Type":   OI["green"],
    "Chunk Method": OI["red"],
}

METRIC_LABELS = {
    "f1_score":          "LLM Judge Score",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
    "faithfulness":      "Faithfulness",
}

# ── Data loaders ─────────────────────────────────────────────────────────────
def _enrich(df):
    df = df.copy()
    df["chunk_method"] = df["strategy"].apply(lambda s: s.rsplit("_", 1)[0])
    df["chunk_method_label"] = df["chunk_method"].map(
        {"fixed": "Fixed", "lc_semantic": "LC Semantic"}
    )
    df["chunk_size"] = df["strategy"].apply(lambda s: int(s.rsplit("_", 1)[1]))
    df["domain_label"] = df["domain"].str.title()
    df["qt_label"] = df["query_type"].str.title()
    df["llm_short"] = df["llm"].map({
        "claude-3.5-sonnet": "Claude 3.5",
        "gemini-2.5-pro":    "Gemini 2.5",
        "gpt-4o-mini":       "GPT-4o Mini",
        "llama-3.1-70b":     "Llama 3.1",
        "mistral-large":     "Mistral Large",
    })
    return df


def load_data():
    """Fixed (128/256/512) + LC Semantic (128/256/512) with proper RAGAS metrics."""
    df = pd.read_csv("results/ragas_evaluation.csv")
    df = _enrich(df)
    # Cast all non-numeric columns to plain object dtype (patsy/statsmodels compatibility)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(object)
    assert set(df["chunk_method"].unique()) == {"fixed", "lc_semantic"}, \
        "Unexpected strategies in dataset"
    return df


# ── Shared helpers ────────────────────────────────────────────────────────────
def _sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _cohens_d(a, b):
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1-1)*a.std()**2 + (n2-1)*b.std()**2) / (n1+n2-2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


def _save(fig, name):
    fig.savefig(f"{OUT}/{name}")
    plt.close(fig)
    print(f"  Saved {name}")


def _grouped_barplot(ax, data, x, y, hue, palette, order, hue_order, capsize=0.05):
    """Matplotlib grouped bar chart replacing sns.barplot (seaborn 0.13 broken)."""
    n_groups = len(order)
    n_hues   = len(hue_order)
    width    = 0.75 / n_hues
    x_pos    = np.arange(n_groups)

    for i, hue_val in enumerate(hue_order):
        means, cis = [], []
        for x_val in order:
            vals = data[(data[x] == x_val) & (data[hue] == hue_val)][y].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            cis.append(1.96 * vals.sem() if len(vals) > 1 else 0.0)
        offset = (i - n_hues / 2 + 0.5) * width
        ax.bar(x_pos + offset, means, width,
               label=hue_val, color=palette[hue_val],
               alpha=0.92, edgecolor="none")
        ax.errorbar(x_pos + offset, means, yerr=cis,
                    fmt="none", color="#444444", capsize=capsize * 60,
                    linewidth=1.0)

    ax.set_xticks(x_pos)
    ax.legend()  # create legend so callers can remove it if needed


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Eta-squared: which factors explain LLM Judge Score variance?
# ANOVA factors: Chunk Size, Domain, Query Type  (main effects, full dataset)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_eta_squared():
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    df = load_data()
    metrics = ["f1_score", "faithfulness", "context_precision", "context_recall"]
    factor_map = {
        "Chunk Size":  "C(chunk_size)",
        "Domain":      "C(domain)",
        "Query Type":  "C(query_type)",
    }
    rows = []
    for metric in metrics:
        sub = df.dropna(subset=[metric])
        model = ols(
            f"{metric} ~ C(chunk_size) + C(domain) + C(query_type)",
            data=sub,
        ).fit()
        tbl = anova_lm(model, typ=2)
        ss_total = tbl["sum_sq"].sum()
        for label, key in factor_map.items():
            rows.append({
                "Metric": METRIC_LABELS[metric],
                "Factor": label,
                "eta2":   tbl.loc[key, "sum_sq"] / ss_total * 100,
                "p":      tbl.loc[key, "PR(>F)"],
            })

    res  = pd.DataFrame(rows)
    mnames = [METRIC_LABELS[m] for m in metrics]
    x, w  = np.arange(len(mnames)), 0.24

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, factor in enumerate(factor_map):
        sub  = res[res["Factor"] == factor]
        vals = sub["eta2"].values
        ps   = sub["p"].values
        ax.bar(x + (i-1)*w, vals, w, label=factor,
               color=FACTOR_PAL[factor], edgecolor="#ffffff",
               linewidth=0.5, alpha=0.92)
        for j, (v, p) in enumerate(zip(vals, ps)):
            ax.text(x[j]+(i-1)*w, v+0.15, f"{v:.1f}%\n{_sig(p)}",
                    ha="center", va="bottom", fontsize=7.5, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(mnames)
    ax.set_ylabel("Variance Explained (η², %)")
    ax.set_ylim(0, float(np.nanmax(res["eta2"].values)) * 1.45)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(frameon=True, loc="upper left", ncol=3)
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    _save(fig, "fig1_eta_squared.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — LLM Judge Score by domain × LLM (shared y-axis)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_domain_comparison():
    df = load_data()
    dom_labels = {"general": "General", "medical": "Medical", "finance": "Finance"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    metrics = ["f1_score", "context_precision", "context_recall"]

    for ax, metric in zip(axes, metrics):
        _grouped_barplot(ax, df, x="domain", y=metric, hue="llm_short",
                         palette=LLM_PAL, order=DOMAIN_ORDER, hue_order=LLM_ORDER)
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticklabels([dom_labels[d] for d in DOMAIN_ORDER])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        if ax != axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="LLM", bbox_to_anchor=(1.02, 1),
                      loc="upper left", frameon=True)
    plt.tight_layout()
    _save(fig, "fig2_domain_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — LLM Judge Score by query type × LLM
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_querytype_comparison():
    df = load_data()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _grouped_barplot(ax, df, x="query_type", y="f1_score", hue="llm_short",
                     palette=LLM_PAL, order=QT_ORDER, hue_order=LLM_ORDER)
    ax.set_xlabel("")
    ax.set_ylabel("Mean LLM Judge Score (0–1)")
    ax.set_xticklabels([q.title() for q in QT_ORDER], rotation=0, ha="center")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(title="LLM", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    _save(fig, "fig3_querytype_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3b — All four metrics by query type (grouped bars, shared x-axis)
# ═══════════════════════════════════════════════════════════════════════════════
def fig3b_metrics_by_querytype():
    df = load_data()
    metrics = ["f1_score", "context_precision", "context_recall"]
    colors  = [QT_PAL[qt.title()] for qt in QT_ORDER]
    xtick_labels = [qt.title() for qt in QT_ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)

    for ax, metric in zip(axes, metrics):
        means, cis = [], []
        for qt in QT_ORDER:
            vals = df[df["query_type"] == qt][metric].dropna()
            means.append(vals.mean())
            cis.append(1.96 * vals.sem())
        x = np.arange(len(QT_ORDER))
        ax.bar(x, means, color=colors, edgecolor="none", width=0.55, alpha=0.92)
        ax.errorbar(x, means, yerr=cis, fmt="none", color="#333333",
                    capsize=4, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=0, ha="center")
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    _save(fig, "fig3b_metrics_by_querytype.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Fixed vs. LC Semantic: LLM Judge Score + context metrics × chunk size
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_fixed_vs_lc_semantic():
    df    = load_data()
    sizes = [128, 256, 512]
    metric = "f1_score"

    fig, ax = plt.subplots(figsize=(6, 4.2))

    _grouped_barplot(ax, df, x="chunk_size", y=metric, hue="chunk_method_label",
                     palette=METHOD_PAL, order=sizes, hue_order=METHOD_ORDER)
    ax.set_xlabel("Chunk Size (tokens)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Significance brackets
    ymin, ymax = ax.get_ylim()
    head = (ymax - ymin) * 0.14
    ax.set_ylim(ymin, ymax + head * 1.6)
    for j, cs in enumerate(sizes):
        a = df[(df["chunk_method"] == "fixed")       & (df["chunk_size"] == cs)][metric].dropna()
        b = df[(df["chunk_method"] == "lc_semantic") & (df["chunk_size"] == cs)][metric].dropna()
        if len(a) > 1 and len(b) > 1:
            _, p = stats.ttest_ind(a, b, equal_var=False)
            bx_y = ymax + head * 0.5
            ax.plot([j-0.2, j-0.2, j+0.2, j+0.2],
                    [bx_y-head*0.1, bx_y, bx_y, bx_y-head*0.1],
                    color="#555555", linewidth=0.75)
            ax.text(j, bx_y+head*0.05, _sig(p),
                    ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.legend(title="Method", frameon=True, fontsize=8, title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    _save(fig, "fig4_fixed_vs_lc_semantic.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4b — LLM Judge Score by query type × LLM (n-annotated x-axis)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_metrics_by_query_type():
    df = load_data()
    qt_ns = {qt: len(df[df["query_type"] == qt]) // len(LLM_ORDER) for qt in QT_ORDER}

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _grouped_barplot(ax, df, x="query_type", y="f1_score", hue="llm_short",
                     palette=LLM_PAL, order=QT_ORDER, hue_order=LLM_ORDER)
    ax.set_xlabel("")
    ax.set_ylabel("Mean LLM Judge Score (0–1)")
    ax.set_xticklabels(
        [f"{q.title()}\n(n≈{qt_ns[q]})" for q in QT_ORDER],
        fontsize=8.5,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(title="LLM", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    _save(fig, "fig4_metrics_by_query_type.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Context quality vs. chunk size: Fixed vs. LC Semantic (line + CI band)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_chunksize_context_quality():
    df    = load_data()
    sizes = [128, 256, 512]
    mk    = {"Fixed": "o", "LC Semantic": "s"}
    ls    = {"Fixed": "-", "LC Semantic": "--"}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=False)

    for ax, metric in zip(axes, ["context_precision", "context_recall"]):
        for method in METHOD_ORDER:
            key   = "fixed" if method == "Fixed" else "lc_semantic"
            sub   = df[df["chunk_method"] == key]
            means = sub.groupby("chunk_size")[metric].mean()
            sems  = sub.groupby("chunk_size")[metric].sem()
            y     = np.array([means[s] for s in sizes])
            e     = np.array([1.96 * sems[s] for s in sizes])
            color = METHOD_PAL[method]
            ax.fill_between(sizes, y-e, y+e, color=color, alpha=0.13)
            ax.errorbar(sizes, y, yerr=e, marker=mk[method], linestyle=ls[method],
                        color=color, linewidth=2.0, markersize=7, capsize=4, label=method)
            for i, s in enumerate(sizes):
                ax.annotate(f"{y[i]:.3f}", xy=(s, y[i]),
                            xytext=(0, 9 if method == "Fixed" else -16),
                            textcoords="offset points",
                            ha="center", fontsize=7.5, color=color)
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticks(sizes)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.legend(frameon=True)
    plt.tight_layout()
    _save(fig, "fig5_chunksize_context_quality.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5b — LLM Judge Score by domain × LLM (single-metric focus)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_domain_comparison():
    df = load_data()
    dom_labels = {"general": "General", "medical": "Medical", "finance": "Finance"}
    dom_ns     = {d: len(df[df["domain"] == d]) for d in DOMAIN_ORDER}

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _grouped_barplot(ax, df, x="domain", y="f1_score", hue="llm_short",
                     palette=LLM_PAL, order=DOMAIN_ORDER, hue_order=LLM_ORDER)
    ax.set_xlabel("")
    ax.set_ylabel("Mean LLM Judge Score (0–1)")
    ax.set_xticklabels(
        [f"{dom_labels[d]}\n(N={dom_ns[d]:,})" for d in DOMAIN_ORDER],
        fontsize=8.5,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(title="LLM", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    _save(fig, "fig5_domain_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — LLM Judge Score: chunk size × domain interaction (line plot)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_chunksize_domain_interaction():
    df    = load_data()
    sizes = [128, 256, 512]
    ls    = {"General": "-", "Medical": "-", "Finance": "-"}
    # Finance labels go below the line (line rises steeply, upward offset overlaps)
    # Medical labels go below; General labels go above
    y_offset = {"General": 8, "Medical": -16, "Finance": -16}

    fig, ax = plt.subplots(figsize=(7, 4.4))

    for dom in DOMAIN_ORDER:
        label = dom.title()
        sub   = df[df["domain"] == dom]
        n     = len(sub)
        means = sub.groupby("chunk_size")["f1_score"].mean()
        sems  = sub.groupby("chunk_size")["f1_score"].sem()
        y     = np.array([means[s] for s in sizes])
        e     = np.array([1.96 * sems[s] for s in sizes])
        ax.errorbar(sizes, y, yerr=e,
                    marker=None, linestyle=ls[label],
                    color=DOMAIN_PAL[label],
                    linewidth=2.0, capsize=4,
                    label=f"{label} (N={n:,})")
        for i, s in enumerate(sizes):
            ax.annotate(f"{y[i]:.3f}", xy=(s, y[i]),
                        xytext=(7, y_offset[label]), textcoords="offset points",
                        fontsize=7.5, color=DOMAIN_PAL[label])
    ax.set_xlabel("Chunk Size (tokens)")
    ax.set_ylabel("Mean LLM Judge Score (0–1)")
    ax.set_xticks(sizes)
    ax.set_ylim(0.48, 1.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(frameon=True)
    plt.tight_layout()
    _save(fig, "fig6_chunksize_domain_interaction.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6q — LLM Judge Score: chunk size × query type interaction (line plot)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_chunksize_querytype_interaction():
    df    = load_data()
    sizes = [128, 256, 512]

    fig, ax = plt.subplots(figsize=(7, 4.4))

    # First pass: plot all lines, store y-values per query type
    line_y = {}
    for qt in QT_ORDER:
        label = qt.title()
        sub   = df[df["query_type"] == qt]
        n     = len(sub)
        means = sub.groupby("chunk_size")["f1_score"].mean()
        sems  = sub.groupby("chunk_size")["f1_score"].sem()
        y     = np.array([means[s] for s in sizes])
        e     = np.array([1.96 * sems[s] for s in sizes])
        ax.errorbar(sizes, y, yerr=e,
                    marker=None, linestyle="-",
                    color=QT_PAL[label],
                    linewidth=2.0, capsize=4,
                    label=f"{label} (N={n:,})")
        line_y[label] = y

    # Second pass: annotate each chunk-size position using rank-based offsets.
    # At each x, sort lines by value and alternate above/below so closely-spaced
    # lines (e.g. Single-Hop ≈ Comparative at 256/512) never share the same side.
    # Offsets in points: rank 0 → +14 above, rank 1 → -16 below,
    #                    rank 2 → +10 above, rank 3 → -22 below
    rank_offsets = [+14, -16, +10, -22]
    for i, s in enumerate(sizes):
        ranked = sorted(
            [(lbl, line_y[lbl][i]) for lbl in line_y],
            key=lambda x: x[1], reverse=True,
        )
        for rank, (lbl, yval) in enumerate(ranked):
            yoff = rank_offsets[rank]
            ax.annotate(
                f"{yval:.3f}",
                xy=(s, yval),
                xytext=(5, yoff),
                textcoords="offset points",
                fontsize=7.5,
                color=QT_PAL[lbl],
                ha="left",
            )

    ax.set_xlabel("Chunk Size (tokens)")
    ax.set_ylabel("Mean LLM Judge Score (0–1)")
    ax.set_xticks(sizes)
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(frameon=True)
    plt.tight_layout()
    _save(fig, "fig6_chunksize_querytype_interaction.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Table — Pairwise Welch t-tests between query types
# ═══════════════════════════════════════════════════════════════════════════════
def table_querytype_pairwise():
    """
    Writes table_querytype_pairwise.tex — pairwise Welch's t-test comparisons
    between all four query types, for each of the four RAGAS metrics.
    Significance: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05
    Effect size: Cohen's d (pooled SD).
    """
    from itertools import combinations
    df = load_data()

    metrics = ["f1_score"]
    metric_tex = {
        "f1_score": "LLM Judge Score",
    }
    qt_labels = {
        "single-hop":   "Single-Hop",
        "multi-hop":    "Multi-Hop",
        "comparative":  "Comparative",
        "unanswerable": "Unanswerable",
    }
    qt_keys = ["single-hop", "multi-hop", "comparative", "unanswerable"]
    pairs = list(combinations(qt_keys, 2))

    def _sig_tex(p):
        return (r"$<0.001$***" if p < 0.001 else
                f"${p:.3f}$**" if p < 0.01 else
                f"${p:.3f}$*"  if p < 0.05 else
                f"${p:.3f}$\\phantom{{***}}")

    lines = []
    lines.append(r"% Pairwise Welch's t-tests between query types, per metric")
    lines.append(r"% Cohen's d < 0.2 = trivial, 0.2-0.5 = small, 0.5-0.8 = medium, >0.8 = large")
    lines.append(r"")
    lines.append(r"\begin{table}[!t]")
    lines.append(r"  \caption{Pairwise Comparison of Query Types on LLM Judge Score (Welch's $t$-Test, All Chunk Sizes Pooled)}")
    lines.append(r"  \label{tab:querytype_pairwise}")
    lines.append(r"  \centering")
    lines.append(r"  \renewcommand{\arraystretch}{1.2}")
    lines.append(r"  \begin{tabular}{lccccc}")
    lines.append(r"    \hline")
    lines.append(r"    \textbf{Comparison} & \textbf{Mean A} & \textbf{Mean B} & \textbf{\textit{p}} & \textbf{\textit{d}} \\")
    lines.append(r"    \hline")

    for metric in metrics:
        for qa, qb in pairs:
            a = df[df["query_type"] == qa][metric].dropna()
            b = df[df["query_type"] == qb][metric].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            _, p_val = stats.ttest_ind(a, b, equal_var=False)
            d_val    = _cohens_d(a, b)
            la, lb   = qt_labels[qa], qt_labels[qb]
            lines.append(
                f"    {la} vs.\\ {lb} & {a.mean():.3f} & {b.mean():.3f} "
                f"& {_sig_tex(p_val)} & {d_val:.3f} \\\\"
            )
        lines.append(r"    \hline")

    lines.append(r"  \end{tabular}")
    lines.append(r"  \begin{tablenotes}")
    lines.append(r"    \small")
    lines.append(r"    \item Mean A and Mean B are group means across all chunk sizes.")
    lines.append(r"    $d$ = Cohen's $d$ (effect size, sign indicates direction).")
    lines.append(r"    Significance: ***\,$p < 0.001$; **\,$p < 0.01$; *\,$p < 0.05$;")
    lines.append(r"    unlabelled $p$-values are non-significant ($p \geq 0.05$).")
    lines.append(r"  \end{tablenotes}")
    lines.append(r"\end{table}")
    lines.append(r"")

    tex = "\n".join(lines)
    path = "table_querytype_pairwise.tex"
    with open(path, "w") as f:
        f.write(tex)
    print(f"  Saved {path}")

    # Also print a readable markdown summary
    print("\n=== Query Type Pairwise Tests (Markdown) ===\n")
    print(f"{'Metric':<22} {'Comparison':<38} {'Mean A':>7} {'Mean B':>7} {'p':>10} {'d':>7}")
    print("-" * 100)
    for metric in metrics:
        for qa, qb in pairs:
            a = df[df["query_type"] == qa][metric].dropna()
            b = df[df["query_type"] == qb][metric].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            _, p_val = stats.ttest_ind(a, b, equal_var=False)
            d_val    = _cohens_d(a, b)
            sig      = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            pair_str = f"{qt_labels[qa]} vs. {qt_labels[qb]}"
            print(f"{metric_tex[metric]:<22} {pair_str:<38} {a.mean():>7.3f} {b.mean():>7.3f} {p_val:>10.4f} {sig:>4}  d={d_val:>6.3f}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Table — Pairwise Welch t-tests between domains (LLM Judge Score)
# ═══════════════════════════════════════════════════════════════════════════════
def table_domain_pairwise():
    from itertools import combinations
    df = load_data()

    domains = ["general", "medical", "finance"]
    dom_labels = {"general": "General", "medical": "Medical", "finance": "Finance"}
    pairs = list(combinations(domains, 2))

    def _sig_tex(p):
        return (r"$<0.001$***" if p < 0.001 else
                f"${p:.3f}$**" if p < 0.01 else
                f"${p:.3f}$*"  if p < 0.05 else
                f"${p:.3f}$\\phantom{{***}}")

    lines = []
    lines.append(r"% Pairwise Welch's t-tests between domains on LLM Judge Score")
    lines.append(r"% Cohen's d < 0.2 = trivial, 0.2-0.5 = small, 0.5-0.8 = medium, >0.8 = large")
    lines.append(r"")
    lines.append(r"\begin{table}[!t]")
    lines.append(r"  \caption{Pairwise Comparison of Domains on LLM Judge Score (Welch's $t$-Test, All Chunk Sizes Pooled)}")
    lines.append(r"  \label{tab:domain_pairwise}")
    lines.append(r"  \centering")
    lines.append(r"  \renewcommand{\arraystretch}{1.2}")
    lines.append(r"  \begin{tabular}{lccccc}")
    lines.append(r"    \hline")
    lines.append(r"    \textbf{Comparison} & \textbf{Mean A} & \textbf{Mean B} & \textbf{\textit{p}} & \textbf{\textit{d}} \\")
    lines.append(r"    \hline")

    for da, db in pairs:
        a = df[df["domain"] == da]["f1_score"].dropna()
        b = df[df["domain"] == db]["f1_score"].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        _, p_val = stats.ttest_ind(a, b, equal_var=False)
        d_val    = _cohens_d(a, b)
        la, lb   = dom_labels[da], dom_labels[db]
        lines.append(
            f"    {la} vs.\\ {lb} & {a.mean():.3f} & {b.mean():.3f} "
            f"& {_sig_tex(p_val)} & {d_val:.3f} \\\\"
        )

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \begin{tablenotes}")
    lines.append(r"    \small")
    lines.append(r"    \item Mean A and Mean B are group means across all chunk sizes.")
    lines.append(r"    $d$ = Cohen's $d$ (effect size, sign indicates direction).")
    lines.append(r"    Significance: ***\,$p < 0.001$; **\,$p < 0.01$; *\,$p < 0.05$;")
    lines.append(r"    unlabelled $p$-values are non-significant ($p \geq 0.05$).")
    lines.append(r"  \end{tablenotes}")
    lines.append(r"\end{table}")
    lines.append(r"")

    tex = "\n".join(lines)
    path = "table_domain_pairwise.tex"
    with open(path, "w") as f:
        f.write(tex)
    print(f"  Saved {path}")

    print("\n=== Domain Pairwise Tests (Markdown) ===\n")
    print(f"{'Comparison':<35} {'Mean A':>7} {'Mean B':>7} {'p':>10} {'d':>7}")
    print("-" * 70)
    for da, db in pairs:
        a = df[df["domain"] == da]["f1_score"].dropna()
        b = df[df["domain"] == db]["f1_score"].dropna()
        _, p_val = stats.ttest_ind(a, b, equal_var=False)
        d_val    = _cohens_d(a, b)
        sig      = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"{dom_labels[da] + ' vs. ' + dom_labels[db]:<35} {a.mean():>7.3f} {b.mean():>7.3f} {p_val:>10.4f} {sig:>4}  d={d_val:>6.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6b — Fixed vs. LC Semantic: LLM Judge Score + context metrics, with
#           significance brackets and Cohen's d
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_fixed_vs_semantic():
    df    = load_data()
    sizes = [128, 256, 512]
    metrics = ["f1_score", "context_precision", "context_recall"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)

    for ax, metric in zip(axes, metrics):
        _grouped_barplot(ax, df, x="chunk_size", y=metric, hue="chunk_method_label",
                         palette=METHOD_PAL, order=sizes, hue_order=METHOD_ORDER)
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        ymin, ymax = ax.get_ylim()
        head = (ymax - ymin) * 0.16
        ax.set_ylim(ymin, ymax + head * 1.7)

        for j, cs in enumerate(sizes):
            a = df[(df["chunk_method"] == "fixed")       & (df["chunk_size"] == cs)][metric].dropna()
            b = df[(df["chunk_method"] == "lc_semantic") & (df["chunk_size"] == cs)][metric].dropna()
            if len(a) > 1 and len(b) > 1:
                _, p = stats.ttest_ind(a, b, equal_var=False)
                d    = _cohens_d(a, b)
                bx_y = ymax + head * 0.55
                ax.plot([j-0.2, j-0.2, j+0.2, j+0.2],
                        [bx_y-head*0.1, bx_y, bx_y, bx_y-head*0.1],
                        color="#555555", linewidth=0.75)
                ax.text(j, bx_y+head*0.05, f"{_sig(p)}, d={d:.2f}",
                        ha="center", va="bottom", fontsize=7.5, color="#333333")

        if ax != axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Method", frameon=True, fontsize=8, title_fontsize=8)

    n_f  = len(df[df["chunk_method"] == "fixed"])
    n_lc = len(df[df["chunk_method"] == "lc_semantic"])
    plt.tight_layout()
    _save(fig, "fig6_fixed_vs_semantic.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Context quality vs. chunk size, Fixed vs. LC Semantic (CI band)
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_context_by_chunk_size():
    df    = load_data()
    sizes = [128, 256, 512]
    mk    = {"Fixed": "o", "LC Semantic": "x"}
    ls    = {"Fixed": "-", "LC Semantic": "--"}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=False)

    for ax, metric in zip(axes, ["context_precision", "context_recall"]):
        for method in METHOD_ORDER:
            key   = "fixed" if method == "Fixed" else "lc_semantic"
            sub   = df[df["chunk_method"] == key]
            means = sub.groupby("chunk_size")[metric].mean()
            sems  = sub.groupby("chunk_size")[metric].sem()
            y     = np.array([means[s] for s in sizes])
            e     = np.array([1.96 * sems[s] for s in sizes])
            color = METHOD_PAL[method]
            ax.fill_between(sizes, y-e, y+e, color=color, alpha=0.13)
            ax.plot(sizes, y, marker=mk[method], linestyle=ls[method],
                    color=color, linewidth=2.0, markersize=8, label=method)
            for i, s in enumerate(sizes):
                n = len(sub[sub["chunk_size"] == s])
                ax.annotate(f"{y[i]:.3f} (n={n})", xy=(s, y[i]),
                            xytext=(7, 7 if method == "Fixed" else -15),
                            textcoords="offset points",
                            fontsize=7.5, color=color)
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticks(sizes)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.legend(frameon=True)
    plt.tight_layout()
    _save(fig, "fig8_context_by_chunk_size.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig A — Eta-squared: 3-way ANOVA (General domain only)
# Factors: Query Type, Chunk Size, Chunk Method (Fixed vs LC Semantic)
# ═══════════════════════════════════════════════════════════════════════════════
def figA_eta_squared_factors():
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    df      = load_data()
    general = df[df["domain"] == "general"]
    metrics = ["f1_score", "faithfulness", "context_precision", "context_recall"]
    factor_map = {
        "Query Type":  "C(query_type)",
        "Chunk Size":  "C(chunk_size)",
        "Chunk Method":"C(chunk_method)",
    }
    rows = []
    for metric in metrics:
        sub = general.dropna(subset=[metric])
        model = ols(
            f"{metric} ~ C(chunk_method) * C(chunk_size) * C(query_type)",
            data=sub,
        ).fit()
        tbl = anova_lm(model, typ=2)
        ss_total = tbl["sum_sq"].sum()
        for label, key in factor_map.items():
            rows.append({
                "Metric": METRIC_LABELS[metric],
                "Factor": label,
                "eta2":   tbl.loc[key, "sum_sq"] / ss_total * 100,
                "p":      tbl.loc[key, "PR(>F)"],
            })

    res    = pd.DataFrame(rows)
    mnames = [METRIC_LABELS[m] for m in metrics]
    x, w   = np.arange(len(mnames)), 0.24
    n      = len(general)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, factor in enumerate(factor_map):
        sub  = res[res["Factor"] == factor]
        vals = sub["eta2"].values
        ps   = sub["p"].values
        ax.bar(x+(i-1)*w, vals, w, label=factor,
               color=FACTOR_PAL[factor], edgecolor="#ffffff",
               linewidth=0.5, alpha=0.92)
        for j, (v, p) in enumerate(zip(vals, ps)):
            ax.text(x[j]+(i-1)*w, v+0.15, f"{v:.1f}%\n{_sig(p)}",
                    ha="center", va="bottom", fontsize=7.5, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(mnames)
    ax.set_ylabel("Variance Explained (η², %)")
    ax.set_ylim(0, float(np.nanmax(res["eta2"].values)) * 1.45)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(frameon=True, loc="upper left", ncol=3)
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    _save(fig, "figA_eta_squared_factors.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig C — LLM Judge Score: chunk size × domain interaction, single-hop only
# (Single-hop used because it is the only query type present across all 3 domains)
# ═══════════════════════════════════════════════════════════════════════════════
def figC_interaction_chunksize_domain():
    df     = load_data()
    single = df[df["query_type"] == "single-hop"]
    sizes  = [128, 256, 512]
    mk     = {"General": "o", "Medical": "s", "Finance": "^"}
    ls     = {"General": "-", "Medical": "--", "Finance": "-."}

    fig, ax = plt.subplots(figsize=(7, 4.4))

    for dom in DOMAIN_ORDER:
        label = dom.title()
        sub   = single[single["domain"] == dom]
        n     = len(sub)
        means = sub.groupby("chunk_size")["f1_score"].mean()
        sems  = sub.groupby("chunk_size")["f1_score"].sem()
        y     = np.array([means[s] for s in sizes])
        e     = np.array([1.96 * sems[s] for s in sizes])
        ax.errorbar(sizes, y, yerr=e,
                    marker=mk[label], linestyle=ls[label],
                    color=DOMAIN_PAL[label],
                    linewidth=2.0, markersize=7, capsize=4,
                    label=f"{label} (n={n:,})")
        for i, s in enumerate(sizes):
            ax.annotate(f"{y[i]:.3f}", xy=(s, y[i]),
                        xytext=(7, 5), textcoords="offset points",
                        fontsize=7.5, color=DOMAIN_PAL[label])
    ax.set_xlabel("Chunk Size (tokens)")
    ax.set_ylabel("Mean LLM Judge Score (0–1)")
    ax.set_xticks(sizes)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(frameon=True)
    plt.tight_layout()
    _save(fig, "figC_interaction_chunksize_domain.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig D — LLM Judge Score: chunk size × query type interaction (general domain)
# All 4 query types included; restricted to general domain (only domain with all types)
# ═══════════════════════════════════════════════════════════════════════════════
def figD_chunksize_querytype_interaction():
    df      = load_data()
    general = df[df["domain"] == "general"]
    sizes   = [128, 256, 512]
    mk  = {"Single-Hop": "o", "Multi-Hop": "s", "Comparative": "D", "Unanswerable": "^"}
    ls  = {"Single-Hop": "-", "Multi-Hop": "--", "Comparative": "-.", "Unanswerable": ":"}
    qt_labels = {
        "single-hop": "Single-Hop", "multi-hop": "Multi-Hop",
        "comparative": "Comparative", "unanswerable": "Unanswerable",
    }

    fig, ax = plt.subplots(figsize=(8, 4.4))

    for qt in QT_ORDER:
        label = qt_labels[qt]
        sub   = general[general["query_type"] == qt]
        n     = len(sub)
        means = sub.groupby("chunk_size")["f1_score"].mean()
        sems  = sub.groupby("chunk_size")["f1_score"].sem()
        y     = np.array([means[s] for s in sizes])
        e     = np.array([1.96 * sems[s] for s in sizes])
        ax.errorbar(sizes, y, yerr=e,
                    marker=mk[label], linestyle=ls[label],
                    color=QT_PAL[label],
                    linewidth=2.0, markersize=7, capsize=4,
                    label=f"{label} (n={n:,})")
        for i, s in enumerate(sizes):
            ax.annotate(f"{y[i]:.3f}", xy=(s, y[i]),
                        xytext=(7, 5 if i < 2 else -14),
                        textcoords="offset points",
                        fontsize=7.5, color=QT_PAL[label])
    ax.set_xlabel("Chunk Size (tokens)")
    ax.set_ylabel("Mean LLM Judge Score (0–1)")
    ax.set_xticks(sizes)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    _save(fig, "figD_chunksize_querytype_interaction.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig E — LLM Judge Score + context recall vs. chunk size (CI bands)
#          Fixed vs. LC Semantic side-by-side
# ═══════════════════════════════════════════════════════════════════════════════
def figE_metrics_chunksize_method():
    df    = load_data()
    sizes = [128, 256, 512]
    mk    = {"Fixed": "o", "LC Semantic": "s"}
    ls    = {"Fixed": "-", "LC Semantic": "--"}
    metrics = ["f1_score", "context_recall"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=False)

    for ax, metric in zip(axes, metrics):
        for method in METHOD_ORDER:
            key   = "fixed" if method == "Fixed" else "lc_semantic"
            sub   = df[df["chunk_method"] == key]
            means = sub.groupby("chunk_size")[metric].mean()
            sems  = sub.groupby("chunk_size")[metric].sem()
            y     = np.array([means[s] for s in sizes])
            e     = np.array([1.96 * sems[s] for s in sizes])
            color = METHOD_PAL[method]
            ax.fill_between(sizes, y-e, y+e, color=color, alpha=0.13)
            ax.plot(sizes, y, marker=mk[method], linestyle=ls[method],
                    color=color, linewidth=2.0, markersize=7, label=method)
            for i, s in enumerate(sizes):
                ax.annotate(f"{y[i]:.3f}", xy=(s, y[i]),
                            xytext=(0, 9 if method == "Fixed" else -16),
                            textcoords="offset points",
                            ha="center", fontsize=7.5, color=color)
        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticks(sizes)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.legend(frameon=True)
    plt.tight_layout()
    _save(fig, "figE_metrics_chunksize_method.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig F — LLM Judge Score: domain × query type facet grid (Fixed vs LC Semantic)
# ═══════════════════════════════════════════════════════════════════════════════
def figF_chunksize_domain_querytype_facet():
    df = load_data()
    sizes    = [128, 256, 512]
    dom_labels = {"general": "General", "medical": "Medical", "finance": "Finance"}
    qt_labels  = {
        "single-hop": "Single-Hop", "multi-hop": "Multi-Hop",
        "comparative": "Comparative", "unanswerable": "Unanswerable",
    }

    fig, axes = plt.subplots(
        3, 4, figsize=(15, 9), sharey="row", sharex=True,
        gridspec_kw={"hspace": 0.45, "wspace": 0.15},
    )

    for r, dom in enumerate(DOMAIN_ORDER):
        for c, qt in enumerate(QT_ORDER):
            ax  = axes[r][c]
            sub = df[(df["domain"] == dom) & (df["query_type"] == qt)]
            if len(sub) == 0:
                ax.set_visible(False)
                continue

            for method in METHOD_ORDER:
                key   = "fixed" if method == "Fixed" else "lc_semantic"
                msub  = sub[sub["chunk_method"] == key]
                means = msub.groupby("chunk_size")["f1_score"].mean()
                sems  = msub.groupby("chunk_size")["f1_score"].sem()
                y     = np.array([means.get(s, np.nan) for s in sizes])
                e     = np.array([1.96 * sems.get(s, np.nan) for s in sizes])
                ax.errorbar(sizes, y, yerr=e,
                            marker="o" if method == "Fixed" else "s",
                            linestyle="-" if method == "Fixed" else "--",
                            color=METHOD_PAL[method],
                            linewidth=1.8, markersize=5, capsize=3,
                            label=method)

            ax.set_xticks(sizes)
            ax.tick_params(axis="x", labelsize=7.5)
            ax.tick_params(axis="y", labelsize=7.5)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            if c == 0:
                ax.set_ylabel(f"{dom_labels[dom]}\nLLM Judge Score", fontsize=8.5)
            if r == 2:
                ax.set_xlabel("Chunk Size", fontsize=8.5)

    handles = [
        plt.Line2D([0], [0], color=METHOD_PAL["Fixed"],        marker="o", linestyle="-",  label="Fixed"),
        plt.Line2D([0], [0], color=METHOD_PAL["LC Semantic"],  marker="s", linestyle="--", label="LC Semantic"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9,
               frameon=True, bbox_to_anchor=(1.0, 1.0))
    _save(fig, "figF_chunksize_domain_querytype_facet.png")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Generating figures (Fixed + LC Semantic only, primary metric: f1_score)...\n")

    fig1_eta_squared()
    fig2_domain_comparison()
    fig3_querytype_comparison()
    fig3b_metrics_by_querytype()
    fig4_fixed_vs_lc_semantic()
    fig4_metrics_by_query_type()
    fig5_chunksize_context_quality()
    fig5_domain_comparison()
    fig6_chunksize_domain_interaction()
    fig6_chunksize_querytype_interaction()
    table_querytype_pairwise()
    table_domain_pairwise()
    fig6_fixed_vs_semantic()
    fig8_context_by_chunk_size()
    figA_eta_squared_factors()
    figC_interaction_chunksize_domain()
    figD_chunksize_querytype_interaction()
    figE_metrics_chunksize_method()
    figF_chunksize_domain_querytype_facet()

    print(f"\nAll 15 figures saved to {OUT}/")


if __name__ == "__main__":
    main()
