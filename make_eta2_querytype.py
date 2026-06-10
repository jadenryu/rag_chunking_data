"""
Recreate the rag_findings_eta2_all.png variance-explained figure, but with
Query type as the highlighted factor (compared against chunking strategy).

Partial eta-squared from the Type III main-effects ANOVA
(chunking_strategy + query_type + domain), matching statistical_analysis.py.
"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import blended_transform_factory
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings("ignore")

# ── colours / style (matched to rag_findings_eta2_all.png) ──────────────────
DARK   = "#1b4f9c"   # highlighted factor (Query type)
LIGHT  = "#a8bdd8"   # comparison factor (Chunking strategy)
TITLE  = "#1a2b4a"
SUB    = "#5a6b8c"
TEXT   = "#1a2b4a"

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    # Arial renders both regular and bold cleanly (the macOS Helvetica .ttc
    # collections fail to expose a bold face in matplotlib); Arial is metrically
    # identical to the original figure's Helvetica-style type.
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})

METRICS = {
    "context_precision":  "Context Precision",
    "context_recall":     "Context Recall",
    "faithfulness":       "Faithfulness",
    "answer_correctness": "Answer Correctness",
    "llm_as_judge":       "LLM-as-Judge",
}


def compute_partial_eta2():
    df = pd.read_csv("results/evaluation_results_judged.csv").rename(
        columns={"strategy": "chunking_strategy"})
    df["answer_correctness"] = df["answer_relevancy"]
    df["llm_as_judge"] = df["f1_score"]
    df = df.dropna(subset=["chunking_strategy", "query_type", "domain"])

    rows = []
    for col, label in METRICS.items():
        dm = df.dropna(subset=[col])
        a = anova_lm(smf.ols(
            f"{col} ~ C(chunking_strategy)+C(query_type)+C(domain)",
            data=dm).fit(), typ=3)
        ssr = a.loc["Residual", "sum_sq"]
        def pe(k):
            s = a.loc[k, "sum_sq"]
            return s / (s + ssr) * 100
        rows.append((label, pe("C(query_type)"), pe("C(chunking_strategy)")))
    return rows


def rounded_bar(ax, x0, y, width, height, color, xmax, mut_aspect, r_data):
    """Horizontal bar with a fixed-radius rounded-RECTANGLE corner (not a full
    pill). `r_data` is the corner radius in x-data units; `mut_aspect` makes the
    corner render circular given the axis aspect ratio. Both are dpi-independent
    so the shape is identical on screen and in the saved PNG/SVG."""
    w = max(width, 0.012 * xmax)            # keep tiny bars visible
    r = min(r_data, w / 2)                  # never round more than half the bar
    box = FancyBboxPatch(
        (x0, y - height / 2), w, height,
        boxstyle=f"round,pad=0,rounding_size={r}",
        mutation_aspect=mut_aspect,
        linewidth=0, facecolor=color, clip_on=False, zorder=3)
    ax.add_patch(box)


def main():
    data = compute_partial_eta2()
    xmax = max(v for _, v, _ in data) * 1.18

    fig, ax = plt.subplots(figsize=(10.2, 7.0))
    # leave the left ~20% of the figure for left-aligned metric labels
    fig.subplots_adjust(left=0.205, right=0.99, top=0.84, bottom=0.12)
    label_x = 0.018   # figure-fraction x for left-aligned labels + title

    group_gap = 1.0
    bar_h = 0.34
    inner = 0.40            # gap between the two bars in a metric group
    y = 0.0
    yticks = []
    bars = []

    for label, qt, ck in data[::-1]:        # reverse so first metric is on top
        y_dark = y + inner / 2
        y_light = y - inner / 2
        bars.append((qt, y_dark, DARK))
        bars.append((ck, y_light, LIGHT))
        yticks.append((y, label))
        y += group_gap + inner

    # finalise axes geometry first so the corner radius can be aspect-corrected
    ax.set_yticks([])
    ax.set_xlim(0, xmax)
    ax.set_xticks([])
    ymin, ymax = -group_gap, y - group_gap + inner / 2
    ax.set_ylim(ymin, ymax)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.canvas.draw()

    # pixels-per-data-unit (ratio is dpi-independent)
    bb = ax.get_window_extent()
    px_per_x = bb.width / xmax
    px_per_y = bb.height / (ymax - ymin)
    mut_aspect = px_per_x / px_per_y                 # circular-looking corners
    bar_h_px = bar_h * px_per_y
    r_px = 0.33 * bar_h_px                            # moderate rounded-rect
    r_data = r_px / px_per_x

    for val, yc, color in bars:
        rounded_bar(ax, 0, yc, val, bar_h, color, xmax, mut_aspect, r_data)
        ax.text(max(val, 0.012 * xmax) + 0.015 * xmax, yc,
                f"{val:.2f}%", va="center", ha="left",
                fontsize=15, color=TEXT)

    # left-aligned metric labels (figure-x, data-y blended transform)
    trans = blended_transform_factory(fig.transFigure, ax.transData)
    for yc, label in yticks:
        ax.text(label_x, yc, label, transform=trans, ha="left", va="center",
                fontsize=15, color=TITLE, clip_on=False)

    # titles (left-aligned with the metric labels)
    fig.text(label_x, 0.965, "Query type drives context precision",
             fontsize=23, fontweight="bold", color=TITLE, ha="left", va="top")
    fig.text(label_x, 0.905,
             "Variance explained (η²) – Type III ANOVA, main effects",
             fontsize=16, color=SUB, ha="left", va="top")

    # legend (bottom, left-aligned with labels)
    leg_y = 0.020
    fig.patches.append(plt.Rectangle((label_x, leg_y), 0.022, 0.030,
                                      transform=fig.transFigure, facecolor=DARK,
                                      edgecolor="none", clip_on=False))
    fig.text(label_x + 0.032, leg_y + 0.015, "Query type", fontsize=15,
             color=TITLE, va="center", ha="left")
    fig.patches.append(plt.Rectangle((label_x + 0.235, leg_y), 0.022, 0.030,
                                      transform=fig.transFigure, facecolor=LIGHT,
                                      edgecolor="none", clip_on=False))
    fig.text(label_x + 0.267, leg_y + 0.015, "Chunking strategy", fontsize=15,
             color=TITLE, va="center", ha="left")

    out = "figures/rag_findings_eta2_querytype"
    fig.savefig(out + ".png", dpi=200)
    fig.savefig(out + ".svg")
    plt.close(fig)
    print(f"Saved {out}.png / .svg")
    for label, qt, ck in data:
        print(f"  {label:20s} query_type={qt:6.2f}%  chunking={ck:5.2f}%")


if __name__ == "__main__":
    main()
