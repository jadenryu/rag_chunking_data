import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

METRICS = ["f1_score", "answer_relevancy", "faithfulness", "context_precision", "context_recall"]


def load_results(filename: str = "evaluation_results.json") -> pd.DataFrame:
    path = os.path.join(config.RESULTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def _save_fig(fig, name: str):
    path = os.path.join(config.FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_by_strategy(df: pd.DataFrame):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 5))
    strategies = sorted(df["strategy"].unique())

    for ax, metric in zip(axes, METRICS):
        grouped = df.groupby("strategy")[metric]
        means = grouped.mean().reindex(strategies)
        cis = grouped.sem().reindex(strategies) * 1.96
        bars = ax.bar(range(len(strategies)), means, yerr=cis, capsize=3, alpha=0.8)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha="right", fontsize=11)
        ax.set_title(metric.replace("_", " ").title(), fontsize=13)
        ax.set_ylim(0, 1)

    fig.suptitle("Performance by Chunking Strategy", fontsize=18, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "performance_by_strategy")


def plot_by_domain(df: pd.DataFrame):
    domains = sorted(df["domain"].unique())
    strategies = sorted(df["strategy"].unique())

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(domains))
        width = 0.8 / len(strategies)

        for i, strategy in enumerate(strategies):
            means = []
            for domain in domains:
                subset = df[(df["domain"] == domain) & (df["strategy"] == strategy)]
                means.append(subset[metric].mean() if len(subset) > 0 else 0)
            offset = (i - len(strategies) / 2 + 0.5) * width
            ax.bar(x + offset, means, width, label=strategy, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(domains)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} by Domain and Strategy")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        _save_fig(fig, f"domain_{metric}")


def plot_by_query_type(df: pd.DataFrame):
    query_types = sorted(df["query_type"].unique())
    strategies = sorted(df["strategy"].unique())

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(query_types))
        width = 0.8 / len(strategies)

        for i, strategy in enumerate(strategies):
            means = []
            for qt in query_types:
                subset = df[(df["query_type"] == qt) & (df["strategy"] == strategy)]
                means.append(subset[metric].mean() if len(subset) > 0 else 0)
            offset = (i - len(strategies) / 2 + 0.5) * width
            ax.bar(x + offset, means, width, label=strategy, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(query_types, rotation=30, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} by Query Type and Strategy")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        _save_fig(fig, f"querytype_{metric}")


def plot_by_llm(df: pd.DataFrame):
    llms = sorted(df["llm"].unique())

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 5))

    for ax, metric in zip(axes, METRICS):
        grouped = df.groupby("llm")[metric]
        means = grouped.mean().reindex(llms)
        cis = grouped.sem().reindex(llms) * 1.96
        ax.bar(range(len(llms)), means, yerr=cis, capsize=3, alpha=0.8)
        ax.set_xticks(range(len(llms)))
        ax.set_xticklabels(llms, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.set_ylim(0, 1)

    fig.suptitle("Performance by LLM", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "performance_by_llm")


def plot_heatmap_strategy_domain(df: pd.DataFrame):
    strategies = sorted(df["strategy"].unique())
    domains = sorted(df["domain"].unique())

    for metric in METRICS:
        pivot = df.pivot_table(
            values=metric, index="strategy", columns="domain", aggfunc="mean"
        ).reindex(index=strategies, columns=domains)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies)
        ax.set_title(f"{metric.replace('_', ' ').title()}: Strategy vs Domain", fontsize=16)

        for i in range(len(strategies)):
            for j in range(len(domains)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)

        fig.colorbar(im)
        fig.tight_layout()
        _save_fig(fig, f"heatmap_strategy_domain_{metric}")


def plot_heatmap_strategy_querytype(df: pd.DataFrame):
    strategies = sorted(df["strategy"].unique())
    query_types = sorted(df["query_type"].unique())

    for metric in METRICS:
        pivot = df.pivot_table(
            values=metric, index="strategy", columns="query_type", aggfunc="mean"
        ).reindex(index=strategies, columns=query_types)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(query_types)))
        ax.set_xticklabels(query_types, rotation=30, ha="right")
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies)
        ax.set_title(f"{metric.replace('_', ' ').title()}: Strategy vs Query Type")

        for i in range(len(strategies)):
            for j in range(len(query_types)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)

        fig.colorbar(im)
        fig.tight_layout()
        _save_fig(fig, f"heatmap_strategy_querytype_{metric}")


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print("\n--- By Chunking Strategy ---")
    summary = df.groupby("strategy")[METRICS].agg(["mean", "std"]).round(4)
    print(summary.to_string())

    print("\n--- By Domain ---")
    summary = df.groupby("domain")[METRICS].agg(["mean", "std"]).round(4)
    print(summary.to_string())

    print("\n--- By Query Type ---")
    summary = df.groupby("query_type")[METRICS].agg(["mean", "std"]).round(4)
    print(summary.to_string())

    print("\n--- By LLM ---")
    summary = df.groupby("llm")[METRICS].agg(["mean", "std"]).round(4)
    print(summary.to_string())

    full_summary = df.groupby(["strategy", "domain", "query_type", "llm"])[METRICS].mean().round(4)
    csv_path = os.path.join(config.RESULTS_DIR, "summary_statistics.csv")
    full_summary.to_csv(csv_path)
    print(f"\nFull summary saved to {csv_path}")


BALANCED_DATASETS = {
    "hotpotqa", "squad2",
    "pubmedqa", "pubmedqa_artificial",
    "financeqa", "financebench",
}


def run_all_analysis():
    df = load_results("evaluation_results_judged.json")
    print(f"Loaded {len(df)} evaluation records")
    df = df[df["dataset"].isin(BALANCED_DATASETS)].copy()
    print(f"Filtered to {len(df)} records ({', '.join(sorted(BALANCED_DATASETS))})")

    print("\nGenerating plots...")
    plot_by_strategy(df)
    plot_by_domain(df)
    plot_by_query_type(df)
    plot_by_llm(df)
    plot_heatmap_strategy_domain(df)
    plot_heatmap_strategy_querytype(df)

    print_summary(df)
    print(f"\nAll figures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    run_all_analysis()
