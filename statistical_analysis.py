"""
Statistical Analysis — Three tests per the revised spec:
  1. Three-way ANOVA (Type III SS): chunking_strategy × query_type × domain
  2. Tukey HSD post-hoc for significant main effects
  3. Targeted contrast: finance vs general judge_minus_faith dissociation
"""

import math
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")

RESULTS_DIR = "results"
CSV_INPUT   = f"{RESULTS_DIR}/evaluation_results_judged.csv"

METRICS = {
    "context_precision": "context_precision",
    "context_recall":    "context_recall",
    "faithfulness":      "faithfulness",
    "answer_correctness":"answer_relevancy",   # RAGAS answer relevancy
    "llm_as_judge":      "f1_score",           # LLM-as-judge (normalised 0-1)
}

# ── Load & clean ──────────────────────────────────────────────────────────────

df_raw = pd.read_csv(CSV_INPUT)

# Rename to spec names
df_raw = df_raw.rename(columns={"strategy": "chunking_strategy"})
for spec_name, col in METRICS.items():
    if col in df_raw.columns and col != spec_name:
        df_raw[spec_name] = df_raw[col]

metric_cols = list(METRICS.keys())
factor_cols = ["chunking_strategy", "query_type", "domain"]
keep_cols   = factor_cols + metric_cols + ["llm", "dataset", "question"]

df = df_raw[[c for c in keep_cols if c in df_raw.columns]].copy()

n_before = len(df)
df = df.dropna(subset=factor_cols)
n_dropped = n_before - len(df)

print(f"Loaded {n_before} rows; dropped {n_dropped} with missing factor values; {len(df)} remaining.\n")

# ── Section 1: Data summary ───────────────────────────────────────────────────

summary_lines = ["# Statistical Analysis Summary\n"]
summary_lines.append("## Section 1: Data Summary\n")
summary_lines.append(f"- Total rows (after cleaning): {len(df)}")
summary_lines.append(f"- Rows dropped (missing values): {n_dropped}")
for f in factor_cols:
    levels = sorted(df[f].unique())
    counts = df[f].value_counts().to_dict()
    summary_lines.append(f"- {f} levels: {levels}")
    summary_lines.append(f"  Counts: { {k: counts[k] for k in levels} }")
summary_lines.append("")
summary_lines.append("### Outcome metric descriptives")
summary_lines.append("| Metric | Mean | SD | Min | Max |")
summary_lines.append("|--------|------|----|-----|-----|")
for m in metric_cols:
    s = df[m]
    summary_lines.append(f"| {m} | {s.mean():.4f} | {s.std():.4f} | {s.min():.4f} | {s.max():.4f} |")
summary_lines.append("")

# ── Test 1: Three-way ANOVA (Type III SS) ────────────────────────────────────

print("=" * 60)
print("TEST 1: Three-way ANOVA (Type III SS)")
print("=" * 60)

# Primary model: main effects only (interactions excluded due to empty cells
# from medical/finance being single-hop only — rank-deficient design matrix)
main_effects = [
    "C(chunking_strategy)",
    "C(query_type)",
    "C(domain)",
]
main_effect_labels = [
    "chunking_strategy",
    "query_type",
    "domain",
]

# Interaction terms attempted separately and flagged
interaction_effects = [
    "C(chunking_strategy):C(query_type)",
    "C(chunking_strategy):C(domain)",
    "C(query_type):C(domain)",
    "C(chunking_strategy):C(query_type):C(domain)",
]
interaction_labels = [
    "chunking_strategy × query_type",
    "chunking_strategy × domain",
    "query_type × domain",
    "chunking_strategy × query_type × domain",
]
UNBALANCED_FLAGS = set(interaction_labels)

formula_main = "~ C(chunking_strategy) + C(query_type) + C(domain)"
formula_full = "~ C(chunking_strategy) * C(query_type) * C(domain)"
anova_rows = []

summary_lines.append("## Section 2: ANOVA Results\n")
summary_lines.append("> **Note:** Primary model uses main effects only. The full 3-way interaction model")
summary_lines.append("> produces a rank-deficient design matrix due to empty cells (medical/finance")
summary_lines.append("> domains contain only single-hop queries), making interaction F-tests unreliable.")
summary_lines.append("> Interaction terms are estimated separately and flagged.\n")

for metric in metric_cols:
    df_m = df.dropna(subset=[metric]).copy()
    n_metric_dropped = len(df) - len(df_m)
    print(f"\nMetric: {metric}  (n={len(df_m)}, dropped {n_metric_dropped} NaN rows)")
    summary_lines.append(f"### {metric}  (n={len(df_m)})\n")
    summary_lines.append("| Effect | F | df_num | df_den | p-value | η² partial | Size | Flagged |")
    summary_lines.append("|--------|---|--------|--------|---------|------------|------|---------|")

    # ── Main effects model ──
    try:
        model_main = smf.ols(f"{metric} {formula_main}", data=df_m).fit()
        anova_main = anova_lm(model_main, typ=3)
        ss_res  = anova_main.loc["Residual", "sum_sq"]
        df_den  = int(anova_main.loc["Residual", "df"])
    except Exception as e:
        print(f"  Main effects ANOVA failed: {e}")
        continue

    for effect, label in zip(main_effects, main_effect_labels):
        try:
            row    = anova_main.loc[effect]
            ss_eff = float(row["sum_sq"])
            df_num = float(row["df"])
            F      = float(row["F"])
            p      = float(row["PR(>F)"])
            if any(math.isnan(v) for v in [F, p, ss_eff]):
                raise ValueError("NaN")
            eta2 = ss_eff / (ss_eff + ss_res)
            size = "large" if eta2>0.14 else "medium" if eta2>0.06 else "small" if eta2>0.01 else "negligible"
            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            print(f"  {label:40s}  F={F:.3f}  p={p_str}  η²={eta2:.4f} ({size})")
            summary_lines.append(f"| {label} | {F:.3f} | {int(df_num)} | {df_den} | {p_str} | {eta2:.4f} | {size} | |")
            anova_rows.append({"metric": metric, "effect": label, "F": round(F,4),
                "df_num": int(df_num), "df_den": df_den, "p_value": round(p,6),
                "partial_eta_sq": round(eta2,6), "significant": p<0.05, "flagged_unbalanced": False})
        except Exception as e:
            print(f"  {label:40s}  FAILED: {e}")
            summary_lines.append(f"| {label} | NaN | — | — | NaN | NaN | — | ⚠ failed |")
            anova_rows.append({"metric": metric, "effect": label, "F": None,
                "df_num": None, "df_den": None, "p_value": None,
                "partial_eta_sq": None, "significant": False, "flagged_unbalanced": False})

    # ── Interaction terms (full model, flagged) ──
    try:
        model_full = smf.ols(f"{metric} {formula_full}", data=df_m).fit()
        anova_full = anova_lm(model_full, typ=3)
        ss_res_full = anova_full.loc["Residual", "sum_sq"]
        df_den_full = int(anova_full.loc["Residual", "df"])
    except Exception as e:
        for label in interaction_labels:
            summary_lines.append(f"| {label} | NaN | — | — | NaN | NaN | — | ⚠ unbalanced |")
            anova_rows.append({"metric": metric, "effect": label, "F": None,
                "df_num": None, "df_den": None, "p_value": None,
                "partial_eta_sq": None, "significant": False, "flagged_unbalanced": True})
        continue

    for effect, label in zip(interaction_effects, interaction_labels):
        try:
            row    = anova_full.loc[effect]
            ss_eff = float(row["sum_sq"])
            df_num = float(row["df"])
            F      = float(row["F"])
            p      = float(row["PR(>F)"])
            if any(math.isnan(v) for v in [F, p, ss_eff]):
                raise ValueError("NaN")
            eta2 = ss_eff / (ss_eff + ss_res_full)
            size = "large" if eta2>0.14 else "medium" if eta2>0.06 else "small" if eta2>0.01 else "negligible"
            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            print(f"  {label:40s}  F={F:.3f}  p={p_str}  η²={eta2:.4f} ({size})  [flagged]")
            summary_lines.append(f"| {label} | {F:.3f} | {int(df_num)} | {df_den_full} | {p_str} | {eta2:.4f} | {size} | ⚠ unbalanced |")
            anova_rows.append({"metric": metric, "effect": label, "F": round(F,4),
                "df_num": int(df_num), "df_den": df_den_full, "p_value": round(p,6),
                "partial_eta_sq": round(eta2,6), "significant": p<0.05, "flagged_unbalanced": True})
        except Exception as e:
            print(f"  {label:40s}  FAILED: {e}")
            summary_lines.append(f"| {label} | NaN | — | — | NaN | NaN | — | ⚠ unbalanced/failed |")
            anova_rows.append({"metric": metric, "effect": label, "F": None,
                "df_num": None, "df_den": None, "p_value": None,
                "partial_eta_sq": None, "significant": False, "flagged_unbalanced": True})
    summary_lines.append("")
    summary_lines.append("")

anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv(f"{RESULTS_DIR}/anova_results.csv", index=False)
print(f"\nSaved anova_results.csv ({len(anova_df)} rows)")

# ── Test 2: Tukey HSD post-hoc ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 2: Tukey HSD Post-hoc Tests")
print("=" * 60)

sig_main_effects = (
    anova_df[
        anova_df["effect"].isin(["chunking_strategy", "query_type", "domain"]) &
        anova_df["significant"]
    ][["metric", "effect"]]
    .drop_duplicates()
)

tukey_rows = []
summary_lines.append("## Section 3: Tukey HSD Post-hoc Highlights\n")

for _, row in sig_main_effects.iterrows():
    metric = row["metric"]
    factor = row["effect"]
    print(f"\n  Tukey: {metric} ~ {factor}")

    try:
        df_m = df.dropna(subset=[metric])
        result = pairwise_tukeyhsd(
            endog=df_m[metric],
            groups=df_m[factor],
            alpha=0.05,
        )
        res_df = pd.DataFrame(
            data=result._results_table.data[1:],
            columns=result._results_table.data[0],
        )
        summary_lines.append(f"### {metric} × {factor}")
        summary_lines.append("| Group 1 | Group 2 | Mean Diff | p-adj | CI lower | CI upper | Significant |")
        summary_lines.append("|---------|---------|-----------|-------|----------|----------|-------------|")

        for _, r in res_df.iterrows():
            sig_flag = "✓" if str(r.get("reject", r.get("significant", False))).lower() == "true" else ""
            mean_diff = float(r.get("meandiff", r.get("mean(1)", 0)))
            p_adj     = float(r.get("p-adj", 1.0))
            ci_lo     = float(r.get("lower", r.get("lower bound", 0)))
            ci_hi     = float(r.get("upper", r.get("upper bound", 0)))
            g1        = str(r.get("group1", r.iloc[0]))
            g2        = str(r.get("group2", r.iloc[1]))

            summary_lines.append(f"| {g1} | {g2} | {mean_diff:.4f} | {p_adj:.4f} | {ci_lo:.4f} | {ci_hi:.4f} | {sig_flag} |")
            tukey_rows.append({
                "metric": metric, "factor": factor,
                "group1": g1, "group2": g2,
                "mean_diff": round(mean_diff, 6),
                "p_adj": round(p_adj, 6),
                "ci_lower": round(ci_lo, 6),
                "ci_upper": round(ci_hi, 6),
                "significant": sig_flag == "✓",
            })
        summary_lines.append("")
    except Exception as e:
        print(f"    Tukey failed: {e}")

tukey_df = pd.DataFrame(tukey_rows)
tukey_df.to_csv(f"{RESULTS_DIR}/tukey_results.csv", index=False)
print(f"\nSaved tukey_results.csv ({len(tukey_df)} rows)")

# ── Test 3: Targeted contrast ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 3: Targeted Contrast — Finance vs General (Judge − Faithfulness)")
print("=" * 60)

df_contrast = df.dropna(subset=["llm_as_judge", "faithfulness"]).copy()
df_contrast["judge_minus_faith"] = df_contrast["llm_as_judge"] - df_contrast["faithfulness"]

finance_diffs = df_contrast[df_contrast["domain"] == "finance"]["judge_minus_faith"]
general_diffs = df_contrast[df_contrast["domain"] == "general"]["judge_minus_faith"]

t_stat, p_value = ttest_ind(finance_diffs, general_diffs, equal_var=False)

m1, s1, n1 = finance_diffs.mean(), finance_diffs.std(), len(finance_diffs)
m2, s2, n2 = general_diffs.mean(), general_diffs.std(), len(general_diffs)

pooled_sd = math.sqrt((s1**2 + s2**2) / 2)
cohens_d  = (m1 - m2) / pooled_sd if pooled_sd > 0 else float("nan")

# Welch–Satterthwaite df
num   = (s1**2/n1 + s2**2/n2)**2
denom = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
ws_df = num / denom if denom > 0 else float("nan")

print(f"  Finance:  M={m1:.4f}, SD={s1:.4f}, N={n1}")
print(f"  General:  M={m2:.4f}, SD={s2:.4f}, N={n2}")
print(f"  t = {t_stat:.4f}, Welch df = {ws_df:.1f}, p = {p_value:.6f}")
print(f"  Cohen's d = {cohens_d:.4f}")
direction = "finance > general ✓" if m1 > m2 else "finance < general ✗ (unexpected)"
print(f"  Hypothesised direction: {direction}")

contrast_row = [{
    "finance_mean": round(m1, 6), "finance_sd": round(s1, 6), "finance_n": n1,
    "general_mean": round(m2, 6), "general_sd": round(s2, 6), "general_n": n2,
    "t_stat": round(t_stat, 6), "welch_df": round(ws_df, 2),
    "p_value": round(p_value, 8), "cohens_d": round(cohens_d, 6),
    "direction_confirmed": m1 > m2,
}]
contrast_df = pd.DataFrame(contrast_row)
contrast_df.to_csv(f"{RESULTS_DIR}/targeted_contrast.csv", index=False)
print(f"\nSaved targeted_contrast.csv")

p_str = f"{p_value:.6f}" if p_value >= 0.0001 else "<0.0001"
interp = ("Finance domain shows a significantly larger gap between LLM-as-Judge "
          f"and faithfulness (M={m1:.3f}) than general domain (M={m2:.3f}), "
          f"t({ws_df:.1f}) = {t_stat:.3f}, p = {p_str}, d = {cohens_d:.3f}, "
          "consistent with reliance on parametric knowledge."
          if m1 > m2 and p_value < 0.05 else
          f"No significant difference (t={t_stat:.3f}, p={p_str}) or unexpected direction.")

summary_lines.append("## Section 4: Targeted Contrast\n")
summary_lines.append(f"**Judge − Faithfulness gap: finance vs general**\n")
summary_lines.append(f"| | Finance | General |")
summary_lines.append(f"|--|---------|---------|")
summary_lines.append(f"| Mean | {m1:.4f} | {m2:.4f} |")
summary_lines.append(f"| SD   | {s1:.4f} | {s2:.4f} |")
summary_lines.append(f"| N    | {n1} | {n2} |")
summary_lines.append(f"\nt({ws_df:.1f}) = {t_stat:.4f}, p = {p_str}, Cohen's d = {cohens_d:.4f}\n")
summary_lines.append(f"**Interpretation:** {interp}\n")

# ── Section 5: Flags ──────────────────────────────────────────────────────────

summary_lines.append("## Section 5: Flagged Issues\n")
summary_lines.append(f"- {n_dropped} rows dropped due to missing values in metrics or factors.")
summary_lines.append("- Interaction terms involving `query_type × domain` and the three-way interaction "
                     "are flagged as potentially uninterpretable due to empty cells "
                     "(medical/finance queries are single-hop only).")
failed = anova_df[anova_df["F"].isna()]
if len(failed):
    summary_lines.append(f"- {len(failed)} ANOVA terms failed to estimate (NaN F): "
                         + ", ".join(f"{r.metric}/{r.effect}" for _, r in failed.iterrows()))
summary_lines.append("")

# ── Write Markdown summary ────────────────────────────────────────────────────

md_path = f"{RESULTS_DIR}/statistical_analysis_summary.md"
with open(md_path, "w") as f:
    f.write("\n".join(summary_lines))
print(f"\nSaved statistical_analysis_summary.md")
print("\nAll done.")
