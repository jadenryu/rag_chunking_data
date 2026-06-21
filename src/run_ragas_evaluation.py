"""
Full RAGAS 0.4.x evaluation over the paper dataset (fixed + LC semantic).
Runs in batches of 50 with checkpoint saves so it can resume if interrupted.

Output: results/ragas_evaluation.csv
"""
import warnings
warnings.filterwarnings("ignore")

import os, json, time, traceback
import pandas as pd
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    LLMContextRecall,
    LLMContextPrecisionWithReference,
    ResponseRelevancy,
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import config

BATCH_SIZE  = 50
CHECKPOINT  = "results/ragas_checkpoint.csv"
OUTPUT_CSV  = "results/ragas_evaluation.csv"
SKIPPED_LOG = "results/ragas_skipped.txt"
RAG_JSONS   = [
    "results/rag_results.json",
    "results/rag_results_lc_semantic.json",
]

# ── Load retrieved chunks lookup ──────────────────────────────────────────────
print("Loading RAG results for retrieved chunks...")
lookup = {}
for path in RAG_JSONS:
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        continue
    with open(path) as f:
        for r in json.load(f):
            for llm in r.get("responses", {}):
                key = (r["question"].strip(), r["strategy"], llm)
                lookup[key] = r.get("retrieved_chunks", [])
print(f"  {len(lookup):,} chunk lookups loaded")

# ── Load paper dataset ────────────────────────────────────────────────────────
df = pd.read_csv("results/paper_dataset.csv")
print(f"  {len(df):,} rows in paper dataset")

# ── Skip already-completed rows (checkpoint) ──────────────────────────────────
completed_keys = set()
if os.path.exists(CHECKPOINT):
    done = pd.read_csv(CHECKPOINT)
    # Deduplicate checkpoint in case of resume overlap
    done = done.drop_duplicates(subset=["question", "strategy", "llm"])
    done.to_csv(CHECKPOINT, index=False)
    completed_keys = set(zip(done["question"], done["strategy"], done["llm"]))
    print(f"  Resuming — {len(completed_keys):,} rows already done")

rows_to_run = df[
    ~df.apply(lambda r: (r["question"], r["strategy"], r["llm"]) in completed_keys, axis=1)
].reset_index(drop=True)
print(f"  {len(rows_to_run):,} rows remaining\n")

if len(rows_to_run) == 0:
    print("All rows already completed. Finalizing output...")
else:
    # ── Configure RAGAS ───────────────────────────────────────────────────────
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="openai/gpt-4o-mini",
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    metrics = [
        Faithfulness(llm=llm),
        LLMContextRecall(llm=llm),
        LLMContextPrecisionWithReference(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
    ]

    # ── Run in batches ────────────────────────────────────────────────────────
    total_batches = (len(rows_to_run) + BATCH_SIZE - 1) // BATCH_SIZE
    skipped_rows = []
    batch_times = []

    # Open checkpoint file in append mode so we never re-read the whole file
    checkpoint_exists = os.path.exists(CHECKPOINT)
    checkpoint_file = open(CHECKPOINT, "a", buffering=1)  # line-buffered

    for batch_num in range(total_batches):
        t_start = time.time()
        batch = rows_to_run.iloc[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]

        # ETA estimate
        if batch_times:
            avg = sum(batch_times) / len(batch_times)
            remaining = (total_batches - batch_num) * avg
            eta = f"  ETA ~{remaining/60:.1f} min remaining"
        else:
            eta = ""
        print(f"Batch {batch_num + 1}/{total_batches} ({len(batch)} rows)...{eta}")

        samples, meta = [], []
        for _, row in batch.iterrows():
            key = (row["question"].strip(), row["strategy"], row["llm"])
            chunks = lookup.get(key)
            response = row["response"]
            if not chunks or not isinstance(response, str) or not response.strip():
                skipped_rows.append(f"{row['strategy']} | {row['llm']} | {row['question'][:80]}")
                continue
            samples.append(SingleTurnSample(
                user_input=row["question"],
                response=response,
                retrieved_contexts=chunks,
                reference=str(row["ground_truth"]),
            ))
            meta.append({
                "question":         row["question"],
                "ground_truth":     row["ground_truth"],
                "response":         row["response"],
                "query_type":       row["query_type"],
                "domain":           row["domain"],
                "dataset":          row["dataset"],
                "strategy":         row["strategy"],
                "llm":              row["llm"],
                "f1_score":         row.get("f1_score"),
                "judge_raw_score":  row.get("judge_raw_score"),
                "token_f1":         row.get("token_f1"),
            })

        if not samples:
            print(f"  All {len(batch)} rows in batch had no chunk lookup — skipped")
            continue

        try:
            result   = evaluate(dataset=EvaluationDataset(samples=samples), metrics=metrics)
            df_batch = result.to_pandas()

            # Guard: RAGAS must return exactly as many rows as we sent
            if len(df_batch) != len(meta):
                raise ValueError(
                    f"RAGAS returned {len(df_batch)} rows but expected {len(meta)}. "
                    "Skipping batch to avoid row misalignment."
                )

            rows_out = []
            for i, m in enumerate(meta):
                m["faithfulness"]      = df_batch.iloc[i].get("faithfulness")
                m["context_recall"]    = df_batch.iloc[i].get("context_recall")
                m["context_precision"] = df_batch.iloc[i].get("llm_context_precision_with_reference")
                m["answer_relevancy"]  = df_batch.iloc[i].get("answer_relevancy")
                rows_out.append(m)

            # Append batch to checkpoint (never re-read the file)
            batch_df = pd.DataFrame(rows_out)
            write_header = not checkpoint_exists
            batch_df.to_csv(checkpoint_file, index=False, header=write_header)
            checkpoint_exists = True
            checkpoint_file.flush()

            elapsed = time.time() - t_start
            batch_times.append(elapsed)
            rows_done = len(completed_keys) + (batch_num + 1) * BATCH_SIZE
            print(f"  Done in {elapsed:.1f}s — ~{rows_done:,} rows completed so far")

        except Exception as e:
            print(f"  ERROR in batch {batch_num + 1}: {e}")
            traceback.print_exc()
            print("  Skipping batch — these rows will be retried on next run.")
            continue

    checkpoint_file.close()

    # Log skipped rows
    if skipped_rows:
        with open(SKIPPED_LOG, "w") as f:
            f.write(f"{len(skipped_rows)} rows had no chunk lookup and were skipped:\n\n")
            f.write("\n".join(skipped_rows))
        print(f"\n  {len(skipped_rows)} rows skipped (no chunk lookup) — see {SKIPPED_LOG}")

# ── Finalize: deduplicate checkpoint → output CSV ─────────────────────────────
if os.path.exists(CHECKPOINT):
    final = pd.read_csv(CHECKPOINT)
    before = len(final)
    final = final.drop_duplicates(subset=["question", "strategy", "llm"])
    if before != len(final):
        print(f"  Removed {before - len(final)} duplicate rows from checkpoint")
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. {len(final):,} rows saved to {OUTPUT_CSV}")
    expected = len(df)
    if len(final) < expected:
        print(f"  NOTE: {expected - len(final)} rows missing from output (no chunk lookup or batch errors)")
else:
    print("\nNo results were saved.")
