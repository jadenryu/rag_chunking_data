"""
Quick smoke-test for RAGAS 0.4.x integration.
Runs on 5 rows from paper_dataset.csv — verify this passes before running full pipeline.
"""
import warnings
warnings.filterwarnings("ignore")

import os, json
import pandas as pd
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    LLMContextRecall,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import config

# ── Load 5 rows from paper_dataset ───────────────────────────────────────────
df = pd.read_csv("results/paper_dataset.csv").head(5)

# We need retrieved_contexts — load from the raw rag_results json
RAG_JSON = "results/rag_results.json"
if not os.path.exists(RAG_JSON):
    print(f"ERROR: {RAG_JSON} not found — need the raw RAG results to get retrieved chunks.")
    exit(1)

with open(RAG_JSON) as f:
    rag_raw = json.load(f)

# Build a lookup: (question, strategy, llm) -> retrieved_chunks
lookup = {}
for r in rag_raw:
    for llm, response in r.get("responses", {}).items():
        key = (r["question"].strip(), r["strategy"], llm)
        lookup[key] = r.get("retrieved_chunks", [])

# ── Build RAGAS samples ───────────────────────────────────────────────────────
samples = []
missing = 0
for _, row in df.iterrows():
    key = (row["question"].strip(), row["strategy"], row["llm"])
    chunks = lookup.get(key)
    if not chunks:
        missing += 1
        continue
    samples.append(SingleTurnSample(
        user_input=row["question"],
        response=row["response"],
        retrieved_contexts=chunks,
        reference=row["ground_truth"],
    ))

print(f"Built {len(samples)} samples ({missing} skipped — no chunk lookup)")

if not samples:
    print("ERROR: No samples could be built. Check rag_results.json path.")
    exit(1)

# ── Configure LLM + embeddings ────────────────────────────────────────────────
# Use OpenRouter (OpenAI-compatible) for LLM calls
llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )
)
# Use local sentence-transformers for embeddings (no OpenAI key needed)
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

metrics = [
    Faithfulness(llm=llm),
    LLMContextRecall(llm=llm),
    LLMContextPrecisionWithReference(llm=llm),
    ResponseRelevancy(llm=llm, embeddings=embeddings),
]

# ── Run evaluation ────────────────────────────────────────────────────────────
dataset = EvaluationDataset(samples=samples)
print(f"\nRunning RAGAS on {len(samples)} samples...")
result = evaluate(dataset=dataset, metrics=metrics)

df_out = result.to_pandas()
print("\n=== Columns returned ===")
print(list(df_out.columns))
print("\n=== Results ===")
metric_cols = [c for c in df_out.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
print(df_out[metric_cols].to_string())
print("\nMeans:")
print(df_out[metric_cols].mean())
print("\nSUCCESS — RAGAS 0.4.x is working correctly.")
