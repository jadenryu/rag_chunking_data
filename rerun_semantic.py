import json
import os
from collections import defaultdict

import config
from chunking import apply_chunking
from dataset_loader import load_processed_dataset
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from evaluation import evaluate_all, save_evaluation

SEMANTIC_STRATEGIES = {
    "lc_semantic_128": {"type": "langchain_semantic", "max_tokens": 128},
    "lc_semantic_256": {"type": "langchain_semantic", "max_tokens": 256},
    "lc_semantic_512": {"type": "langchain_semantic", "max_tokens": 512},
}

RAG_RESULTS_FILE = "rag_results_lc_semantic.json"
EVAL_RESULTS_FILE = "evaluation_results_lc_semantic.json"
EVAL_CSV_FILE = "full_evaluation_results_lc_semantic.csv"


def step_chunk_and_embed():
    print("=" * 60)
    print("STEP 1: Chunking and Embedding (LangChain Semantic)")
    print("=" * 60)

    records = load_processed_dataset()
    vs = VectorStore()

    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)

    for strategy_name, strategy_config in SEMANTIC_STRATEGIES.items():
        coll_name = vs.collection_name(strategy_name)
        existing = [c.name for c in vs.qdrant.get_collections().collections]

        if coll_name in existing:
            vs.qdrant.delete_collection(coll_name)
            print(f"  Deleted existing collection: {coll_name}")

        vs.create_collection(strategy_name, recreate=True)
        print(f"\n--- Strategy: {strategy_name} ---")
        point_id = 0

        for dataset_name, dataset_records in by_dataset.items():
            print(f"  Dataset: {dataset_name} ({len(dataset_records)} records)")

            seen_contexts = set()
            all_chunks = []
            all_metadata = []

            for doc_idx, record in enumerate(dataset_records):
                ctx = record.get("context", "")
                if not ctx or ctx in seen_contexts:
                    continue
                seen_contexts.add(ctx)

                chunks = apply_chunking(ctx, strategy_name, strategy_config)
                for chunk_idx, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "dataset": dataset_name,
                        "domain": record["domain"],
                        "doc_index": doc_idx,
                        "chunk_index": chunk_idx,
                    })

            if all_chunks:
                print(f"    {len(all_chunks)} chunks from {len(seen_contexts)} documents")
                point_id = vs.store_chunks(
                    strategy_name, all_chunks, all_metadata,
                    start_id=point_id,
                )
            else:
                print(f"    No context documents to chunk")

        info = vs.get_collection_info(strategy_name)
        print(f"  Collection {info['name']}: {info['vectors_count']} total vectors")

    print("\nChunking and embedding complete")


def step_generate():
    print("=" * 60)
    print("STEP 2: RAG Generation (LangChain Semantic)")
    print("=" * 60)

    records = load_processed_dataset()
    records = [r for r in records if r.get("context")]

    pipeline = RAGPipeline()
    results_path = os.path.join(config.RESULTS_DIR, RAG_RESULTS_FILE)

    target_strategies = list(SEMANTIC_STRATEGIES.keys())

    for s in target_strategies:
        config.CHUNKING_STRATEGIES[s] = SEMANTIC_STRATEGIES[s]

    results = pipeline.run_full_pipeline(
        records,
        strategies=target_strategies,
        resume_from=results_path,
    )

    print(f"\nGeneration complete: {len(results)} results")


def step_evaluate():
    print("=" * 60)
    print("STEP 3: Evaluation (LangChain Semantic)")
    print("=" * 60)

    results_path = os.path.join(config.RESULTS_DIR, RAG_RESULTS_FILE)
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    evaluated = evaluate_all(results)
    save_evaluation(evaluated, filename=EVAL_RESULTS_FILE)

    import pandas as pd
    df = pd.DataFrame(evaluated)
    csv_path = os.path.join(config.RESULTS_DIR, EVAL_CSV_FILE)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved CSV to {csv_path}")

    print(f"\nEvaluation complete: {len(evaluated)} results")


if __name__ == "__main__":
    import sys

    missing = []
    if not config.OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not config.OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    if not config.QDRANT_URL:
        missing.append("QDRANT_URL")
    if not config.QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")

    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    step = sys.argv[1] if len(sys.argv) > 1 else "all"

    if step == "chunk":
        step_chunk_and_embed()
    elif step == "generate":
        step_generate()
    elif step == "evaluate":
        step_evaluate()
    elif step == "all":
        step_chunk_and_embed()
        step_generate()
        step_evaluate()
    else:
        print(f"Usage: python rerun_semantic.py [chunk|generate|evaluate|all]")
