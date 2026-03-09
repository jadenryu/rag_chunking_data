import argparse
import json
import os
import sys

import config
from chunking import apply_chunking, count_tokens
from dataset_loader import (
    load_all_datasets,
    load_processed_dataset,
    save_processed_dataset,
)
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from evaluation import evaluate_all, save_evaluation, load_evaluation
from analysis import run_all_analysis


def step_load_datasets(args):
    print("=" * 60)
    print("STEP 1: Loading Datasets")
    print("=" * 60)

    records = load_all_datasets(max_per_dataset=args.max_queries)
    save_processed_dataset(records)
    print(f"\nLoaded {len(records)} total records")
    return records


def step_classify_query_types(args):
    print("=" * 60)
    print("STEP 1b: Classifying Query Types")
    print("=" * 60)

    records = load_processed_dataset()
    pipeline = RAGPipeline()
    needs_classification = [r for r in records if r["query_type"] == "needs_classification"]

    print(f"{len(needs_classification)} records need query type classification")

    for i, record in enumerate(needs_classification):
        query_type = pipeline.classify_query_type(record["question"])
        record["query_type"] = query_type
        if (i + 1) % 50 == 0:
            print(f"  Classified {i+1}/{len(needs_classification)}")

    save_processed_dataset(records)
    print("Query type classification complete")


def step_chunk_and_embed(args):
    print("=" * 60)
    print("STEP 2+3: Chunking and Embedding")
    print("=" * 60)

    records = load_processed_dataset()
    vs = VectorStore()

    print("\nCreating Qdrant collections...")
    vs.create_all_collections(recreate=args.recreate)

    from collections import defaultdict
    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)

    for strategy_name, strategy_config in config.CHUNKING_STRATEGIES.items():
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


def step_generate(args):
    print("=" * 60)
    print("STEP 4: RAG Generation")
    print("=" * 60)

    records = load_processed_dataset()
    records = [r for r in records if r.get("context")]

    pipeline = RAGPipeline()
    results_path = os.path.join(config.RESULTS_DIR, "rag_results.json")

    strategies = args.strategies.split(",") if args.strategies else None
    llms = args.llms.split(",") if args.llms else None

    results = pipeline.run_full_pipeline(
        records,
        strategies=strategies,
        llm_names=llms,
        resume_from=results_path if args.resume else None,
    )

    print(f"\nGeneration complete: {len(results)} results")


def step_evaluate(args):
    print("=" * 60)
    print("STEP 5: Evaluation")
    print("=" * 60)

    results_path = os.path.join(config.RESULTS_DIR, "rag_results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    evaluated = evaluate_all(results)
    save_evaluation(evaluated)
    print(f"\nEvaluation complete: {len(evaluated)} results")


def step_analyze(args):
    print("=" * 60)
    print("STEP 6: Analysis and Visualization")
    print("=" * 60)
    run_all_analysis()


def run_full_pipeline(args):
    step_load_datasets(args)
    step_classify_query_types(args)
    step_chunk_and_embed(args)
    step_generate(args)
    step_evaluate(args)
    step_analyze(args)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Research Pipeline - Chunking Strategy Analysis"
    )
    parser.add_argument(
        "--step",
        choices=["load", "classify", "chunk", "generate", "evaluate", "analyze", "all"],
        default="all",
        help="Which pipeline step to run",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=config.MAX_QUERIES_PER_DATASET,
        help="Max queries per dataset",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated chunking strategies (e.g., 'fixed_128,fixed_256')",
    )
    parser.add_argument(
        "--llms",
        type=str,
        default=None,
        help="Comma-separated LLM names (e.g., 'gpt-4,claude-3.5-sonnet')",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved progress",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collections (delete existing)",
    )

    args = parser.parse_args()

    missing = []
    if not config.OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not config.OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    if not config.QDRANT_URL:
        missing.append("QDRANT_URL")
    if not config.QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")

    if missing and args.step not in ["analyze"]:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Please set them in your .env file")
        sys.exit(1)

    steps = {
        "load": step_load_datasets,
        "classify": step_classify_query_types,
        "chunk": step_chunk_and_embed,
        "generate": step_generate,
        "evaluate": step_evaluate,
        "analyze": step_analyze,
        "all": run_full_pipeline,
    }

    steps[args.step](args)


if __name__ == "__main__":
    main()
