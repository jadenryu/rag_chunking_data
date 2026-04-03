import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

import config
from vector_store import VectorStore


class RAGPipeline:

    def __init__(self):
        self.vector_store = VectorStore()
        self.openrouter = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

    def build_prompt(self, query: str, chunks: list[dict]) -> str:
        context = "\n\n---\n\n".join(
            [f"[Chunk {i+1}]: {c['text']}" for i, c in enumerate(chunks)]
        )
        return config.RAG_PROMPT_TEMPLATE.format(query=query, context=context)

    def generate(self, prompt: str, llm_name: str, max_retries: int = 3) -> str:
        model_id = config.LLMS[llm_name]

        for attempt in range(max_retries):
            try:
                response = self.openrouter.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=1024,
                    timeout=60,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    time.sleep(wait)
                else:
                    return f"ERROR: {e}"

    def generate_parallel(self, prompt: str, llm_names: list[str]) -> dict[str, str]:
        responses = {}

        with ThreadPoolExecutor(max_workers=len(llm_names)) as executor:
            futures = {
                executor.submit(self.generate, prompt, name): name
                for name in llm_names
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    responses[name] = future.result()
                except Exception as e:
                    responses[name] = f"ERROR: {e}"

        return responses

    def classify_query_type(self, query: str) -> str:
        prompt = config.QUERY_CLASSIFICATION_PROMPT.format(query=query)
        response = self.generate(prompt, "gpt-4o-mini")
        response_lower = response.strip().lower()

        for qt in config.QUERY_TYPES:
            if qt in response_lower:
                return qt
        return "unanswerable"

    def process_single_query(
        self,
        record: dict,
        strategy_name: str,
        llm_names: Optional[list[str]] = None,
        top_k: int = config.TOP_K,
    ) -> dict:
        llms = llm_names or list(config.LLMS.keys())

        chunks = self.vector_store.retrieve(
            query=record["question"],
            strategy_name=strategy_name,
            dataset_filter=record["dataset"],
            top_k=top_k,
        )

        prompt = self.build_prompt(record["question"], chunks)

        responses = self.generate_parallel(prompt, llms)

        return {
            "question": record["question"],
            "ground_truth": record["answer"],
            "query_type": record["query_type"],
            "domain": record["domain"],
            "dataset": record["dataset"],
            "strategy": strategy_name,
            "top_k": top_k,
            "retrieved_chunks": [c["text"] for c in chunks],
            "chunk_scores": [c["score"] for c in chunks],
            "responses": responses,
        }

    def run_full_pipeline(
        self,
        records: list[dict],
        strategies: Optional[list[str]] = None,
        llm_names: Optional[list[str]] = None,
        resume_from: Optional[str] = None,
    ) -> list[dict]:
        target_strategies = strategies or list(config.CHUNKING_STRATEGIES.keys())

        all_results = []
        completed_keys = set()

        output_path = resume_from or os.path.join(
            config.RESULTS_DIR, "rag_results.json"
        )

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8", errors="replace") as f:
                all_results = json.load(f)
            for r in all_results:
                key = f"{r['dataset']}_{r['question'][:50]}_{r['strategy']}"
                completed_keys.add(key)
            print(f"Resuming from {len(all_results)} completed results")

        total = len(records) * len(target_strategies)
        pbar = tqdm(total=total, desc="RAG Pipeline")

        for strategy_name in target_strategies:
            print(f"\n--- Strategy: {strategy_name} ---")
            for record in records:
                key = f"{record['dataset']}_{record['question'][:50]}_{strategy_name}"
                if key in completed_keys:
                    pbar.update(1)
                    continue

                result = self.process_single_query(
                    record, strategy_name, llm_names=llm_names
                )
                all_results.append(result)

                if len(all_results) % 10 == 0:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)

                pbar.update(1)

        pbar.close()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(all_results)} results to {output_path}")

        return all_results


if __name__ == "__main__":
    from dataset_loader import load_processed_dataset

    pipeline = RAGPipeline()
    records = load_processed_dataset()
    print(f"Loaded {len(records)} records")

    test_records = records[:2]
    results = pipeline.run_full_pipeline(
        test_records,
        strategies=["fixed_256"],
        llm_names=["gpt-4o-mini"],
    )
    for r in results:
        print(f"\nQ: {r['question'][:80]}")
        print(f"A (ground truth): {r['ground_truth'][:80]}")
        for llm, resp in r["responses"].items():
            print(f"  {llm}: {resp[:100]}")
