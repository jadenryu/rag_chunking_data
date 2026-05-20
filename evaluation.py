import json
import os
import re
from collections import Counter
from typing import Optional

from datasets import Dataset
from tqdm import tqdm

import config


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    truth_tokens = _normalize_text(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_context_recall(ground_truth: str, contexts: list[str]) -> float:
    truth_tokens = set(_normalize_text(ground_truth).split())
    if not truth_tokens:
        return 0.0
    context_text = " ".join(contexts)
    context_tokens = set(_normalize_text(context_text).split())
    overlap = truth_tokens & context_tokens
    return len(overlap) / len(truth_tokens)


def compute_context_precision(ground_truth: str, contexts: list[str]) -> float:
    truth_tokens = set(_normalize_text(ground_truth).split())
    if not truth_tokens or not contexts:
        return 0.0

    relevant_count = 0
    precision_sum = 0.0
    for i, ctx in enumerate(contexts):
        ctx_tokens = set(_normalize_text(ctx).split())
        if truth_tokens & ctx_tokens:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)

    if relevant_count == 0:
        return 0.0
    return precision_sum / relevant_count


def compute_answer_relevance(question: str, answer: str) -> float:
    q_tokens = set(_normalize_text(question).split())
    a_tokens = set(_normalize_text(answer).split())

    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "being", "have", "has", "had", "do", "does", "did", "will",
                  "would", "could", "should", "may", "might", "shall", "can",
                  "to", "of", "in", "for", "on", "with", "at", "by", "from",
                  "as", "into", "through", "during", "before", "after", "and",
                  "but", "or", "not", "no", "this", "that", "these", "those",
                  "it", "its", "what", "which", "who", "whom", "how", "when",
                  "where", "why", "if", "then", "than", "so", "very", "just",
                  "about", "above", "also", "both", "each", "other", "some",
                  "such", "more", "most", "own", "same", "based", "answer",
                  "question", "context", "following", "below"}

    q_tokens -= stop_words
    a_tokens -= stop_words

    if not q_tokens or not a_tokens:
        return 0.5

    overlap = q_tokens & a_tokens
    return min(1.0, len(overlap) / max(len(q_tokens), 1) + 0.3)


def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    a_tokens = set(_normalize_text(answer).split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "have", "has", "had", "do", "does", "did", "will", "would",
                  "to", "of", "in", "for", "on", "with", "at", "by", "from",
                  "as", "and", "but", "or", "not", "this", "that", "it", "its"}
    a_tokens -= stop_words

    if not a_tokens:
        return 0.0

    context_text = " ".join(contexts)
    c_tokens = set(_normalize_text(context_text).split())
    grounded = a_tokens & c_tokens
    return len(grounded) / len(a_tokens)


BALANCED_DATASETS = {
    "hotpotqa", "squad2",
    "pubmedqa", "pubmedqa_artificial",
    "financeqa", "financebench",
}


def _try_ragas_evaluate(results: list[dict], llm_name: str) -> Optional[list[dict]]:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import AnswerRelevancy, Faithfulness, ContextPrecision, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    lc_llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    lc_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=config.OPENAI_API_KEY)
    ragas_llm = LangchainLLMWrapper(lc_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

    questions, answers, contexts, ground_truths = [], [], [], []
    for r in results:
        if llm_name not in r["responses"]:
            continue
        response = r["responses"][llm_name]
        if response.startswith("ERROR:"):
            continue
        questions.append(r["question"])
        answers.append(response)
        contexts.append(r["retrieved_chunks"])
        ground_truths.append(r["ground_truth"])

    if not questions:
        return None

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    from ragas import RunConfig
    run_config = RunConfig(max_workers=10, timeout=120, max_retries=5)

    ragas_result = ragas_evaluate(
        ds,
        metrics=[
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            Faithfulness(llm=ragas_llm),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ],
        run_config=run_config,
        raise_exceptions=False,
    )
    return ragas_result.to_pandas().to_dict("records")


def evaluate_with_metrics(results: list[dict], llm_name: str) -> list[dict]:
    print(f"  Running RAGAS for {llm_name}...")
    ragas_results = _try_ragas_evaluate(results, llm_name)
    if ragas_results:
        print(f"  RAGAS succeeded: {len(ragas_results)} records")
    else:
        print(f"  RAGAS returned no results — falling back to manual metrics")

    evaluated = []
    idx = 0
    for r in results:
        if llm_name not in r["responses"]:
            continue
        response = r["responses"][llm_name]
        if response.startswith("ERROR:"):
            continue

        metrics = {
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "response": response,
            "query_type": r["query_type"],
            "domain": r["domain"],
            "dataset": r["dataset"],
            "strategy": r["strategy"],
            "llm": llm_name,
            "token_f1": compute_f1(response, r["ground_truth"]),
            "f1_score": None,  # Populated by llm_judge.py
        }

        if ragas_results and idx < len(ragas_results):
            row = ragas_results[idx]
            metrics["answer_relevancy"] = float(row.get("answer_relevancy", 0))
            metrics["faithfulness"] = float(row.get("faithfulness", 0))
            metrics["context_precision"] = float(row.get("context_precision", 0))
            metrics["context_recall"] = float(row.get("context_recall", 0))
        else:
            metrics["answer_relevancy"] = compute_answer_relevance(
                r["question"], response
            )
            metrics["faithfulness"] = compute_faithfulness(
                response, r["retrieved_chunks"]
            )
            metrics["context_precision"] = compute_context_precision(
                r["ground_truth"], r["retrieved_chunks"]
            )
            metrics["context_recall"] = compute_context_recall(
                r["ground_truth"], r["retrieved_chunks"]
            )

        evaluated.append(metrics)
        idx += 1

    return evaluated


def evaluate_all(results: list[dict]) -> list[dict]:
    import os
    checkpoint_dir = os.path.join(config.RESULTS_DIR, "ragas_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_evaluated = []
    llm_names = set()

    results = [r for r in results if r.get("dataset") in BALANCED_DATASETS]
    print(f"Filtered to {len(results)} results across balanced datasets")

    for r in results:
        llm_names.update(r["responses"].keys())

    for llm_name in sorted(llm_names):
        checkpoint_path = os.path.join(checkpoint_dir, f"{llm_name}.json")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                evaluated = json.load(f)
            print(f"\nLoaded checkpoint for {llm_name}: {len(evaluated)} results")
        else:
            print(f"\nEvaluating {llm_name}...")
            evaluated = evaluate_with_metrics(results, llm_name)
            with open(checkpoint_path, "w") as f:
                json.dump(evaluated, f)
            print(f"  {len(evaluated)} results evaluated for {llm_name}")

        all_evaluated.extend(evaluated)

    return all_evaluated


def save_evaluation(evaluated: list[dict], filename: str = "evaluation_results.json"):
    path = os.path.join(config.RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(evaluated, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(evaluated)} evaluation results to {path}")


def load_evaluation(filename: str = "evaluation_results.json") -> list[dict]:
    path = os.path.join(config.RESULTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    results_path = os.path.join(config.RESULTS_DIR, "rag_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        evaluated = evaluate_all(results)
        save_evaluation(evaluated)

        from collections import defaultdict
        by_strategy = defaultdict(list)
        for e in evaluated:
            by_strategy[e["strategy"]].append(e.get("f1_score", 0))

        print("\n--- F1 by Strategy ---")
        for s, scores in sorted(by_strategy.items()):
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  {s}: {avg:.4f} (n={len(scores)})")
    else:
        print("No results found. Run the RAG pipeline first.")
