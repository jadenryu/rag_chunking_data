import json
import os
import random
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

import config


def _classify_query_type_heuristic(question: str, answer: str, metadata: dict) -> str:
    if metadata.get("type") == "bridge":
        return "multi-hop"
    if metadata.get("type") == "comparison":
        return "comparative"

    if metadata.get("is_impossible", False):
        return "unanswerable"

    if metadata.get("final_decision") in ("yes", "no", "maybe"):
        return "single-hop"

    return "needs_classification"


def _load_hotpotqa(max_n: Optional[int] = None) -> list[dict]:
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    records = []
    for row in ds:
        context_parts = []
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            context_parts.append(f"{title}\n" + " ".join(sents))
        context = "\n\n".join(context_parts)

        records.append({
            "question": row["question"],
            "answer": row["answer"],
            "context": context,
            "query_type": _classify_query_type_heuristic(
                row["question"], row["answer"], {"type": row["type"]}
            ),
            "domain": "general",
            "dataset": "hotpotqa",
        })
        if max_n and len(records) >= max_n:
            break
    return records


def _load_naturalqa(max_n: Optional[int] = None) -> list[dict]:
    ds = load_dataset("sentence-transformers/natural-questions", split="train", streaming=True)
    records = []
    for row in ds:
        query = row.get("query", "")
        passage = row.get("answer", "")

        records.append({
            "question": query,
            "answer": passage.split(".")[0] if passage else "",
            "context": passage,
            "query_type": "single-hop",
            "domain": "general",
            "dataset": "naturalqa",
        })
        if max_n and len(records) >= max_n:
            break
    return records


def _load_squad2(max_n: Optional[int] = None) -> list[dict]:
    ds = load_dataset("squad_v2", split="validation")
    records = []
    for row in ds:
        is_impossible = len(row["answers"]["text"]) == 0
        answer = row["answers"]["text"][0] if not is_impossible else ""
        records.append({
            "question": row["question"],
            "answer": answer,
            "context": row["context"],
            "query_type": _classify_query_type_heuristic(
                row["question"], answer, {"is_impossible": is_impossible}
            ),
            "domain": "general",
            "dataset": "squad2",
        })
        if max_n and len(records) >= max_n:
            break
    return records


def _load_pubmedqa(max_n: Optional[int] = None) -> list[dict]:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    records = []
    for row in ds:
        context = "\n\n".join(row["context"]["contexts"]) if row.get("context") else ""
        records.append({
            "question": row["question"],
            "answer": row.get("long_answer", row.get("final_decision", "")),
            "context": context,
            "query_type": _classify_query_type_heuristic(
                row["question"], "", {"final_decision": row.get("final_decision", "")}
            ),
            "domain": "medical",
            "dataset": "pubmedqa",
        })
        if max_n and len(records) >= max_n:
            break
    return records


def _load_financeqa(max_n: Optional[int] = None) -> list[dict]:
    from huggingface_hub import hf_hub_download
    import json as _json

    import urllib.request
    import tempfile
    try:
        url = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json"
        with urllib.request.urlopen(url) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"WARNING: Could not load FinQA: {e}")
        print("  Skipping financeqa dataset.")
        return []

    records = []
    for row in data:
        context_parts = []
        if row.get("pre_text"):
            context_parts.extend(row["pre_text"])
        if row.get("post_text"):
            context_parts.extend(row["post_text"])
        if row.get("table"):
            table_rows = row["table"]
            table_text = "\n".join([" | ".join(str(c) for c in r) for r in table_rows])
            context_parts.append(f"Table:\n{table_text}")

        qa = row.get("qa", {})
        records.append({
            "question": qa.get("question", row.get("question", "")),
            "answer": str(qa.get("answer", row.get("answer", ""))),
            "context": "\n\n".join(context_parts),
            "query_type": "single-hop",
            "domain": "finance",
            "dataset": "financeqa",
        })
        if max_n and len(records) >= max_n:
            break
    return records


def _load_multihop_rag(max_n: Optional[int] = None) -> list[dict]:
    from huggingface_hub import hf_hub_download
    import json as _json

    try:
        path = hf_hub_download(
            repo_id="yixuantt/MultiHopRAG",
            filename="MultiHopRAG.json",
            repo_type="dataset",
        )
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
    except Exception as e:
        print(f"WARNING: MultiHop-RAG dataset not found: {e}")
        print("  Skipping this dataset.")
        return []

    records = []
    for row in data:
        evidence_list = row.get("evidence_list", [])
        context_parts = []
        for ev in evidence_list:
            title = ev.get("title", "")
            fact = ev.get("fact", "")
            if title and fact:
                context_parts.append(f"{title}\n{fact}")
            elif fact:
                context_parts.append(fact)
        context = "\n\n".join(context_parts)

        qt = row.get("question_type", "")
        if "inference" in qt or "bridge" in qt:
            query_type = "multi-hop"
        elif "comparison" in qt or "compar" in qt:
            query_type = "comparative"
        else:
            query_type = "multi-hop"

        records.append({
            "question": row.get("query", row.get("question", "")),
            "answer": row.get("answer", ""),
            "context": context,
            "query_type": query_type,
            "domain": "general",
            "dataset": "multihop_rag",
        })
        if max_n and len(records) >= max_n:
            break
    return records


def _load_pubmedqa_artificial(max_n: Optional[int] = None) -> list[dict]:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    records = []
    for row in ds:
        context = "\n\n".join(row["context"]["contexts"]) if row.get("context") else ""
        if not context.strip():
            continue
        records.append({
            "question": row["question"],
            "answer": row.get("long_answer", row.get("final_decision", "")),
            "context": context,
            "query_type": _classify_query_type_heuristic(
                row["question"], "", {"final_decision": row.get("final_decision", "")}
            ),
            "domain": "medical",
            "dataset": "pubmedqa_artificial",
        })
        if max_n and len(records) >= max_n:
            break
    return records


def _load_financebench(max_n: Optional[int] = None) -> list[dict]:
    try:
        ds = load_dataset("PatronusAI/financebench", split="train")
    except Exception as e:
        print(f"WARNING: Could not load FinanceBench: {e}")
        print("  Skipping financebench dataset.")
        return []

    records = []
    for row in ds:
        evidence = row.get("evidence", [])
        context = "\n\n".join(e["evidence_text"] for e in evidence if e.get("evidence_text", "").strip())
        if not context.strip():
            continue
        records.append({
            "question": row["question"],
            "answer": str(row.get("answer", "")),
            "context": context,
            "query_type": "single-hop",
            "domain": "finance",
            "dataset": "financebench",
        })
        if max_n and len(records) >= max_n:
            break
    return records


def _load_ragcareqa(max_n: Optional[int] = None) -> list[dict]:
    local_path = os.path.join(config.DATA_DIR, "ragcareqa.json")
    if not os.path.exists(local_path):
        print(f"WARNING: RAGCareQA not found at {local_path}")
        print("  Please download the dataset and save it as data/ragcareqa.json")
        print("  Expected format: [{\"question\": ..., \"answer\": ..., \"context\": ...}, ...]")
        print("  Skipping this dataset.")
        return []

    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for row in data:
        records.append({
            "question": row["question"],
            "answer": row.get("answer", ""),
            "context": row.get("context", ""),
            "query_type": "needs_classification",
            "domain": "medical",
            "dataset": "ragcareqa",
        })
        if max_n and len(records) >= max_n:
            break
    return records


LOADERS = {
    "hotpotqa": _load_hotpotqa,
    "naturalqa": _load_naturalqa,
    "squad2": _load_squad2,
    "pubmedqa": _load_pubmedqa,
    "pubmedqa_artificial": _load_pubmedqa_artificial,
    "financeqa": _load_financeqa,
    "financebench": _load_financebench,
    "multihop_rag": _load_multihop_rag,
    "ragcareqa": _load_ragcareqa,
}


def load_all_datasets(
    max_per_dataset: Optional[int] = None,
    datasets: Optional[list[str]] = None,
) -> list[dict]:
    max_n = max_per_dataset or config.MAX_QUERIES_PER_DATASET
    target_datasets = datasets or list(LOADERS.keys())
    all_records = []

    for name in target_datasets:
        if name not in LOADERS:
            print(f"WARNING: Unknown dataset '{name}', skipping.")
            continue
        try:
            print(f"Loading {name}...")
            records = LOADERS[name](max_n)
            print(f"  Loaded {len(records)} records from {name}")
            all_records.extend(records)
        except Exception as e:
            print(f"ERROR loading {name}: {e}")
            print(f"  Skipping {name}.")
    print(f"\nTotal records loaded: {len(all_records)}")
    return all_records


def save_processed_dataset(records: list[dict], filename: str = "processed_dataset.json"):
    path = os.path.join(config.DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} records to {path}")


def load_processed_dataset(filename: str = "processed_dataset.json") -> list[dict]:
    path = os.path.join(config.DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    records = load_all_datasets()
    save_processed_dataset(records)

    from collections import Counter
    domain_counts = Counter(r["domain"] for r in records)
    type_counts = Counter(r["query_type"] for r in records)
    dataset_counts = Counter(r["dataset"] for r in records)
    print(f"\nDomains: {dict(domain_counts)}")
    print(f"Query types: {dict(type_counts)}")
    print(f"Datasets: {dict(dataset_counts)}")
