"""
LLM-as-Judge evaluation for RAG pipeline responses.

Replaces token-level F1 with semantic correctness scoring using GPT-4o-mini.
Based on:
  - Zheng et al. (2023) "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
  - Liu et al. (2023) "G-Eval: NLG Evaluation using GPT-4 with Chain-of-Thought"
  - Saad-Falcon et al. (2023) "ARES: Automated Evaluation Framework for RAG"

Uses a 1-5 Likert scale (normalized to 0-1) for answer correctness,
with structured rubric criteria for consistent grading.
"""

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Optional

from openai import AsyncOpenAI
from tqdm import tqdm

import config

# ── Judge prompt (G-Eval style with rubric) ──────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator assessing the correctness of a RAG system's answer.

## Task
Rate how correctly the RESPONSE answers the QUESTION, compared to the REFERENCE ANSWER.

## Inputs
- QUESTION: {question}
- REFERENCE ANSWER: {ground_truth}
- RESPONSE: {response}

## Scoring Rubric (1-5)
5 = The response contains the correct answer with the same essential information as the reference. Minor wording differences or additional correct elaboration are acceptable.
4 = The response is correct but misses minor details from the reference, or adds minor inaccuracies alongside the correct answer.
3 = Partially correct. Captures some key information but misses important details or includes significant inaccuracies.
2 = Mostly incorrect. Attempts to answer but misses the main point or contradicts the reference on key facts.
1 = Completely incorrect, does not answer the question, or contradicts the reference answer entirely.

## Important
- Focus on SEMANTIC correctness, not lexical overlap.
- A verbose response that contains the correct answer should score 5.
- Short reference answers (e.g. "yes", "no", a name, a date): the response must contain or clearly convey that answer.
- If the question is unanswerable and the reference indicates this, a response that appropriately declines should score highly.

Respond with ONLY valid JSON: {{"score": <int 1-5>, "reasoning": "<one sentence>"}}"""


def _record_key(record: dict) -> str:
    raw = f"{record['question']}|{record['strategy']}|{record['llm']}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def judge_single(
    client: AsyncOpenAI,
    record: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 4,
) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=record["question"],
        ground_truth=record["ground_truth"],
        response=record["response"][:3000],
    )

    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=config.JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=150,
                )
            text = resp.choices[0].message.content.strip()

            # Parse JSON response
            try:
                parsed = json.loads(text)
                raw_score = int(parsed["score"])
            except (json.JSONDecodeError, KeyError, ValueError):
                # Fallback: extract first integer 1-5
                match = re.search(r'\b([1-5])\b', text)
                if match:
                    raw_score = int(match.group(1))
                    parsed = {"score": raw_score, "reasoning": text}
                else:
                    raise ValueError(f"Could not parse score from: {text}")

            raw_score = max(1, min(5, raw_score))
            normalized = (raw_score - 1) / 4.0

            return {
                "key": _record_key(record),
                "score": normalized,
                "raw_score": raw_score,
                "reasoning": parsed.get("reasoning", ""),
            }

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                await asyncio.sleep(wait)
            else:
                return {
                    "key": _record_key(record),
                    "score": None,
                    "raw_score": None,
                    "reasoning": f"ERROR: {e}",
                }


def _load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"model": config.JUDGE_MODEL, "completed": {}}


def _save_checkpoint(checkpoint: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False)
    os.replace(tmp, path)


async def run_judge(eval_file: str, checkpoint_file: str):
    eval_path = os.path.join(config.RESULTS_DIR, eval_file)
    ckpt_path = os.path.join(config.RESULTS_DIR, checkpoint_file)

    with open(eval_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    checkpoint = _load_checkpoint(ckpt_path)
    completed = checkpoint["completed"]

    # Filter to records not yet judged
    todo = []
    for r in records:
        key = _record_key(r)
        if key not in completed:
            todo.append(r)

    print(f"Loaded {len(records)} records, {len(completed)} already judged, {len(todo)} remaining")

    if not todo:
        print("All records already judged.")
        return checkpoint

    client = AsyncOpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    semaphore = asyncio.Semaphore(config.JUDGE_CONCURRENCY)

    # Process in batches for incremental saving
    batch_size = 100
    pbar = tqdm(total=len(todo), desc="Judging")

    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        tasks = [judge_single(client, r, semaphore) for r in batch]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result["score"] is not None:
                completed[result["key"]] = {
                    "score": result["score"],
                    "raw_score": result["raw_score"],
                    "reasoning": result["reasoning"],
                }
            pbar.update(1)

        _save_checkpoint(checkpoint, ckpt_path)

    pbar.close()
    print(f"Judging complete. {len(completed)} total records scored.")
    return checkpoint


def merge_scores(eval_file: str, checkpoint_file: str, output_file: str):
    eval_path = os.path.join(config.RESULTS_DIR, eval_file)
    ckpt_path = os.path.join(config.RESULTS_DIR, checkpoint_file)
    out_path = os.path.join(config.RESULTS_DIR, output_file)

    with open(eval_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    checkpoint = _load_checkpoint(ckpt_path)
    completed = checkpoint["completed"]

    merged = []
    scored, missing = 0, 0
    for r in records:
        key = _record_key(r)
        # Rename old f1_score to token_f1
        r["token_f1"] = r.pop("f1_score", 0.0)

        if key in completed:
            r["f1_score"] = completed[key]["score"]
            r["judge_raw_score"] = completed[key]["raw_score"]
            r["judge_reasoning"] = completed[key]["reasoning"]
            scored += 1
        else:
            r["f1_score"] = None
            r["judge_raw_score"] = None
            r["judge_reasoning"] = ""
            missing += 1

        merged.append(r)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Merged: {scored} scored, {missing} missing -> {out_path}")

    # Also export CSV for analysis/visualizations
    import pandas as pd
    df = pd.DataFrame(merged)
    csv_name = output_file.replace(".json", ".csv")
    csv_path = os.path.join(config.RESULTS_DIR, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"CSV exported -> {csv_path}")

    return merged


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation")
    parser.add_argument(
        "--dataset",
        choices=["fixed", "semantic", "both"],
        default="both",
        help="Which evaluation set to judge",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip judging, just merge existing checkpoint scores",
    )
    args = parser.parse_args()

    datasets = {
        "fixed": {
            "eval": "evaluation_results.json",
            "checkpoint": "judge_checkpoint.json",
            "output": "evaluation_results_judged.json",
        },
        "semantic": {
            "eval": "evaluation_results_lc_semantic.json",
            "checkpoint": "judge_checkpoint_semantic.json",
            "output": "evaluation_results_lc_semantic_judged.json",
        },
    }

    targets = list(datasets.keys()) if args.dataset == "both" else [args.dataset]

    for name in targets:
        d = datasets[name]
        print(f"\n{'='*60}")
        print(f"Processing: {name} ({d['eval']})")
        print(f"{'='*60}")

        if not args.merge_only:
            await run_judge(d["eval"], d["checkpoint"])

        merge_scores(d["eval"], d["checkpoint"], d["output"])


if __name__ == "__main__":
    asyncio.run(main())
