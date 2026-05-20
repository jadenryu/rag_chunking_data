import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

COLLECTION_PREFIX = "rag_research"

CHUNKING_STRATEGIES = {
    "fixed_128":    {"type": "fixed",    "chunk_size": 128, "overlap": 0},
    "fixed_256":    {"type": "fixed",    "chunk_size": 256, "overlap": 0},
    "fixed_512":    {"type": "fixed",    "chunk_size": 512, "overlap": 0},
    "semantic_128": {"type": "semantic", "max_tokens": 128},
    "semantic_256": {"type": "semantic", "max_tokens": 256},
    "semantic_512": {"type": "semantic", "max_tokens": 512},
}

DATASETS = {
    "hotpotqa": {
        "hf_path": "hotpot_qa",
        "hf_name": "distractor",
        "split": "validation",
        "domain": "general",
    },
    "naturalqa": {
        "hf_path": "nq_open",
        "hf_name": None,
        "split": "validation",
        "domain": "general",
    },
    "squad2": {
        "hf_path": "rajpurkar/SQuAD-v2.0",
        "hf_name": None,
        "split": "validation",
        "domain": "general",
    },
    "pubmedqa": {
        "hf_path": "qiaojin/PubMedQA",
        "hf_name": "pqa_labeled",
        "split": "train",
        "domain": "medical",
    },
    "financeqa": {
        "hf_path": "ibm/finqa",
        "hf_name": None,
        "split": "validation",
        "domain": "finance",
    },
    "pubmedqa_artificial": {
        "hf_path": "qiaojin/PubMedQA",
        "hf_name": "pqa_artificial",
        "split": "train",
        "domain": "medical",
    },
    "financebench": {
        "hf_path": "PatronusAI/financebench",
        "hf_name": None,
        "split": "train",
        "domain": "finance",
    },
    "multihop_rag": {
        "hf_path": "yixuantt/MultiHop-RAG",
        "hf_name": None,
        "split": "train",
        "domain": "general",
    },
    "ragcareqa": {
        "hf_path": None,
        "hf_name": None,
        "split": "test",
        "domain": "medical",
    },
}

QUERY_TYPES = ["single-hop", "multi-hop", "comparative", "unanswerable"]

QUERY_CLASSIFICATION_PROMPT = (
    "Classify the following question into exactly one of these query types: "
    "single-hop, multi-hop, comparative, unanswerable. "
    "A single-hop question requires one fact. "
    "A multi-hop question requires combining facts from multiple sources. "
    "A comparative question requires comparing two or more entities. "
    "An unanswerable question cannot be answered from typical reference material. "
    "Respond with ONLY the query type label.\n\n"
    "Question: {query}"
)

LLMS = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "mistral-large": "mistralai/mistral-large",
}

TOP_K = 5

RAG_PROMPT_TEMPLATE = (
    "Based on the context below, answer this question: {query}\n\n"
    "Context:\n{context}"
)

MAX_QUERIES_PER_DATASET = 50

JUDGE_MODEL = "openai/gpt-4o-mini"
JUDGE_CONCURRENCY = 15
JUDGE_MAX_CONTEXT_CHARS = 6000

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)
