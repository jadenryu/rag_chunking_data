import re
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def tokenize(text: str) -> list[int]:
    return _enc.encode(text)


def detokenize(tokens: list[int]) -> str:
    return _enc.decode(tokens)


def fixed_chunk(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    tokens = tokenize(text)
    if not tokens:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = detokenize(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def _split_into_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(text: str, max_tokens: int) -> list[str]:
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks = []
    current_parts = []
    current_token_count = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if current_token_count + para_tokens <= max_tokens:
            current_parts.append(para)
            current_token_count += para_tokens
            continue

        if current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_token_count = 0

        if para_tokens > max_tokens:
            sentences = _split_into_sentences(para)
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if current_token_count + sent_tokens <= max_tokens:
                    current_parts.append(sent)
                    current_token_count += sent_tokens
                else:
                    if current_parts:
                        chunks.append(" ".join(current_parts))
                        current_parts = []
                        current_token_count = 0
                    if sent_tokens > max_tokens:
                        chunks.extend(fixed_chunk(sent, max_tokens))
                    else:
                        current_parts.append(sent)
                        current_token_count = sent_tokens
        else:
            current_parts.append(para)
            current_token_count = para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def apply_chunking(text: str, strategy_name: str, strategy_config: dict) -> list[str]:
    if strategy_config["type"] == "fixed":
        return fixed_chunk(
            text,
            chunk_size=strategy_config["chunk_size"],
            overlap=strategy_config.get("overlap", 0),
        )
    elif strategy_config["type"] == "semantic":
        return semantic_chunk(text, max_tokens=strategy_config["max_tokens"])
    else:
        raise ValueError(f"Unknown chunking type: {strategy_config['type']}")


if __name__ == "__main__":
    sample = (
        "This is the first paragraph with some content. It has multiple sentences. "
        "Each sentence adds information.\n\n"
        "This is the second paragraph. It discusses a different topic entirely. "
        "More details are provided here.\n\n"
        "A third paragraph follows. Short one."
    )
    from config import CHUNKING_STRATEGIES

    for name, cfg in CHUNKING_STRATEGIES.items():
        chunks = apply_chunking(sample, name, cfg)
        print(f"\n{'='*60}")
        print(f"Strategy: {name} -> {len(chunks)} chunks")
        for i, c in enumerate(chunks):
            print(f"  [{i}] ({count_tokens(c)} tokens): {c[:80]}...")
