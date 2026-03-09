import time
from typing import Optional
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from tqdm import tqdm

import config


class VectorStore:

    def __init__(self):
        self.openai = OpenAI(api_key=config.OPENAI_API_KEY)
        self.qdrant = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )

    def embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch = [t if t.strip() else " " for t in batch]
            response = self.openai.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            if i + batch_size < len(texts):
                time.sleep(0.1)
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    def collection_name(self, strategy_name: str) -> str:
        return f"{config.COLLECTION_PREFIX}_{strategy_name}"

    def create_collection(self, strategy_name: str, recreate: bool = False):
        name = self.collection_name(strategy_name)
        existing = [c.name for c in self.qdrant.get_collections().collections]

        if name in existing:
            if recreate:
                self.qdrant.delete_collection(name)
                print(f"  Deleted existing collection: {name}")
            else:
                print(f"  Collection already exists: {name}")
                return

        self.qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        from qdrant_client.models import PayloadSchemaType
        self.qdrant.create_payload_index(
            collection_name=name,
            field_name="dataset",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"  Created collection: {name}")

    def create_all_collections(self, recreate: bool = False):
        for strategy_name in config.CHUNKING_STRATEGIES:
            self.create_collection(strategy_name, recreate=recreate)

    def store_chunks(
        self,
        strategy_name: str,
        chunks: list[str],
        metadata_list: list[dict],
        batch_size: int = 100,
        start_id: int = 0,
    ) -> int:
        coll_name = self.collection_name(strategy_name)
        current_id = start_id

        for i in tqdm(
            range(0, len(chunks), batch_size),
            desc=f"  Storing {strategy_name}",
            leave=False,
        ):
            batch_chunks = chunks[i : i + batch_size]
            batch_meta = metadata_list[i : i + batch_size]

            embeddings = self.embed_texts(batch_chunks)

            points = []
            for j, (emb, text, meta) in enumerate(
                zip(embeddings, batch_chunks, batch_meta)
            ):
                payload = {
                    "text": text,
                    "dataset": meta.get("dataset", ""),
                    "doc_index": meta.get("doc_index", 0),
                    "chunk_index": meta.get("chunk_index", j),
                    "domain": meta.get("domain", ""),
                }
                points.append(
                    PointStruct(
                        id=current_id,
                        vector=emb,
                        payload=payload,
                    )
                )
                current_id += 1

            self.qdrant.upsert(collection_name=coll_name, points=points)

        return current_id

    def retrieve(
        self,
        query: str,
        strategy_name: str,
        dataset_filter: Optional[str] = None,
        top_k: int = config.TOP_K,
    ) -> list[dict]:
        query_embedding = self.embed_query(query)
        coll_name = self.collection_name(strategy_name)

        query_filter = None
        if dataset_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="dataset",
                        match=MatchValue(value=dataset_filter),
                    )
                ]
            )

        results = self.qdrant.query_points(
            collection_name=coll_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
        )

        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "dataset": hit.payload.get("dataset", ""),
                "domain": hit.payload.get("domain", ""),
                "doc_index": hit.payload.get("doc_index", 0),
                "chunk_index": hit.payload.get("chunk_index", 0),
            }
            for hit in results.points
        ]

    def get_collection_info(self, strategy_name: str) -> dict:
        name = self.collection_name(strategy_name)
        info = self.qdrant.get_collection(name)
        return {
            "name": name,
            "vectors_count": info.points_count,
            "status": str(info.status),
        }


if __name__ == "__main__":
    vs = VectorStore()
    vs.create_all_collections()
    for name in config.CHUNKING_STRATEGIES:
        info = vs.get_collection_info(name)
        print(f"  {info['name']}: {info['vectors_count']} vectors ({info['status']})")
