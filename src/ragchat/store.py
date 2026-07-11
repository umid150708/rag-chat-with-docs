"""Chroma vector store wrapper: add chunks, search by vector.

We bring our own Gemini embeddings (Chroma's default embedder is a different
model), so the collection is created with no embedding function and we pass
vectors in explicitly.
"""

from __future__ import annotations

import chromadb

from .config import COLLECTION, STORE_DIR
from .models import Chunk


class VectorStore:
    def __init__(self) -> None:
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(STORE_DIR))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        # Stable id => re-ingesting the same file updates rather than duplicates.
        return f"{chunk.source}::{chunk.chunk_index}"

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[self._chunk_id(c) for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "chunk_index": c.chunk_index} for c in chunks
            ],
        )

    def query(self, embedding: list[float], top_k: int) -> list[Chunk]:
        res = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return [
            Chunk(text=doc, source=meta["source"], chunk_index=int(meta["chunk_index"]))
            for doc, meta in zip(docs, metas)
        ]

    def count(self) -> int:
        return self._collection.count()
