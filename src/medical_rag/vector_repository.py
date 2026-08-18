"""Chroma adapter; the only module that owns vector-store infrastructure."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .models import Chunk, SearchResult


class ChromaVectorRepository:
    """Persist chunks and run cosine similarity search with one embedding provider."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embedding_function: Any,
    ):
        from langchain_chroma import Chroma

        persist_directory.mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=str(persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: Iterable[Chunk], batch_size: int = 32) -> int:
        from langchain_core.documents import Document

        chunk_list = list(chunks)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(chunk_list), batch_size):
            batch = chunk_list[start : start + batch_size]
            documents = [
                Document(page_content=chunk.text, metadata=chunk.metadata())
                for chunk in batch
            ]
            self._store.add_documents(
                documents=documents,
                ids=[chunk.chunk_id for chunk in batch],
            )
        return len(chunk_list)

    def count(self) -> int:
        return len(self._store.get(include=[])["ids"])

    def search(
        self, query: str, top_k: int = 5, min_score_threshold: float = 0.0
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        matches = self._store.similarity_search_with_relevance_scores(query, k=top_k)
        results = [
            SearchResult(
                rank=rank,
                score=float(score),
                text=document.page_content,
                metadata=dict(document.metadata),
            )
            for rank, (document, score) in enumerate(matches, start=1)
        ]

        if min_score_threshold > 0.0:
            filtered = [r for r in results if r.score >= min_score_threshold]
            results = [
                SearchResult(
                    rank=new_rank,
                    score=r.score,
                    text=r.text,
                    metadata=r.metadata,
                )
                for new_rank, r in enumerate(filtered, start=1)
            ]

        return results
