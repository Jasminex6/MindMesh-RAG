"""Lexical (BM25), Hybrid (RRF), and Two-Stage Reranking retrieval strategies."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

from .models import Chunk, SearchResult


ACRONYM_MAP = {
    "eib": "exercise-induced bronchoconstriction",
    "feno": "fractional exhaled nitric oxide",
    "ics": "inhaled corticosteroids",
    "saba": "short-acting beta2-agonist",
    "laba": "long-acting beta2-agonist",
    "aerd": "aspirin-exacerbated respiratory disease",
    "fev1": "forced expiratory volume in 1 second",
    "pef": "peak expiratory flow",
    "ocs": "oral corticosteroids",
}


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def expand_medical_query(query: str) -> str:
    """Expand medical acronyms into full terms for broader recall."""
    tokens = tokenize(query)
    added_terms = []
    for token in tokens:
        if token in ACRONYM_MAP and ACRONYM_MAP[token] not in query.lower():
            added_terms.append(ACRONYM_MAP[token])
    if added_terms:
        return f"{query} {' '.join(added_terms)}"
    return query


class BM25Retriever:
    """Okapi BM25 lexical retriever over a collection of Chunks."""

    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c.text) for c in self.chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.num_docs = len(self.chunks)
        self.avg_dl = sum(self.doc_lens) / max(self.num_docs, 1)

        # Compute document frequencies
        self.df: dict[str, int] = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        if n == 0:
            return 0.0
        return math.log((self.num_docs - n + 0.5) / (n + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5, min_score_threshold: float = 0.0) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: list[tuple[int, float]] = []
        for idx, tokens in enumerate(self.doc_tokens):
            if not tokens:
                continue
            doc_len = self.doc_lens[idx]
            tf_counter = Counter(tokens)
            score = 0.0
            for q_term in query_tokens:
                freq = tf_counter.get(q_term, 0)
                if freq == 0:
                    continue
                idf = self._idf(q_term)
                num = freq * (self.k1 + 1.0)
                den = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avg_dl, 1.0)))
                score += idf * (num / den)
            if score > min_score_threshold:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rank, (idx, score) in enumerate(scores[:top_k], start=1):
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(score),
                    text=chunk.text,
                    metadata=chunk.metadata(),
                )
            )
        return results


def reciprocal_rank_fusion(
    runs: list[list[SearchResult]], k_rrf: int = 60, top_k: int = 5
) -> list[SearchResult]:
    """Combine multiple ranked SearchResult lists using Reciprocal Rank Fusion (RRF)."""
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, SearchResult] = {}

    for run in runs:
        for result in run:
            chunk_id = str(result.metadata.get("chunk_id", hash(result.text)))
            chunk_map[chunk_id] = result
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k_rrf + result.rank))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    max_possible = len(runs) / (k_rrf + 1.0) if runs else 1.0
    fused_results = []
    for rank, cid in enumerate(sorted_ids[:top_k], start=1):
        original = chunk_map[cid]
        norm_score = min(1.0, float(rrf_scores[cid]) / max_possible) if max_possible > 0 else float(rrf_scores[cid])
        fused_results.append(
            SearchResult(
                rank=rank,
                score=norm_score,
                text=original.text,
                metadata=original.metadata,
            )
        )
    return fused_results


class CrossEncoderReranker:
    """Two-stage reranker refining candidate passages via cross-attention or term alignment."""

    def __init__(self, use_ml_model: bool = False):
        self.use_ml_model = use_ml_model
        self._model = None

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not candidates:
            return []

        q_terms = set(tokenize(query))
        scored_candidates = []

        for cand in candidates:
            cand_tokens = tokenize(cand.text)
            matches = sum(1 for t in q_terms if t in cand_tokens)
            overlap_ratio = matches / max(len(q_terms), 1)
            
            phrase_bonus = 0.35 if query.lower() in cand.text.lower() else 0.0
            orig_score_norm = min(1.0, max(0.0, cand.score))
            
            combined_score = (0.55 * overlap_ratio) + (0.25 * orig_score_norm) + phrase_bonus
            scored_candidates.append((cand, combined_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        reranked = []
        for rank, (cand, score) in enumerate(scored_candidates[:top_k], start=1):
            reranked.append(
                SearchResult(
                    rank=rank,
                    score=float(score),
                    text=cand.text,
                    metadata=cand.metadata,
                )
            )
        return reranked


class UnifiedRetriever:
    """Unified retrieval engine supporting Dense, BM25, Hybrid, Reranking, and Hybrid+Reranking strategies."""

    def __init__(self, vector_repository: Any, chunks: Iterable[Chunk]):
        self.vector_repository = vector_repository
        self.chunks = list(chunks)
        self.bm25 = BM25Retriever(self.chunks)
        self.reranker = CrossEncoderReranker()

    def search(
        self,
        query: str,
        strategy: str = "dense",
        top_k: int = 5,
        candidate_k: int = 15,
        min_score_threshold: float = 0.0,
        expand_acronyms: bool = True,
    ) -> list[SearchResult]:
        strategy = strategy.lower().strip()
        search_query = expand_medical_query(query) if expand_acronyms else query

        if strategy == "dense":
            results = self.vector_repository.search(search_query, top_k=top_k, min_score_threshold=min_score_threshold)

        elif strategy == "bm25":
            results = self.bm25.search(search_query, top_k=top_k, min_score_threshold=min_score_threshold)

        elif strategy == "hybrid":
            dense_runs = self.vector_repository.search(search_query, top_k=candidate_k, min_score_threshold=min_score_threshold)
            bm25_runs = self.bm25.search(search_query, top_k=candidate_k, min_score_threshold=min_score_threshold)
            results = reciprocal_rank_fusion([dense_runs, bm25_runs], top_k=top_k)

        elif strategy in ("rerank", "dense_rerank"):
            dense_candidates = self.vector_repository.search(search_query, top_k=candidate_k, min_score_threshold=min_score_threshold)
            results = self.reranker.rerank(search_query, dense_candidates, top_k=top_k)

        elif strategy in ("hybrid_rerank", "hybrid+rerank"):
            dense_runs = self.vector_repository.search(search_query, top_k=candidate_k, min_score_threshold=min_score_threshold)
            bm25_runs = self.bm25.search(search_query, top_k=candidate_k, min_score_threshold=min_score_threshold)
            hybrid_candidates = reciprocal_rank_fusion([dense_runs, bm25_runs], top_k=candidate_k)
            results = self.reranker.rerank(search_query, hybrid_candidates, top_k=top_k)

        else:
            raise ValueError(f"Unknown retrieval strategy: '{strategy}'. Supported: dense, bm25, hybrid, rerank, hybrid_rerank")

        if min_score_threshold > 0.0 and results:
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
