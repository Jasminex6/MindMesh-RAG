"""Multi-question query decomposition for compound clinical queries.

Splits compound queries like:
  "What symptoms suggest asthma, how is it diagnosed, and what first-line controller is recommended?"
into independent retrieval intents:
  Q1: "What symptoms suggest asthma?"
  Q2: "How is asthma diagnosed?"
  Q3: "What is the first-line controller recommended for asthma?"

Each sub-question is retrieved and reranked independently. Results are merged
(deduplicated by chunk_id) before being passed to generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import SearchResult

# ── Compound query detection ──────────────────────────────────────────────────
# Patterns that indicate a query contains multiple distinct clinical questions.
_COMPOUND_CONJUNCTIONS = re.compile(
    r"\b(and|also|additionally|furthermore|moreover|plus|as well as|"
    r"and also|what about|how about|in addition)\b",
    re.IGNORECASE,
)

# Question-starters that signal a new sub-question within a compound query
_QUESTION_STARTERS = re.compile(
    r"\b(what|when|how|why|which|who|where|is|are|does|do|should|can|"
    r"could|would|will|may|might|must)\b",
    re.IGNORECASE,
)

# Separator patterns for splitting compound questions
_SPLIT_PATTERNS = [
    re.compile(r",\s*(?=(?:and\s+)?(?:what|when|how|why|which|who|where|is|are|does|do|should|can|could|would|will|may|might|must)\b)", re.IGNORECASE),
    re.compile(r"\?\s+(?=[A-Z\u0600-\u06FF])"),   # ? followed by new sentence
    re.compile(r"\band\s+(?=(?:what|when|how|why|which|who|where|is|are|does|do|should|can)\b)", re.IGNORECASE),
]

# Minimum length to treat a fragment as a real sub-question
_MIN_SUBQUERY_LEN = 12


@dataclass
class DecomposedQuery:
    """Result of decomposing a compound query."""
    original: str
    sub_questions: list[str]
    is_compound: bool
    compound_indicators: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.sub_questions)


def is_compound_query(query: str) -> bool:
    """Return True if the query likely contains multiple distinct clinical questions."""
    # Must be long enough to be compound
    if len(query.split()) < 8:
        return False

    # Must contain a compound conjunction
    if not _COMPOUND_CONJUNCTIONS.search(query):
        return False

    # Must contain at least 2 question-starter words
    starters = _QUESTION_STARTERS.findall(query)
    return len(starters) >= 2


def decompose_query(query: str) -> DecomposedQuery:
    """Decompose a compound clinical query into independent sub-questions.

    Args:
        query: Raw user query (may be compound or simple).

    Returns:
        DecomposedQuery with sub_questions list. If not compound, sub_questions
        contains only the original query.
    """
    query = query.strip()

    if not is_compound_query(query):
        return DecomposedQuery(
            original=query,
            sub_questions=[query],
            is_compound=False,
        )

    # Collect indicators
    indicators = [m.group(0) for m in _COMPOUND_CONJUNCTIONS.finditer(query)]

    # Try splitting on each pattern
    segments = _split_on_patterns(query)

    # Filter out fragments that are too short to be meaningful
    sub_questions = [s.strip(" .,;?") + "?" for s in segments if len(s.strip()) >= _MIN_SUBQUERY_LEN]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_subs = []
    for sq in sub_questions:
        key = sq.lower()
        if key not in seen:
            seen.add(key)
            unique_subs.append(sq)

    if len(unique_subs) < 2:
        # Decomposition failed — fall back to single query
        return DecomposedQuery(original=query, sub_questions=[query], is_compound=False)

    return DecomposedQuery(
        original=query,
        sub_questions=unique_subs,
        is_compound=True,
        compound_indicators=indicators,
    )


def _split_on_patterns(query: str) -> list[str]:
    """Apply split patterns in order; return segments."""
    segments = [query]
    for pattern in _SPLIT_PATTERNS:
        new_segments: list[str] = []
        for seg in segments:
            parts = pattern.split(seg)
            new_segments.extend(p.strip() for p in parts if p.strip())
        if len(new_segments) > len(segments):
            segments = new_segments

    return segments


@dataclass
class MultiQuestionResult:
    """Aggregated results from a multi-question retrieval."""
    original_query: str
    sub_questions: list[str]
    results_per_question: list[list[SearchResult]]
    merged_results: list[SearchResult]
    is_compound: bool


def retrieve_multi_question(
    query: str,
    retriever: Any,
    strategy: str = "hybrid_rerank",
    top_k_per_question: int = 5,
    final_top_k: int = 5,
    candidate_k: int = 20,
) -> MultiQuestionResult:
    """Decompose a query and retrieve evidence for each sub-question independently.

    Each sub-question is retrieved and reranked independently. Results are
    deduplicated by chunk_id and merged, with earlier sub-questions given
    priority in the final merged ranking.

    Args:
        query: Raw user query.
        retriever: A UnifiedRetriever (or MultilingualRetriever) instance.
        strategy: Retrieval strategy to use for each sub-question.
        top_k_per_question: How many results to retrieve per sub-question.
        final_top_k: How many to return in the merged result.
        candidate_k: Candidate pool size for hybrid/rerank strategies.

    Returns:
        MultiQuestionResult with per-question results and merged results.
    """
    decomposed = decompose_query(query)
    results_per_q: list[list[SearchResult]] = []

    for sq in decomposed.sub_questions:
        # Support both UnifiedRetriever and MultilingualRetriever
        if hasattr(retriever, "search"):
            raw = retriever.search(
                sq,
                strategy=strategy,
                top_k=top_k_per_question,
                candidate_k=candidate_k,
            )
            # MultilingualRetriever returns (results, lang_profile) tuple
            if isinstance(raw, tuple):
                sub_results = raw[0]
            else:
                sub_results = raw
        else:
            sub_results = []
        results_per_q.append(sub_results)

    merged = _merge_results(results_per_q, final_top_k)

    return MultiQuestionResult(
        original_query=query,
        sub_questions=decomposed.sub_questions,
        results_per_question=results_per_q,
        merged_results=merged,
        is_compound=decomposed.is_compound,
    )


def _merge_results(
    results_per_question: list[list[SearchResult]],
    final_top_k: int,
) -> list[SearchResult]:
    """Merge and deduplicate results from multiple sub-questions.

    Priority: earlier sub-questions take precedence in case of ties.
    Deduplication: by chunk_id (exact match) or text hash.
    """
    seen_ids: set[str] = set()
    merged: list[SearchResult] = []

    # Interleave results: take 1 from each sub-question at a time
    max_results = max((len(r) for r in results_per_question), default=0)
    for i in range(max_results):
        for sub_results in results_per_question:
            if i >= len(sub_results):
                continue
            result = sub_results[i]
            chunk_id = result.metadata.get("chunk_id", hash(result.text))
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(result)
            if len(merged) >= final_top_k:
                break
        if len(merged) >= final_top_k:
            break

    # Re-rank merged results
    return [
        SearchResult(rank=rank, score=r.score, text=r.text, metadata=r.metadata)
        for rank, r in enumerate(merged, start=1)
    ]
