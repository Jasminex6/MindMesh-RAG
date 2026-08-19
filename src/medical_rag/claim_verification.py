"""Claim-support verification.

Verify that a citation does not only exist in chunk IDs, but actually supports
the generated claim sentence via lexical-overlap or semantic grounding checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_STOPWORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "with", "as", "by",
    "this", "that", "these", "those", "it", "its", "should", "may",
    "can", "could", "will", "would", "has", "have", "had", "not", "no",
}
_STOPWORDS_AR = {
    "من", "في", "على", "الى", "إلى", "عن", "و", "أو", "او", "هذا", "هذه",
    "ذلك", "التي", "الذي", "لا", "لم", "قد", "كان", "يكون", "مع",
}
_STOPWORDS = _STOPWORDS_EN | _STOPWORDS_AR

_WORD_RE = re.compile(r"[A-Za-z\u0600-\u06FF]+", re.UNICODE)


def _content_words(text: str) -> set[str]:
    words = {w.lower() for w in _WORD_RE.findall(text or "")}
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


@dataclass
class ClaimSupportResult:
    supported: bool
    overlap_ratio: float
    matched_terms: set[str]


def verify_claim_support(
    claim: str,
    chunk_text: str,
    min_overlap_ratio: float = 0.25,
    min_matched_terms: int = 1,
) -> ClaimSupportResult:
    """Lexical-overlap grounding check between a claim and its cited chunk text."""
    claim_terms = _content_words(claim)
    chunk_terms = _content_words(chunk_text)

    if not claim_terms:
        return ClaimSupportResult(supported=False, overlap_ratio=0.0, matched_terms=set())

    matched = claim_terms & chunk_terms
    ratio = len(matched) / len(claim_terms)

    supported = ratio >= min_overlap_ratio and len(matched) >= min(
        min_matched_terms, len(claim_terms)
    )
    return ClaimSupportResult(supported=supported, overlap_ratio=ratio, matched_terms=matched)


def verify_claim_support_llm(
    claim: str,
    chunk_text: str,
    model: str = "llama3.2",
) -> ClaimSupportResult:
    """Optional LLM-based strict yes/no fact verifier."""
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=model, temperature=0.0)
    prompt = (
        "You are a strict fact-checker. Answer ONLY 'YES' or 'NO'.\n\n"
        f"Claim: {claim}\n\n"
        f"Evidence text: {chunk_text}\n\n"
        "Question: Does the evidence text explicitly support the claim? "
        "Answer YES only if the specific fact in the claim (number, drug, "
        "population, threshold, recommendation) is actually present in "
        "the evidence text, not just topically related."
    )
    response = llm.invoke([("human", prompt)])
    text = (response.content if hasattr(response, "content") else str(response)).strip().upper()
    supported = text.startswith("YES")
    return ClaimSupportResult(supported=supported, overlap_ratio=-1.0, matched_terms=set())
