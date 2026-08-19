"""Claim-support verification.

generation.py's existing `verify_citations()` only checks that a cited
chunk_id EXISTS in the retrieved set. The P0 requirement goes further:

    "Verify that a citation does not only exist, but actually supports
    the generated claim."

This module adds that second, stronger check. It is deliberately kept
separate from generation.py so Workstream A can iterate on it without
touching the generation pipeline directly — wire it in with the one-line
change shown at the bottom of this file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Common English stopwords + a small Arabic stopword set. Kept local
# (no extra dependency) since this only needs to support word-overlap
# scoring, not full NLP.
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
    min_overlap_ratio: float = 0.35,
    min_matched_terms: int = 2,
) -> ClaimSupportResult:
    """Lexical-overlap grounding check between a claim and its cited chunk.

    This is a fast, dependency-free first-pass check: what fraction of the
    claim's meaningful (non-stopword) terms actually appear in the cited
    evidence chunk. It catches the most common failure mode — a citation
    that is topically related but does not actually contain the specific
    fact being claimed (wrong number, wrong drug, wrong population, etc).

    Not a substitute for human review, but a solid automated gate that
    directly feeds the Unsupported Claim Rate metric.

    Tune `min_overlap_ratio` / `min_matched_terms` against your own
    benchmark set — these defaults are a starting point, not a proven
    optimum.
    """
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
    """Optional stronger check: ask the LLM a narrow yes/no grounding
    question. Slower (one extra LLM call per citation) — use for the
    citations the lexical check marks as unsupported, as a second pass,
    rather than for every citation.
    """
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


# ---------------------------------------------------------------------------
# Integration point — drop-in replacement for generation.verify_citations
# ---------------------------------------------------------------------------
#
# In generation.py, replace the body of `verify_citations` with:
#
#   from .claim_verification import verify_claim_support
#
#   def verify_citations(citations, results):
#       retrieved_ids = {str(r.metadata.get("chunk_id", "")): r for r in results}
#       verified = []
#       for cit in citations:
#           r = retrieved_ids.get(cit.chunk_id)
#           if r is None:
#               cit.verified = False
#               verified.append(cit)
#               continue
#
#           # existence check (unchanged) + NEW: claim-support check
#           support = verify_claim_support(cit.claim, r.text)
#           cit.verified = support.supported
#
#           cit.document = str(r.metadata.get("document", cit.document))
#           cit.section = str(r.metadata.get("section", cit.section))
#           page_start = r.metadata.get("page_start", r.metadata.get("page", ""))
#           page_end = r.metadata.get("page_end", page_start)
#           cit.page = f"{page_start}" if page_start == page_end else f"{page_start}-{page_end}"
#           cit.score = r.score
#           verified.append(cit)
#       return verified
