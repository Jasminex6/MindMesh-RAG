"""Multilingual retrieval support — Arabic cross-lingual retrieval against English evidence.

Strategy:
  Arabic query → normalize (query_normalization) → preserve clinical terms (ICS, SABA, etc.)
  → embed with existing nomic-embed-text (English vector space)
  → retrieve English WHO/NICE evidence directly.

No generic translator in the main path. Clinical acronyms and drug names are never translated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import SearchResult
from .query_normalization import normalize_medical_query, has_arabic_content

# ── Arabic clinical term map ──────────────────────────────────────────────────
# Maps Arabic clinical phrases to their English equivalents for query enhancement.
# These are appended to the query so both BM25 and dense retrieval benefit.
_AR_EN_CLINICAL_MAP: dict[str, str] = {
    # Therapy terms
    "\u0627\u0644\u0631\u0628\u0648\u064a\u0629": "bronchodilator",
    "\u0644\u0644\u0631\u0628\u0648": "for asthma",
    "\u0627\u0644\u0631\u0628\u0648 \u0627\u0644\u0634\u0639\u0628\u064a": "bronchial asthma",
    "\u062a\u0636\u064a\u0642 \u0627\u0644\u0634\u0639\u0628": "bronchoconstriction",
    "\u0645\u0648\u0633\u0639\u0627\u062a \u0627\u0644\u0634\u0639\u0628": "bronchodilators",
    "\u0627\u0644\u062a\u0647\u0627\u0628 \u0627\u0644\u0634\u0639\u0628\u064a": "airway inflammation",
    "\u0645\u0633\u062a\u0646\u0634\u0642\u0627\u062a \u0627\u0644\u0643\u0648\u0631\u062a\u064a\u0643\u0648\u0633\u062a\u064a\u0631\u0648\u064a\u062f": "inhaled corticosteroids ICS",
    # Symptoms
    "\u0635\u0641\u064a\u0631": "wheeze wheezing",
    "\u0636\u064a\u0642 \u0627\u0644\u062a\u0646\u0641\u0633": "shortness of breath dyspnea",
    "\u0633\u0639\u0627\u0644": "cough",
    "\u0623\u0632\u0645\u0629": "asthma exacerbation",
    "\u0646\u0648\u0628\u0629 \u0631\u0628\u0648": "asthma attack",
    # Management
    "\u0639\u0644\u0627\u062c": "treatment management",
    "\u062a\u0635\u0639\u064a\u062f": "escalation step up",
    "\u062a\u062e\u0641\u064a\u0641": "step down reduction",
    "\u0637\u0648\u0627\u0631\u0626": "emergency",
    "\u062d\u0627\u062f\u0629": "acute severe",
    # Patients
    "\u0637\u0641\u0644": "child children paediatric",
    "\u0623\u0637\u0641\u0627\u0644": "children paediatric",
    "\u0645\u0631\u0627\u0647\u0642": "adolescent",
    "\u0628\u0627\u0644\u063a": "adult",
    # Diagnostics
    "\u062a\u0634\u062e\u064a\u0635": "diagnosis",
    "\u0642\u064a\u0627\u0633 \u0627\u0644\u062a\u062f\u0641\u0642": "spirometry",
    "\u0627\u062e\u062a\u0628\u0627\u0631": "test",
    # Drug names in Arabic
    "\u0633\u0627\u0644\u0628\u0648\u062a\u0627\u0645\u0648\u0644": "salbutamol",
    "\u0628\u0631\u064a\u062f\u0646\u064a\u0632\u0648\u0644\u0648\u0646": "budesonide",
    "\u0641\u0644\u0648\u062a\u064a\u0643\u0627\u0632\u0648\u0646": "fluticasone",
    "\u0643\u0644\u0648\u0628\u064a\u062f\u0648\u062c\u0631\u064a\u0644": "clopidogrel",
    "\u0645\u0648\u0646\u062a\u064a\u0644\u0648\u0643\u0627\u0633\u062a": "montelukast",
    "\u0633\u0648\u0644\u0641\u0627\u062a \u0627\u0644\u0645\u063a\u0646\u064a\u0633\u064a\u0648\u0645": "magnesium sulfate",
}

# ── Language detection ─────────────────────────────────────────────────────────
_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
_ENGLISH_WORD = re.compile(r"[a-zA-Z]{2,}")


@dataclass
class LanguageProfile:
    """Classification of a query's language content."""
    is_arabic: bool = False
    is_english: bool = False
    is_mixed: bool = False
    arabic_ratio: float = 0.0
    detected_clinical_terms: list[str] = field(default_factory=list)


def detect_language(query: str) -> LanguageProfile:
    """Detect language composition of a query."""
    arabic_chars = len(_ARABIC_RANGE.findall(query))
    english_words = len(_ENGLISH_WORD.findall(query))
    total = len(query.strip())
    if total == 0:
        return LanguageProfile()

    arabic_ratio = arabic_chars / total
    has_arabic = arabic_ratio > 0.1
    has_english = english_words > 0

    clinical = [t for t in ["ICS", "SABA", "LABA", "FeNO", "FEV1", "MART"] if t in query]

    return LanguageProfile(
        is_arabic=has_arabic and not has_english,
        is_english=has_english and not has_arabic,
        is_mixed=has_arabic and has_english,
        arabic_ratio=arabic_ratio,
        detected_clinical_terms=clinical,
    )


def enhance_arabic_query(query: str) -> str:
    """Enhance an Arabic (or mixed) query with English clinical equivalents.

    Clinical acronyms (ICS, SABA, etc.) are preserved as-is.
    Arabic clinical phrases are appended with their English equivalents.
    The original Arabic text is kept so dense embeddings still capture intent.
    """
    normalized = normalize_medical_query(query)
    appended_terms: list[str] = []

    for ar_phrase, en_equiv in _AR_EN_CLINICAL_MAP.items():
        if ar_phrase in normalized:
            appended_terms.append(en_equiv)

    if appended_terms:
        enhancement = " ".join(dict.fromkeys(appended_terms))  # deduplicate
        enhanced = f"{normalized} {enhancement}"
    else:
        enhanced = normalized

    return enhanced.strip()


class MultilingualRetriever:
    """Wraps a UnifiedRetriever to add Arabic cross-lingual retrieval.

    Usage:
        ml_retriever = MultilingualRetriever(unified_retriever)
        results = ml_retriever.search("متى يُستخدم سولفات المغنيسيوم؟", top_k=5)
    """

    def __init__(self, unified_retriever: Any):
        self._retriever = unified_retriever

    def search(
        self,
        query: str,
        strategy: str = "hybrid_rerank",
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> tuple[list[SearchResult], LanguageProfile]:
        """Retrieve evidence for a query, with Arabic cross-lingual support.

        Returns:
            (results, language_profile) — so caller knows what language was detected.
        """
        lang = detect_language(query)

        if lang.is_arabic or lang.is_mixed:
            search_query = enhance_arabic_query(query)
        else:
            search_query = query

        results = self._retriever.search(
            search_query,
            strategy=strategy,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        return results, lang

    def get_enhancement(self, query: str) -> str:
        """Return the enhanced query string (for debugging/transparency)."""
        lang = detect_language(query)
        if lang.is_arabic or lang.is_mixed:
            return enhance_arabic_query(query)
        return query
