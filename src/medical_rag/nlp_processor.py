"""NLP Query Processor module.

Performs bilingual (Arabic/English) clinical terminology recognition,
intelligent topic shift detection, multi-query decomposition, and standalone
query reconstruction before vector retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcessedQueryResult:
    """Output contract of the NLP Query Processor."""
    raw_query: str
    language: str                                  # "en" | "ar"
    reconstructed_query: str                       # Optimized standalone query
    is_topic_shift: bool                           # True if new topic (disconnect from history)
    is_multi_query: bool                           # True if compound multi-question
    sub_questions: list[str] = field(default_factory=list)
    extracted_terms: dict[str, Any] = field(default_factory=dict)
    implied_age_band: str | None = None            # "under_6" | "children_6_11" | "adults_adolescents"


# ---------------------------------------------------------------------------
# Arabic & English Clinical Terminology Dictionaries
# ---------------------------------------------------------------------------

_ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")

_ARABIC_CLINICAL_MAP = {
    "asthma": [r"ربو", r"الربو", r"حساسية الصدر", r"حساسية صدر", r"أزمة ربوية", r"ازمة ربوية", r"أزمات ربوية"],
    "symptoms": [r"أعراض", r"اعراض", r"علامات", r"تصفير", r"أزيز", r"ازيز", r"كحة", r"سعال", r"ضيق تنفس", r"كتمة"],
    "medication": [r"علاج", r"علاجات", r"بخاخ", r"بخاخات", r"كورتيزون", r"فنتولين", r"سالبوتامول", r"موسع شعب", r"جرعة", r"الجرعة"],
    "diagnosis": [r"تشخيص", r"فحص", r"كيف اعرف", r"طريقة الفحص"],
    "under_6": [r"أطفال أقل من 6", r"تحت 6", r"اقل من 6", r"رضيع", r"رضع", r"بيبي", r"طفل صغير", r"سنتين", r"3 سنين", r"4 سنين", r"5 سنين"],
    "children_6_11": [r"من 6 لـ 11", r"من 6 الى 11", r"6-11", r"6–11", r"7 سنين", r"8 سنين", r"9 سنين", r"10 سنين"],
    "adults_adolescents": [r"بالغ", r"بالغين", r"كبار", r"كبير", r"مراهق", r"مراهقين", r"12 سنة وفوق", r"فوق 12"],
}

_NEW_TOPIC_KEYWORDS = [
    "diabetes", "copd", "pneumonia", "bronchiolitis", "hypertension", "covid",
    "cancer", "eczema", "fever", "arthritis", "heart", "headache",
    "السكري", "السكر", "الضغط", "كورونا", "التهاب رئوي", "السرطان", "الجلد", "الحرارة", "نزلة معوية"
]

_ARABIC_IMPLICIT_FOLLOWUP_PATTERNS = [
    r"وما هي آثاره", r"ما هي آثاره", r"ما هي الجرعة", r"وما هي الجرعة",
    r"ماذا عن", r"وماذا عن", r"كيف يتم استخدامه", r"كيف نستخدمه",
    r"آثاره الجانبية", r"اعراضه الجانبية", r"طريقة استخدامه"
]

_ENGLISH_IMPLICIT_PRONOUNS = [
    r"\bit\b", r"\bthey\b", r"\bthat\b", r"\bthese\b", r"\bthose\b",
    r"\bthe second one\b", r"\bside effects\b", r"\bthe dosage\b", r"\bwhat about\b"
]


def detect_language(text: str) -> str:
    """Return 'ar' if Arabic characters are present, else 'en'."""
    return "ar" if _ARABIC_CHAR_PATTERN.search(text) else "en"


def extract_clinical_terms(text: str) -> tuple[dict[str, Any], str | None]:
    """Extract clinical concepts and implied age band from Arabic/English text."""
    extracted = {}
    implied_age = None
    text_lower = text.lower()

    # Match Arabic terms
    for concept, patterns in _ARABIC_CLINICAL_MAP.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                extracted[concept] = True
                if concept in ("under_6", "children_6_11", "adults_adolescents"):
                    implied_age = concept
                break

    # Match English age terms if not set
    if not implied_age:
        if re.search(r"\b(under 6|under-5|infant|toddler|baby|2-year|3-year|4-year|5-year)\b", text_lower):
            implied_age = "under_6"
        elif re.search(r"\b(6-11|6 to 11|6–11|7-year|8-year|9-year|10-year|11-year)\b", text_lower):
            implied_age = "children_6_11"
        elif re.search(r"\b(12\+|adolescent|adult|adults|18-year)\b", text_lower):
            implied_age = "adults_adolescents"

    return extracted, implied_age


def detect_topic_shift(query: str, chat_history: list[dict[str, str]] | list[tuple[str, str]] | None) -> bool:
    """Determine if query is a new standalone question that should NOT inherit prior history.

    Returns True if:
    - No prior history exists.
    - Query contains a new standalone medical topic keyword (e.g. diabetes vs. asthma).
    - Query is self-contained with a complete clinical subject and no implicit pronouns.
    """
    if not chat_history:
        return True

    q_lower = query.lower()

    # 1. New explicit topic keyword check
    for kw in _NEW_TOPIC_KEYWORDS:
        if kw in q_lower:
            return True

    # 2. Implicit pronoun / follow-up check
    lang = detect_language(query)
    if lang == "ar":
        has_followup_marker = any(re.search(pat, query) for pat in _ARABIC_IMPLICIT_FOLLOWUP_PATTERNS)
    else:
        has_followup_marker = any(re.search(pat, q_lower) for pat in _ENGLISH_IMPLICIT_PRONOUNS)

    if has_followup_marker:
        return False

    # 3. If query explicitly names a subject (e.g. "asthma", "الربو", "حساسية الصدر") and has >= 5 words, it is self-contained
    has_subject = any(w in q_lower for w in ["asthma", "ربو", "حساسية", "bronchiolitis"])
    word_count = len(query.split())

    if has_subject and word_count >= 4:
        return True

    return False


def decompose_bilingual_multi_query(query: str) -> tuple[bool, list[str]]:
    """Split compound multi-question queries in English and Arabic."""
    q_str = query.strip()
    if not q_str:
        return False, []

    lang = detect_language(q_str)

    # 1. Numbered lists: 1. ... 2. ... OR ١. ... ٢. ...
    numbered_parts = re.split(r"(?:\r?\n|\s+)(?=(?:\d+|[\u0660-\u0669]+)[\.\)]\s+)", q_str)
    numbered_parts = [re.sub(r"^(?:\d+|[\u0660-\u0669]+)[\.\)]\s*", "", p).strip() for p in numbered_parts if p.strip()]
    if len(numbered_parts) > 1:
        return True, numbered_parts

    # 2. Question mark split: ? or ؟
    q_mark = "؟" if lang == "ar" else "?"
    if q_str.count(q_mark) > 1:
        parts = [p.strip() + q_mark for p in q_str.split(q_mark) if p.strip()]
        parts = [p[:-1] if p.endswith(q_mark + q_mark) else p for p in parts if p.strip()]
        if len(parts) > 1:
            return True, parts

    # 3. Arabic compound conjunction split
    if lang == "ar":
        split_pat = r",\s*|\s+(?=و\s*كيف|و\s*ما\s*هي|و\s*ما\s*هو|بالإضافة\s+إلى|بالاضافة\s+الى)"
        parts = re.split(split_pat, q_str)
        parts = [p.strip() for p in parts if len(p.strip()) >= 8]
        if len(parts) > 1:
            return True, parts

    # 4. English conjunction split
    if lang == "en":
        split_pat = r",\s*(?=(?:and\s+)?(?:what|how|why|when|which|is|are|should|can)\b)|\band\s+(?=(?:what|how|why|when)\b)"
        parts = re.split(split_pat, q_str, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if len(p.strip()) >= 8]
        if len(parts) > 1:
            return True, parts

    return False, [q_str]


class NlpQueryProcessor:
    """Bilingual NLP Processor for Clinical RAG Queries."""

    def process(
        self,
        query: str,
        chat_history: list[dict[str, str]] | list[tuple[str, str]] | None = None,
    ) -> ProcessedQueryResult:
        """Process raw query into an optimized standalone question contract."""
        q_raw = query.strip()
        if not q_raw:
            return ProcessedQueryResult(
                raw_query="",
                language="en",
                reconstructed_query="",
                is_topic_shift=True,
                is_multi_query=False,
                sub_questions=[],
            )

        lang = detect_language(q_raw)
        extracted_terms, implied_age = extract_clinical_terms(q_raw)
        is_shift = detect_topic_shift(q_raw, chat_history)

        # Reconstruct query into optimal standalone format
        if is_shift or not chat_history:
            reconstructed = self._format_standalone_query(q_raw, lang, extracted_terms, implied_age)
        else:
            from .query_rewriter import rewrite_conversational_query
            reconstructed = rewrite_conversational_query(q_raw, chat_history=chat_history)

        is_multi, sub_qs = decompose_bilingual_multi_query(reconstructed)

        return ProcessedQueryResult(
            raw_query=q_raw,
            language=lang,
            reconstructed_query=reconstructed,
            is_topic_shift=is_shift,
            is_multi_query=is_multi,
            sub_questions=sub_qs,
            extracted_terms=extracted_terms,
            implied_age_band=implied_age,
        )

    def _format_standalone_query(
        self,
        query: str,
        lang: str,
        extracted_terms: dict[str, Any],
        implied_age: str | None,
    ) -> str:
        """Reconstruct query into optimal guideline format."""
        q_clean = query.strip()
        q_lower = q_clean.lower()

        # Terse English mapping
        if lang == "en":
            if q_lower in ("asthma symptoms", "symptoms of asthma"):
                return "What are the common symptoms of asthma in children?"
            if q_lower in ("asthma treatment", "treatment of asthma"):
                return "How is pediatric asthma treated?"

        # Terse Arabic mapping
        if lang == "ar":
            if "أعراض الربو" in q_clean or "اعراض الربو" in q_clean or "أعراض حساسية الصدر" in q_clean:
                return "ما هي أعراض الربو عند الأطفال؟"
            if "علاج الربو" in q_clean or "علاج حساسية الصدر" in q_clean:
                return "كيف يتم علاج الربو عند الأطفال؟"

        return q_clean
