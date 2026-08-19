"""Safety detection layer — runs BEFORE anything touches retrieval or the LLM.

Ownership (Workstream A):
    - Emergency / unsafe request detection
    - Prompt-injection detection
    - Out-of-scope (topic) detection

This module is intentionally rule-based (regex + keyword lists), not
LLM-based. Deterministic, fast, and auditable — a safety gate should not
depend on a model that can itself be manipulated by the input it is
supposed to be screening.

Bilingual (English + Arabic) by design, since the project must support
Arabic queries end-to-end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Emergency / unsafe request detection
# ---------------------------------------------------------------------------

_EMERGENCY_PATTERNS_EN = [
    r"\bcan\s*'?t\s+breathe\b",
    r"\bcannot\s+breathe\b",
    r"\bnot\s+breathing\b",
    r"\bstopped\s+breathing\b",
    r"\bturning\s+blue\b",
    r"\bblue\s+lips\b",
    r"\blips?\s+(are\s+)?blue\b",
    r"\bunconscious\b",
    r"\bunresponsive\b",
    r"\bpassed\s+out\b",
    r"\bcollapsed\b",
    r"\bsevere\s+difficulty\s+breathing\b",
    r"\bstruggling\s+to\s+breathe\b",
    r"\bgasping\s+for\s+air\b",
    r"\bchest\s+(is\s+)?(very\s+)?tight.{0,20}(can'?t|cannot|struggling)\b",
    r"\bemergency\b.{0,20}\bnow\b",
    r"\bcall\s+(an\s+)?ambulance\b",
    r"\b911\b|\b999\b|\b112\b",
]

_EMERGENCY_PATTERNS_AR = [
    r"وقف(ت)?\s+ال?تنفس",
    r"مش\s+قادر\s+يتنفس",
    r"مش\s+قادره?\s+تتنفس",
    r"مبي?قدرش?\s+يتنفس",
    r"بيتلون\s+ب?الأزرق",
    r"شفاي?فه?\s+زرقاء?",
    r"وشه?\s+ب?يزرق",
    r"فاقد\s+الوعي",
    r"مش\s+واعي",
    r"غايب\s+عن\s+الوعي",
    r"وقع\s+ومش\s+بيرد",
    r"صعوبة\s+شديدة\s+في\s+ال?تنفس",
    r"مختنق",
    r"اتصل(و)?\s+بال?إسعاف",
    r"طوارئ\s+دلوقتي",
    r"حالة\s+خطيرة\s+دلوقتي",
]

_EMERGENCY_PATTERNS = _EMERGENCY_PATTERNS_EN + _EMERGENCY_PATTERNS_AR
_EMERGENCY_REGEX = re.compile("|".join(_EMERGENCY_PATTERNS), re.IGNORECASE)

EMERGENCY_MESSAGE_EN = (
    "This sounds like it may be a medical emergency. Please call your local "
    "emergency number or go to the nearest emergency department immediately. "
    "This system cannot provide emergency medical care."
)
EMERGENCY_MESSAGE_AR = (
    "الوصف ده ممكن يكون حالة طارئة. من فضلك اتصل بالإسعاف فورًا أو روح لأقرب "
    "قسم طوارئ حالًا. النظام ده مش بديل عن الرعاية الطبية الطارئة."
)


def is_emergency(text: str) -> bool:
    """Detect emergency / acutely unsafe language in the query."""
    if not text:
        return False
    return bool(_EMERGENCY_REGEX.search(text))


# ---------------------------------------------------------------------------
# Prompt-injection detection
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS_EN = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|prompt)",
    r"you\s+are\s+now\s+(a|an)\b",
    r"act\s+as\s+(a|an|if)\b",
    r"pretend\s+(to\s+be|you\s+are)\b",
    r"new\s+instructions?\s*:",
    r"system\s+prompt",
    r"reveal\s+your\s+(system\s+)?prompt",
    r"show\s+me\s+your\s+(instructions?|prompt|rules)",
    r"what\s+are\s+your\s+(instructions?|rules|guidelines)\b",
    r"do\s+not\s+follow\s+(the|your)\s+(rules?|guidelines?|instructions?)",
    r"bypass\s+(the\s+)?(safety|rules?|restrictions?|filters?)",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"override\s+(the\s+)?(rules?|instructions?|safety)",
    r"from\s+now\s+on\s+you\s+(will|must|shall)\b",
    r"respond\s+only\s+with\s+.{0,20}(uncensored|unfiltered|no\s+restrictions?)",
]

_INJECTION_PATTERNS_AR = [
    r"تجاهل\s+(كل\s+)?(التعليمات|الأوامر)\s+(السابقة|اللي\s+فاتت)",
    r"انسى\s+(كل\s+)?(التعليمات|الأوامر|القواعد)",
    r"انت\s+دلوقتي\s+(هتبقى|بقيت)",
    r"تصرف\s+وكأنك\b",
    r"اعمل\s+نفسك\b",
    r"تعليمات\s+جديدة\s*:",
    r"system\s*prompt",
    r"وريني\s+ال?تعليمات\s+بتاعتك",
    r"اظهر\s+ال?system\s*prompt",
    r"متلتزمش\s+بال?قواعد",
    r"تجاوز\s+ال?قيود",
    r"من\s+دلوقتي\s+هتبقى\b",
]

_INJECTION_PATTERNS = _INJECTION_PATTERNS_EN + _INJECTION_PATTERNS_AR
_INJECTION_REGEX = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

INJECTION_MESSAGE_EN = (
    "This request appears to attempt to override the system's safety "
    "instructions. I can only answer clinical questions about asthma "
    "guidelines based on the retrieved evidence."
)
INJECTION_MESSAGE_AR = (
    "الطلب ده بيبدو إنه بيحاول يتجاوز تعليمات الأمان بتاعة النظام. أقدر بس "
    "أجاوب على أسئلة طبية عن إرشادات الربو بناءً على الأدلة المسترجعة."
)


def is_prompt_injection(text: str) -> bool:
    """Detect prompt-injection attempts before the query reaches the LLM."""
    if not text:
        return False
    return bool(_INJECTION_REGEX.search(text))


# ---------------------------------------------------------------------------
# Out-of-scope (topic) detection
# ---------------------------------------------------------------------------

# Lightweight topical allow-list. This is deliberately permissive: it only
# flags a query as OUT_OF_SCOPE when it contains NONE of these terms AND
# does not look like a generic follow-up. Retrieval-score-based rejection
# (already in generation.py) remains the primary out-of-scope safety net —
# this is an early, cheap pre-filter.
_ASTHMA_TOPIC_TERMS_EN = [
    "asthma", "wheez", "inhaler", "bronch", "ics", "saba", "laba", "mart",
    "feno", "spirometry", "peak flow", "exacerbation", "nebuliz",
    "controller", "reliever", "steroid", "allerg", "respirat", "lung",
    "breath", "cough", "chest tight", "gina", "who guideline",
]
_ASTHMA_TOPIC_TERMS_AR = [
    "ربو", "أزيز", "بخاخ", "شعب", "تنفس", "صدر", "كحه", "كحة", "حساسية",
    "رئة", "رئتين", "ازمة", "أزمة", "نفس",
]
_ASTHMA_TOPIC_TERMS = _ASTHMA_TOPIC_TERMS_EN + _ASTHMA_TOPIC_TERMS_AR


def is_probably_out_of_scope(text: str, min_words: int = 3) -> bool:
    """Cheap keyword pre-filter for obviously off-topic questions.

    Conservative on purpose: only flags when NO asthma/respiratory-related
    term is present at all. Genuine borderline cases are still expected to
    be caught downstream by the retrieval-score gate in generation.py.

    Short/vague fragments (e.g. "the dose", "it") are deliberately left to
    the ambiguity router instead of being flagged here — a fragment with
    no topic term is ambiguous, not necessarily off-topic.
    """
    if not text:
        return False
    if len(re.findall(r"\w+", text, re.UNICODE)) < min_words:
        return False
    lowered = text.lower()
    return not any(term in lowered for term in _ASTHMA_TOPIC_TERMS)


# ---------------------------------------------------------------------------
# Combined pre-flight safety check
# ---------------------------------------------------------------------------

@dataclass
class SafetyFlag:
    blocked: bool
    category: str  # "EMERGENCY" | "INJECTION" | "OUT_OF_SCOPE" | "NONE"
    message_en: str = ""
    message_ar: str = ""


def run_safety_gate(query: str) -> SafetyFlag:
    """First line of defense. Call this BEFORE router / retrieval / LLM.

    Order matters: emergency takes priority over injection, which takes
    priority over out-of-scope, since a query could technically match more
    than one pattern.
    """
    if is_emergency(query):
        return SafetyFlag(
            blocked=True,
            category="EMERGENCY",
            message_en=EMERGENCY_MESSAGE_EN,
            message_ar=EMERGENCY_MESSAGE_AR,
        )

    if is_prompt_injection(query):
        return SafetyFlag(
            blocked=True,
            category="INJECTION",
            message_en=INJECTION_MESSAGE_EN,
            message_ar=INJECTION_MESSAGE_AR,
        )

    if is_probably_out_of_scope(query):
        return SafetyFlag(
            blocked=True,
            category="OUT_OF_SCOPE",
            message_en=(
                "This question doesn't appear to relate to asthma "
                "guidelines. This system only answers asthma-related "
                "clinical questions from the loaded WHO/GINA guidelines."
            ),
            message_ar=(
                "السؤال ده مش شكله متعلق بإرشادات الربو. النظام ده بيجاوب "
                "بس على أسئلة طبية عن الربو من إرشادات WHO/GINA المحمّلة."
            ),
        )

    return SafetyFlag(blocked=False, category="NONE")
