"""Safety detection layer — runs BEFORE anything touches retrieval or the LLM.

Ownership (Workstream A + Safety Fixes):
    - Emergency / unsafe request detection
    - Prompt-injection detection
    - Out-of-scope (topic) detection
    - Patient-specific clinical scenario / diagnosis request detection

This module is intentionally rule-based (regex + keyword lists), not LLM-based.
Deterministic, fast, and auditable — a safety gate should not depend on a model
that can itself be manipulated by the input it is supposed to be screening.

Bilingual (English + Arabic) by design, since the project must support Arabic queries end-to-end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class IntentCategory(str, Enum):
    GENERAL_EDUCATION = "GENERAL_EDUCATION"
    PATIENT_SCENARIO = "PATIENT_SCENARIO"
    DOSAGE_REQUEST = "DOSAGE_REQUEST"
    EMERGENCY = "EMERGENCY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    PROCEDURAL_GUIDANCE = "PROCEDURAL_GUIDANCE"


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
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompt)",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompt)",
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
    r"من\s+دلوقتي\s+هتبقى",
]

_INJECTION_PATTERNS = _INJECTION_PATTERNS_EN + _INJECTION_PATTERNS_AR

_INJECTION_REGEX = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

INJECTION_MESSAGE_EN = "Safety Policy Violation: Request refused (Prompt Injection Attempt Detected)."
INJECTION_MESSAGE_AR = "تم رفض الطلب لاختراقه سياسات الأمان."


def is_prompt_injection(text: str) -> bool:
    """Detect prompt-injection attempts before the query reaches the LLM."""
    if not text:
        return False
    return bool(_INJECTION_REGEX.search(text))


# ---------------------------------------------------------------------------
# Out-of-scope (topic) detection
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_EXPLICIT_EN = [
    r"\bbreast\s+cancer\b",
    r"\bcancer\s+screening\b",
    r"\bdiabetes\b",
    r"\bhypertension\b",
    r"\bheart\s+attack\b",
    r"\bcar\s+engine\b",
    r"\bice\s+cream\b",
    r"\bfootball\b|\bsoccer\b",
    r"\brecipe\b",
]

_OUT_OF_SCOPE_EXPLICIT_AR = [
    r"سرطان\s+الثدي",
    r"فحص\s+السرطان",
    r"السكري",
    r"ضغط\s+الدم",
    r"أزمة\s+قلبية",
    r"محرك\s+السيارة",
    r"آيس\s+كريم",
]

_OUT_OF_SCOPE_EXPLICIT_REGEX = re.compile(
    "|".join(_OUT_OF_SCOPE_EXPLICIT_EN + _OUT_OF_SCOPE_EXPLICIT_AR),
    re.IGNORECASE,
)

_ASTHMA_TOPIC_TERMS_EN = [
    "asthma", "wheez", "inhaler", "bronch", "ics", "saba", "laba", "mart",
    "feno", "spirometry", "peak flow", "exacerbation", "nebuliz",
    "controller", "reliever", "steroid", "allerg", "respirat", "lung",
    "breath", "cough", "chest tight", "gina", "who guideline", "nice",
    "dose", "dosage", "treatment", "medication", "therapy", "management",
    "symptom", "symptoms",
]
_ASTHMA_TOPIC_TERMS_AR = [
    "ربو", "أزيز", "بخاخ", "شعب", "تنفس", "صدر", "كحه", "كحة", "حساسية",
    "رئة", "رئتين", "ازمة", "أزمة", "نفس", "جرعة", "علاج", "دواء",
]
_ASTHMA_TOPIC_TERMS = _ASTHMA_TOPIC_TERMS_EN + _ASTHMA_TOPIC_TERMS_AR


def is_out_of_scope(text: str, min_words: int = 1) -> bool:
    """Detect queries clearly outside the scope of pediatric asthma guidelines."""
    if not text:
        return False
    
    # 1. Explicit out of scope topic matches
    if _OUT_OF_SCOPE_EXPLICIT_REGEX.search(text):
        return True
    
    # 2. General keyword check: if query contains no asthma-related topic terms, flag as out of scope
    lowered = text.lower()
    if not any(term in lowered for term in _ASTHMA_TOPIC_TERMS):
        return True
            
    return False


# ---------------------------------------------------------------------------
# Patient-Specific / Individual Case Detection
# ---------------------------------------------------------------------------

_PATIENT_SPECIFIC_PATTERNS = [
    r"\bmy\s+(child|son|daughter|patient|baby|kid|boy|girl|father|mother|relative)\b",
    r"\bi\s+have\s+been\s+coughing\b",
    r"\bmight\s+it\s+be\s+asthma\b",
    r"\bdo\s+i\s+have\s+asthma\b",
    r"\bcould\s+this\s+be\s+asthma\b",
    r"\bdiagnos(e|is)\s+(me|my|this|a\s+patient)\b",
    r"\b(prescribe|dose|dosage)\s+for\s+(me|my)\b",
    r"\bshould\s+i\s+(take|give|start|stop|use|increase|decrease)\b",
    r"\bweighing\s+\d+\s*(kg|lbs?)\b",
    r"\b\d+\s*(kg|lbs)\b",
    r"\bpatient\s+presenting\s+with\b",
]

_PATIENT_SPECIFIC_REGEX = re.compile("|".join(_PATIENT_SPECIFIC_PATTERNS), re.IGNORECASE)

PATIENT_SPECIFIC_MESSAGE_EN = (
    "This system provides general guideline information only and cannot "
    "diagnose individual patient scenarios or prescribe personalized treatment. "
    "Please consult a qualified healthcare professional for medical advice."
)


def is_patient_specific_scenario(text: str) -> bool:
    """Detect patient-specific diagnostic/prescribing scenarios."""
    if not text:
        return False
    return bool(_PATIENT_SPECIFIC_REGEX.search(text))


# ---------------------------------------------------------------------------
# Combined pre-flight safety check
# ---------------------------------------------------------------------------

@dataclass
class SafetyFlag:
    blocked: bool
    category: str  # "EMERGENCY" | "INJECTION" | "OUT_OF_SCOPE" | "PATIENT_SCENARIO" | "NONE"
    intent: IntentCategory = IntentCategory.GENERAL_EDUCATION
    message_en: str = ""
    message_ar: str = ""


def run_safety_gate(query: str) -> SafetyFlag:
    """First line of defense. Call this BEFORE router / retrieval / LLM.

    Priority Cascade:
      EMERGENCY -> INJECTION -> OUT_OF_SCOPE -> PATIENT_SCENARIO -> PROCEED
    """
    if is_emergency(query):
        return SafetyFlag(
            blocked=True,
            category="EMERGENCY",
            intent=IntentCategory.EMERGENCY,
            message_en=EMERGENCY_MESSAGE_EN,
            message_ar=EMERGENCY_MESSAGE_AR,
        )

    if is_prompt_injection(query):
        return SafetyFlag(
            blocked=True,
            category="INJECTION",
            intent=IntentCategory.OUT_OF_SCOPE,
            message_en=INJECTION_MESSAGE_EN,
            message_ar=INJECTION_MESSAGE_AR,
        )

    if is_out_of_scope(query):
        return SafetyFlag(
            blocked=True,
            category="OUT_OF_SCOPE",
            intent=IntentCategory.OUT_OF_SCOPE,
            message_en=(
                "This question is out of scope for the Pediatric Asthma "
                "Clinical Decision Support system (zero guideline retrieval performed)."
            ),
            message_ar=(
                "السؤال ده مش متعلق بإرشادات الربو للأطفال المحمّلة."
            ),
        )

    if is_patient_specific_scenario(query):
        return SafetyFlag(
            blocked=True,
            category="PATIENT_SCENARIO",
            intent=IntentCategory.PATIENT_SCENARIO,
            message_en=PATIENT_SPECIFIC_MESSAGE_EN,
            message_ar=PATIENT_SPECIFIC_MESSAGE_EN,
        )

    # Classify safe clinical queries
    q_lower = query.lower()
    if any(k in q_lower for k in ["how to use", "technique", "steps to", "how do i use"]):
        intent = IntentCategory.PROCEDURAL_GUIDANCE
    elif any(k in q_lower for k in ["dose", "dosage", "how much", "mg", "mcg"]):
        intent = IntentCategory.DOSAGE_REQUEST
    else:
        intent = IntentCategory.GENERAL_EDUCATION

    return SafetyFlag(blocked=False, category="NONE", intent=intent)
