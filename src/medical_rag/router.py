"""Query router — the single entry point every query passes through before
retrieval.

Ownership (Workstream A):
    - Orchestrating the safety gate (safety.py)
    - Ambiguity detection + clarification flow
    - Producing a routing decision the rest of the pipeline can act on

Flow implemented here (per the Day 4 spec):

    query
      -> safety gate (emergency / injection / out-of-scope)
      -> ambiguity check
           AMBIGUOUS -> ask clarification (STOP, do not retrieve)
           CLEAR     -> pass through to retrieval

This module does NOT call the LLM and does NOT do retrieval. It is a pure
routing/decision layer so it stays fast, deterministic, and easy to test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .safety import run_safety_gate, SafetyFlag


# ---------------------------------------------------------------------------
# Ambiguity detection
# ---------------------------------------------------------------------------

# Terms that signal the question depends on a population/age group the
# guidelines answer differently for (pediatric vs adult dosing/management
# differ throughout GINA/WHO). If the query uses one of these terms but
# gives no age/population context, we ask for clarification instead of
# guessing which section to retrieve from.
_POPULATION_DEPENDENT_TERMS_EN = [
    "treatment", "management", "dose", "dosage", "controller", "therapy",
    "medication", "first-line", "first line",
]
_POPULATION_DEPENDENT_TERMS_AR = [
    "علاج", "جرعة", "دواء", "العلاج", "إدارة",
]

_POPULATION_CONTEXT_TERMS_EN = [
    "child", "children", "kid", "pediatric", "paediatric", "infant",
    "toddler", "adult", "adults", "elderly", "pregnan", "age",
]
_POPULATION_CONTEXT_TERMS_AR = [
    "طفل", "أطفال", "رضيع", "بالغ", "بالغين", "كبار", "حامل", "عمر", "سن",
]

# Vague referential queries with no resolvable subject — cannot be
# meaningfully retrieved on their own.
_VAGUE_ONLY_PATTERNS_EN = [
    r"^\s*(it|that|this|those|these)\b.{0,25}$",
    r"^\s*(what about|how about)\b.{0,25}$",
    r"^\s*(more|tell me more|explain)\s*\.?\s*$",
    r"^\s*(and\s+)?(the\s+)?dose\s*\??\s*$",
    r"^\s*(the\s+)?treatment\s*\??\s*$",
]
_VAGUE_ONLY_PATTERNS_AR = [
    r"^\s*(ده|دي|كده|زي\s+كده)\s*\??\s*$",
    r"^\s*(وايه\s+كمان|واكتر\s+من\s+كده)\s*\??\s*$",
    r"^\s*ال?جرعة\s*\??\s*$",
    r"^\s*ال?علاج\s*\??\s*$",
]

_VAGUE_REGEX = re.compile(
    "|".join(_VAGUE_ONLY_PATTERNS_EN + _VAGUE_ONLY_PATTERNS_AR),
    re.IGNORECASE,
)


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, re.UNICODE))


def _mentions_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def classify_ambiguity(query: str) -> tuple[bool, Optional[str]]:
    """Return (is_ambiguous, clarification_question_or_None).

    Two independent triggers:
      1. The query is a bare vague reference with no resolvable subject.
      2. The query depends on population (pediatric vs adult) but gives
         no population context — GINA/WHO recommendations diverge here,
         so guessing risks returning the wrong guideline section.
    """
    stripped = query.strip()

    if not stripped:
        return True, (
            "Could you share your question? / محتاجة تكتبي سؤالك عشان أقدر أساعدك."
        )

    if _VAGUE_REGEX.match(stripped):
        return True, (
            "Could you clarify what you'd like to know more about? "
            "(e.g. symptoms, diagnosis, treatment, triggers) / "
            "ممكن توضحي محتاجة تعرفي إيه بالظبط؟ (الأعراض، التشخيص، العلاج، المحفزات)"
        )

    depends_on_population = _mentions_any(
        stripped, _POPULATION_DEPENDENT_TERMS_EN + _POPULATION_DEPENDENT_TERMS_AR
    )
    has_population_context = _mentions_any(
        stripped, _POPULATION_CONTEXT_TERMS_EN + _POPULATION_CONTEXT_TERMS_AR
    )

    if depends_on_population and not has_population_context:
        return True, (
            "Are you asking about children or adults? Asthma management "
            "guidance differs by age group. / "
            "تقصدي الأطفال ولا البالغين؟ إرشادات إدارة الربو بتختلف حسب الفئة العمرية."
        )

    # Very short, generic queries with no clear clinical anchor.
    if _word_count(stripped) <= 2 and not _mentions_any(
        stripped,
        ["asthma", "ربو", "wheez", "أزيز", "inhaler", "بخاخ"],
    ):
        return True, (
            "Could you provide a bit more detail about your asthma-related "
            "question? / ممكن توضحي سؤالك عن الربو أكتر شوية؟"
        )

    return False, None


# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    status: str  # "BLOCKED" | "CLARIFY" | "PROCEED"
    query: str
    category: str = "NONE"          # from SafetyFlag.category, or "AMBIGUOUS"
    safety_message_en: str = ""
    safety_message_ar: str = ""
    clarification_question: Optional[str] = None


def route_query(query: str) -> RoutingDecision:
    """Single entry point: run safety gate, then ambiguity check.

    Callers (pipeline.py) should check `status` and act accordingly:
        BLOCKED  -> return safety_message_* directly, do NOT retrieve
        CLARIFY  -> return clarification_question directly, do NOT retrieve
        PROCEED  -> continue to retrieval + generation as normal
    """
    safety: SafetyFlag = run_safety_gate(query)
    if safety.blocked:
        return RoutingDecision(
            status="BLOCKED",
            query=query,
            category=safety.category,
            safety_message_en=safety.message_en,
            safety_message_ar=safety.message_ar,
        )

    is_ambiguous, clarification = classify_ambiguity(query)
    if is_ambiguous:
        return RoutingDecision(
            status="CLARIFY",
            query=query,
            category="AMBIGUOUS",
            clarification_question=clarification,
        )

    return RoutingDecision(status="PROCEED", query=query, category="NONE")
