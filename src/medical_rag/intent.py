"""Intent classification and target condition detection."""

from __future__ import annotations

from enum import Enum


class QueryIntent(str, Enum):
    SYMPTOMS = "symptoms"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    MONITORING = "monitoring"
    PREVENTION = "prevention"
    PATIENT_SPECIFIC = "patient_specific"
    OUT_OF_SCOPE = "out_of_scope"
    GENERAL = "general"


class TargetCondition(str, Enum):
    ASTHMA = "asthma"
    BRONCHIOLITIS = "bronchiolitis"
    OTHER = "other"


_OUT_OF_SCOPE_KEYWORDS = [
    "joke", "weather", "poem", "song", "recipe", "game", "movie", "football",
    "soccer", "super bowl", "ignore previous", "forget all", "system prompt",
    "tell me a story", "who is the president", "capital of", "diabetes",
    "appendicitis", "cancer", "hypertension", "heart attack"
]


def classify_query_intent(query: str) -> QueryIntent:
    """Classify clinical intent of query using keyword heuristic patterns."""
    q_lower = query.lower()

    if any(k in q_lower for k in _OUT_OF_SCOPE_KEYWORDS):
        return QueryIntent.OUT_OF_SCOPE

    if any(k in q_lower for k in ["my child", "my patient", "weighing", "prescribe"]):
        return QueryIntent.PATIENT_SPECIFIC

    if "symptom" in q_lower or "sign" in q_lower:
        return QueryIntent.SYMPTOMS
    if "treat" in q_lower or "management" in q_lower or "medication" in q_lower:
        return QueryIntent.TREATMENT
    if "diagnos" in q_lower or "assess" in q_lower:
        return QueryIntent.DIAGNOSIS

    return QueryIntent.GENERAL


def detect_target_condition(query: str) -> TargetCondition:
    """Detect if query targets asthma, bronchiolitis, or other condition."""
    q_lower = query.lower()
    if "bronchiolitis" in q_lower:
        return TargetCondition.BRONCHIOLITIS
    elif "asthma" in q_lower:
        return TargetCondition.ASTHMA
    return TargetCondition.OTHER
