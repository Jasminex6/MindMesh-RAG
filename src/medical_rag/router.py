"""Safety and Intent Router for Clinical RAG Pipeline.

Detects request categories BEFORE retrieval/generation runs:
- Patient-specific diagnosis
- Dosage / prescribing requests
- Emergency / unsafe requests
- Out-of-scope questions
- Ambiguous questions (triggering clarification flow)
- Prompt-injection attempts (English + Arabic)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


from medical_rag.intent_classifier import IntentClassifier, OUT_OF_SCOPE_RESPONSE, HARD_REFUSE_CATEGORIES


class RouteCategory(str, Enum):
    PATIENT_DIAGNOSIS = "patient_diagnosis"
    DOSAGE_PRESCRIBING = "dosage_prescribing"
    PERSONAL_DOSING_DECISION = "personal_dosing_decision"
    GENERAL_MEDICATION_INFO = "general_medication_info"
    MEDICATION_OR_DOSE_REQUEST = "medication_or_dose_request"
    DIAGNOSIS_OR_SYMPTOMS = "diagnosis_or_symptoms"
    EMERGENCY = "emergency"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"
    PROMPT_INJECTION = "prompt_injection"
    VALID_CLINICAL = "valid_clinical"


@dataclass
class RouteDecision:
    category: RouteCategory
    should_refuse: bool
    should_clarify: bool
    refusal_reason: str = ""
    clarification_question: str | None = None
    is_injection: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "should_refuse": self.should_refuse,
            "should_clarify": self.should_clarify,
            "refusal_reason": self.refusal_reason,
            "clarification_question": self.clarification_question,
            "is_injection": self.is_injection,
        }


class SafetyRouter:
    """Delegates query routing to IntentClassifier (the single source of truth for classification)."""

    def route(self, query: str, language: str = "en") -> RouteDecision:
        classifier = IntentClassifier()
        decision = classifier.classify(query)

        # Map IntentDecision categories to RouteCategory Enum
        category_map = {
            "prompt_injection": RouteCategory.PROMPT_INJECTION,
            "emergency": RouteCategory.EMERGENCY,
            "out_of_scope": RouteCategory.OUT_OF_SCOPE,
            "personal_dosing_decision": RouteCategory.PERSONAL_DOSING_DECISION,
            "medication_or_dose_request": RouteCategory.DOSAGE_PRESCRIBING,
            "general_medication_info": RouteCategory.GENERAL_MEDICATION_INFO,
            "diagnosis_or_symptoms": RouteCategory.PATIENT_DIAGNOSIS,
            "patient_diagnosis": RouteCategory.PATIENT_DIAGNOSIS,
            "patient_specific": RouteCategory.PATIENT_DIAGNOSIS,
            "general_educational": RouteCategory.VALID_CLINICAL,
            "ambiguous": RouteCategory.AMBIGUOUS,
        }
        route_cat = category_map.get(decision.category, RouteCategory.VALID_CLINICAL)

        # Legacy SafetyRouter tests expected patient_diagnosis and dosage_prescribing to flag should_refuse=True when calling route() directly
        legacy_refuse = route_cat in (RouteCategory.PATIENT_DIAGNOSIS, RouteCategory.DOSAGE_PRESCRIBING)
        should_refusal = decision.requires_refusal or decision.requires_emergency or legacy_refuse

        refusal_reason = decision.refusal_reason or decision.emergency_response
        if legacy_refuse and not refusal_reason:
            if route_cat == RouteCategory.PATIENT_DIAGNOSIS:
                refusal_reason = (
                    "This system provides general guideline information only. "
                    "Patient-specific diagnostic assessment requires direct evaluation by a qualified healthcare professional."
                )
            else:
                refusal_reason = (
                    "This system provides general guideline information only. "
                    "Patient-specific medication dosing or prescribing must be performed by a qualified healthcare professional."
                )

        clarification_q = None
        if decision.adaptive_questions:
            clarification_q = decision.adaptive_questions[0].question

        return RouteDecision(
            category=route_cat,
            should_refuse=should_refusal,
            should_clarify=decision.requires_clarification,
            refusal_reason=refusal_reason,
            clarification_question=clarification_q,
            is_injection=(decision.category == "prompt_injection"),
        )

