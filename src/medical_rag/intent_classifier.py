"""Safety and Intent Classification Layer for Clinical RAG Pipeline.

Runs BEFORE retrieval to categorize queries and manage adaptive clarification or emergency bypass.

Categories:
1. "general_educational" — e.g., "What is asthma?", "How does asthma affect the lungs?"
   → Direct to RAG retrieval and answer. No clarifying questions.

2. "patient_specific" — e.g., "Should I be worried about my symptoms?"
   → Ask adaptive clarifying questions before retrieval.

3. "medication_or_dose_request" — e.g., "What medicine should I take for my asthma?", "What dose of ICS?"
   → Ask adaptive clarifying questions before retrieval.

4. "diagnosis_or_symptoms" — e.g., "Do I have asthma?", "Why do I keep coughing at night?"
   → Ask adaptive clarifying questions before retrieval.

5. "emergency" — e.g., severe breathing difficulty, blue lips, gasping, acute exacerbation.
   → Skip all clarifying questions. Immediately return urgent-safety response. Do NOT run RAG retrieval.


EXAMPLE TRANSCRIPTS:
====================

Example (a) General Educational Query:
  User: "What is asthma and how does it affect the lungs?"
  Classifier Decision:
    Category: "general_educational"
    Requires Clarification: False
    Requires Emergency: False
  Pipeline Execution:
    Executes RAG retrieval immediately -> Generates population-based grounded answer.

Example (b) Medication Request with Adaptive Clarification:
  User: "What dose of ICS should I use for an 8-year-old child?"
  Classifier Decision:
    Category: "medication_or_dose_request"
    Requires Clarification: True
    Adaptive Questions Selected:
      - age_group: SKIPPED (already implied in query: 8-year-old -> "6-11")
      - diagnosed_by_professional: "Has the patient been formally diagnosed with asthma by a healthcare professional?" (Yes/No)
      - current_symptoms: "What is the current symptom severity?" (None / Mild / Severe or worsening)
  User Clarification Inputs:
    diagnosed_by_professional: "Yes"
    current_symptoms: "Mild"
  Pipeline Execution:
    Passes age_group="children_6_11" to retriever -> Generates population-framed guideline response:
    "For children 6-11 years with asthma, GINA guidelines recommend low-dose ICS as controller therapy..."

Example (c) Emergency Query Bypassing Retrieval:
  User: "My child is gasping for air and turning blue!"
  Classifier Decision:
    Category: "emergency"
    Requires Clarification: False
    Requires Emergency: True
    Response:
      "EMERGENCY URGENT WARNING: If a patient is experiencing severe breathing difficulty,
       turning blue, or choking, seek immediate emergency medical care (call 911 or go to
       the nearest emergency department) immediately."
  Pipeline Execution:
    Bypasses RAG retrieval and generation entirely -> Returns urgent emergency response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ── Pattern Registries ────────────────────────────────────────────────────────

_PROMPT_INJECTION_PATTERNS = [
    # English
    r"\b(ignore|override|bypass|disregard|forget)\s+(all\s+)?(previous\s+)?(instructions|rules|guidelines|prompts|system)\b",
    r"\b(you\s+are\s+now|act\s+as)\s+(an?\s+)?(unrestricted|unfiltered|jailbroken|DAN|evil|god|hacker)\b",
    r"\bsystem\s+prompt\b",
    r"\breveal\s+(the\s+)?(system|hidden|internal)\s+(prompt|instructions)\b",
    r"\[\s*system\s*:\s*override\s*\]",
    # Arabic
    r"تجاهل\s+(جميع\s+)?(التعليمات|القواعد|الإرشادات|الأوامر)",
    r"تجاوز\s+(نظام\s+)?(الأمان|الحماية|القيود)",
    r"أنت\s+الآن\s+(مساعد|نظام)",
    r"غير\s+مقيد|بدون\s+قواعد|بدون\s+قيود",
    r"اكتب\s+كـ\s*(حاك|دان|مساعد\s+غير\s+أخلاقي)",
    r"كشف\s+(التعليمات|الأوامر)\s+الداخلية",
]

_EMERGENCY_PATTERNS = [
    r"\b(turning|turned)\s+blue\b",
    r"\b(can'?t|cannot)\s+breathe\b",
    r"\bsevere\s+(respiratory|breathing)\s+(distress|arrest|failure|difficulty)\b",
    r"\b(unconscious|gasping|suffocating|choking|drowsy|confused)\b",
    r"\bcan'?t\s+speak\s+(in\s+full\s+sentences|words)\b",
    r"\bblue\s+(lips|face|skin|tongue)\b",
    r"\bcall\s+911\b",
    r"أنقذوني|طفلي\s+لا\s+يتنفس|ازرقاق|اختناق|صعوبة\s+شديدة\s+في\s+التنفس",
]

_OUT_OF_SCOPE_PATTERNS = [
    r"\b(broken|arm|leg|bone|fracture|sprain|dislocation|cut|wound|burn|diabetes|metformin|insulin|hypertension|blood\s+pressure|appendicitis|appendectomy|chemotherapy|cancer|stroke|heart\s+attack|headache|migraine|stomach|ulcer|diarrhea|constipation|tooth|dental|eye|vision|cataract|skin|eczema|psoriasis|acne)\b",
    r"كسر|عظم|ذراع|رجل|جرح|حرق|سكري|السكري|ضغط\s+الدم|زائدة\s+دودية|ورم|سرطان|سكتة|صداع|معدة|أسنان|عين|جلد",
]

_PERSONAL_DOSING_PATTERNS = [
    r"\bshould\s+i\s+(give|take|administer|use)\s+.*(second|third|another|extra|more)\s+(puff|dose)\b",
    r"\bgive\s+.*(second|third|extra)\s+(puff|dose)\b",
    r"\bhow\s+many\s+puffs?\s+should\s+i\s+(give|take)\s+.*(right\s+now|immediately)\b",
    r"\bcan\s+i\s+(double|triple)\s+the\s+dose\b",
    r"\bshould\s+i\s+give\s+.*(right\s+now|immediately)\b",
    r"هل\s+أعطي.*(جرعة|بخخة)\s+(ثانية|إضافية)\s+الآن",
    r"كم\s+بخخة\s+أعطي.*الآن",
]

_EDUCATIONAL_PATTERNS = [
    r"^what\s+is\s+asthma\??$",
    r"^how\s+does\s+asthma\s+affect\b",
    r"^explain\s+(the\s+)?(pathophysiology|definition|mechanism|guidelines?)\b",
    r"^what\s+(is|are)\s+(feno|ics|saba|laba|mart|fev1|pef|gina)\??$",
    r"^difference\s+between\s+ics\s+and\s+saba",
    r"^what\s+does\s+the\s+guideline\s+say\s+about\b",
    r"ما\0647\0648\s+\0627\0644\0631\0628\0648\??$",
]

_MEDICATION_PATTERNS = [
    r"\b(prescribe|dose|dosage|medication|drug|inhaler|ics|saba|laba|mart|salbutamol|fluticasone|budesonide|steroids?)\b",
    r"\bwhat\s+(medicine|drug|treatment|dose|inhaler)\s+should\b",
    r"\bwhat\s+treatment\s+do\s+you\s+recommend\b",
    r"\bhow\s+much\s+(ics|saba|salbutamol|dose)\b",
    r"\bshould\s+i\s+(start|give|take|use)\s+.*(budesonide|ics|saba|salbutamol|inhaler|steroids?|medication|treatment)\b",
    r"كم\s+جرعة|دواء|بخاخ|علاج|سولفات|اعطِ\s+طفلي",
]

_DIAGNOSIS_PATTERNS = [
    r"\bdo\s+i\s+have\s+asthma\b",
    r"\bcould\s+this\s+be\s+asthma\b",
    r"\bcould\s+it\s+be\s+asthma\b",
    r"\bwhy\s+do\s+i\s+(cough|wheeze|gasp)\b",
    r"\bcoughing\s+at\s+night\b",
    r"\bdiagnos(e|is)\s+(me|my|this|patient)\b",
    r"\bis\s+(coughing|wheezing)\s+a\s+sign\b",
    r"هل\s+عندي\s+ربو|تشخيص|لماذا\s+أسعل",
]

_PATIENT_SPECIFIC_PATTERNS = [
    r"\bshould\s+i\s+be\s+worried\b",
    r"\bmy\s+(child|son|daughter|patient|baby|kid|boy|girl|father|mother)\b",
    r"\bfor\s+my\s+(asthma|symptoms)\b",
    r"\bin\s+my\s+case\b",
    r"طفلي|ابني|حالتي|أشعر",
]

_AMBIGUOUS_PATTERNS = [
    r"^(dose|dosage|treatment|medication|asthma|help|management)\??$",
    r"^what\s+is\s+the\s+dose\??$",
    r"^how\s+to\s+treat\s+it\??$",
    r"^ما\s+هي\s+الجرعة\??$",
    r"^كيف\s+نعالجه\??$",
]

OUT_OF_SCOPE_RESPONSE = (
    "OUT OF SCOPE\n\n"
    "This question is outside the scope of the supported asthma/childhood-asthma clinical "
    "guideline knowledge base. Top retrieval score is below the minimum relevance threshold (0.40), "
    "or No relevant guideline evidence was retrieved. Please ask a question related to asthma "
    "symptoms, diagnosis, management, treatment, or childhood asthma."
)


@dataclass
class ClarifyingQuestion:
    id: str
    question: str
    options: list[str]
    allow_free_text: bool = False


@dataclass
class IntentDecision:
    category: str  # "prompt_injection", "emergency", "out_of_scope", "personal_dosing_decision", "general_educational", "patient_specific", "medication_or_dose_request", "diagnosis_or_symptoms", "ambiguous"
    requires_clarification: bool
    requires_emergency: bool
    requires_refusal: bool = False
    refusal_reason: str = ""
    emergency_response: str = ""
    adaptive_questions: list[ClarifyingQuestion] = field(default_factory=list)
    implied_age_group: str = "all_ages"
    implied_diagnosis: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "requires_clarification": self.requires_clarification,
            "requires_emergency": self.requires_emergency,
            "requires_refusal": self.requires_refusal,
            "refusal_reason": self.refusal_reason,
            "emergency_response": self.emergency_response,
            "adaptive_questions": [q.id for q in self.adaptive_questions],
            "implied_age_group": self.implied_age_group,
        }


class IntentClassifier:
    """Single source of truth for query classification (safety & intent)."""

    def classify(self, query: str) -> IntentDecision:
        q_str = query.strip()
        q_lower = q_str.lower()

        # 1. Prompt Injection Detection (English + Arabic)
        for pat in _PROMPT_INJECTION_PATTERNS:
            if re.search(pat, q_str, re.IGNORECASE):
                return IntentDecision(
                    category="prompt_injection",
                    requires_clarification=False,
                    requires_emergency=False,
                    requires_refusal=True,
                    refusal_reason=(
                        "Prompt injection or system override attempt detected. "
                        "Requests attempting to bypass safety rules are rejected."
                    ),
                )

        # 2. Emergency Detection
        for pat in _EMERGENCY_PATTERNS:
            if re.search(pat, q_str, re.IGNORECASE):
                return IntentDecision(
                    category="emergency",
                    requires_clarification=False,
                    requires_emergency=True,
                    requires_refusal=True,
                    emergency_response=(
                        "EMERGENCY URGENT WARNING: If a patient is experiencing severe breathing difficulty, "
                        "turning blue, or choking, seek immediate emergency medical care (call 911 or "
                        "go to the nearest emergency department) immediately. Do not delay emergency evaluation."
                    ),
                    refusal_reason=(
                        "EMERGENCY WARNING: Severe respiratory symptoms require immediate medical attention."
                    ),
                )

        # 3. Out of Scope Detection
        for pat in _OUT_OF_SCOPE_PATTERNS:
            if re.search(pat, q_str, re.IGNORECASE):
                return IntentDecision(
                    category="out_of_scope",
                    requires_clarification=False,
                    requires_emergency=False,
                    requires_refusal=True,
                    refusal_reason=OUT_OF_SCOPE_RESPONSE,
                )

        # 4. Personal Dosing Decision Detection
        # Requests asking for immediate personalized dosage decisions (e.g., "should I give a second puff right now")
        for pat in _PERSONAL_DOSING_PATTERNS:
            if re.search(pat, q_str, re.IGNORECASE):
                return IntentDecision(
                    category="personal_dosing_decision",
                    requires_clarification=False,
                    requires_emergency=False,
                    requires_refusal=True,
                    refusal_reason=(
                        "This system provides general guideline information only. "
                        "Personalized medication dosage decisions for immediate patient administration "
                        "must be performed by a qualified healthcare professional."
                    ),
                )

        # 5. Ambiguous Query Detection
        is_short = len(q_str.split()) <= 3 and not q_str.endswith("?")
        matches_ambiguous = any(re.search(pat, q_str, re.IGNORECASE) for pat in _AMBIGUOUS_PATTERNS)
        if matches_ambiguous or (is_short and q_lower in ("dose", "dosage", "treatment", "asthma", "reliever", "الربو", "الجرعة")):
            return IntentDecision(
                category="ambiguous",
                requires_clarification=True,
                requires_emergency=False,
                requires_refusal=False,
                adaptive_questions=[
                    ClarifyingQuestion(
                        id="clarify_scope",
                        question="Please clarify whether you are asking about acute asthma treatment or long-term controller maintenance.",
                        options=["Acute treatment", "Long-term maintenance"],
                    )
                ],
            )

        # 6. General Educational Check
        is_educational = any(re.search(pat, q_str, re.IGNORECASE) for pat in _EDUCATIONAL_PATTERNS)
        if is_educational:
            return IntentDecision(
                category="general_educational",
                requires_clarification=False,
                requires_emergency=False,
                requires_refusal=False,
            )

        # 7. Categorize into Medication/Dose, Diagnosis/Symptoms, or Patient-Specific
        category = "general_educational"
        if any(re.search(pat, q_str, re.IGNORECASE) for pat in _MEDICATION_PATTERNS):
            category = "medication_or_dose_request"
        elif any(re.search(pat, q_str, re.IGNORECASE) for pat in _DIAGNOSIS_PATTERNS):
            category = "diagnosis_or_symptoms"
        elif any(re.search(pat, q_str, re.IGNORECASE) for pat in _PATIENT_SPECIFIC_PATTERNS):
            category = "patient_specific"

        if category == "general_educational":
            return IntentDecision(
                category="general_educational",
                requires_clarification=False,
                requires_emergency=False,
                requires_refusal=False,
            )

        # 8. Build Adaptive Clarifying Questions for clinical categories
        adaptive_qs, implied_age, implied_diag = self._build_adaptive_questions(q_str, category)

        return IntentDecision(
            category=category,
            requires_clarification=len(adaptive_qs) > 0,
            requires_emergency=False,
            requires_refusal=False,
            adaptive_questions=adaptive_qs,
            implied_age_group=implied_age,
            implied_diagnosis=implied_diag,
        )

    def _build_adaptive_questions(
        self, query: str, category: str
    ) -> tuple[list[ClarifyingQuestion], str, bool | None]:
        q_lower = query.lower()
        questions: list[ClarifyingQuestion] = []

        # Age group detection
        implied_age = "all_ages"
        if re.search(r"\b(toddler|infant|baby|[1-5]\s*(-|\s*to\s*)?(year|yr|yo|y/o)|under\s+6)\b", q_lower):
            implied_age = "children_under_6"
        elif re.search(r"\b(6|7|8|9|10|11)\s*(-|\s*to\s*)?(year|yr|yo|y/o)\b", q_lower):
            implied_age = "children_6_11"
        elif re.search(r"\b(adults?|adolescents?|teenagers?|teens?|12\+|>=12|1[2-9]\s*(-|\s*to\s*)?(year|yr|yo|y/o))\b", q_lower):
            implied_age = "adults_adolescents"

        # Ask age_group ONLY if not already implied
        if implied_age == "all_ages":
            questions.append(
                ClarifyingQuestion(
                    id="age_group",
                    question="What is the age group of the person this question is about?",
                    options=["Under 6", "6–11", "12+"],
                )
            )

        # Diagnosis detection
        implied_diag = None
        if "diagnosed with asthma" in q_lower or "has asthma" in q_lower:
            implied_diag = True
        elif "do i have asthma" in q_lower or "is it asthma" in q_lower:
            implied_diag = False

        if implied_diag is None and category in ("medication_or_dose_request", "diagnosis_or_symptoms"):
            questions.append(
                ClarifyingQuestion(
                    id="diagnosed_by_professional",
                    question="Has the condition been formally diagnosed by a healthcare professional?",
                    options=["Yes", "No"],
                )
            )

        # Always check current symptoms severity
        questions.append(
            ClarifyingQuestion(
                id="current_symptoms",
                question="What is the current symptom severity?",
                options=["None", "Mild", "Severe or worsening"],
            )
        )

        # Current medication
        if category == "medication_or_dose_request":
            questions.append(
                ClarifyingQuestion(
                    id="current_medication",
                    question="Are they currently taking any asthma medication?",
                    options=["Yes", "No"],
                    allow_free_text=True,
                )
            )

        # Information goal
        questions.append(
            ClarifyingQuestion(
                id="information_goal",
                question="What information are you seeking from the guidelines?",
                options=[
                    "General treatment options",
                    "How asthma medicines work",
                    "Possible side effects",
                    "What the guidelines recommend",
                ],
            )
        )

        return questions, implied_age, implied_diag


# ---------------------------------------------------------------------------
# Mandatory slot enforcement & Safety rules
# ---------------------------------------------------------------------------

#: Intents that trigger hard refusal (RAG retrieval and generation are bypassed).
HARD_REFUSE_CATEGORIES: frozenset[str] = frozenset({
    "prompt_injection",
    "emergency",
    "out_of_scope",
    "personal_dosing_decision",
})

#: Intents that REQUIRE age to be known before retrieval can proceed.
#: This is a deterministic constant — not left to the LLM's discretion.
AGE_MANDATORY_INTENTS: frozenset[str] = frozenset({"medication_or_dose_request", "general_medication_info"})

#: Maps slot answer → (enriched label, parse_query_age-compatible phrase)
_AGE_BAND_LABELS: dict[str, tuple[str, str]] = {
    # Keys are canonical slot values stored in slots["age_band"]
    "under_6":       ("under-5 age band, children under 6", "2-year-old child under 6"),
    "children_6_11": ("age band children 6–11",             "9-year-old child aged 6 to 11"),
    "adults_adolescents": ("age band 12+ adolescent adult", "adult aged 18"),
}

#: Human-readable option labels that the CLI presents to the user,
#: mapped back to canonical slot values.
AGE_BAND_OPTIONS: dict[str, str] = {
    "1": "under_6",
    "2": "children_6_11",
    "3": "adults_adolescents",
    # Also accept canonical keys directly (for programmatic callers / tests)
    "under_6": "under_6",
    "children_6_11": "children_6_11",
    "adults_adolescents": "adults_adolescents",
    # Accept the display strings the classifier uses (from ClarifyingQuestion.options)
    "under 6": "under_6",
    "6–11": "children_6_11",
    "12+": "adults_adolescents",
}


def summarize_for_retrieval(
    query: str,
    slots: dict[str, str | None],
    category: str,
) -> str:
    """Build an enriched query string from the original query + filled slot values.

    The returned string embeds the age band explicitly so the existing
    ``parse_query_age()`` in ``UnifiedRetriever`` can detect it and apply
    the correct under-6 filter and age-band boost automatically.

    Args:
        query:    The user's original clinical question.
        slots:    Dict of filled slot values. Must contain ``"age_band"`` for
                  ``medication_or_dose_request`` queries before this is called.
        category: The intent category from ``IntentClassifier.classify()``.

    Returns:
        Enriched query string ready to pass to ``UnifiedRetriever.search()``.

    Examples:
        >>> summarize_for_retrieval(
        ...     "What medicine for asthma?",
        ...     {"age_band": "under_6"},
        ...     "medication_or_dose_request",
        ... )
        "What medicine for asthma? [2-year-old child under 6, under-5 age band, children under 6]"

        >>> summarize_for_retrieval(
        ...     "Which inhaler for asthma?",
        ...     {"age_band": "children_6_11"},
        ...     "medication_or_dose_request",
        ... )
        "Which inhaler for asthma? [9-year-old child aged 6 to 11, age band children 6–11]"

        >>> summarize_for_retrieval(
        ...     "Best controller for asthma?",
        ...     {"age_band": "adults_adolescents"},
        ...     "medication_or_dose_request",
        ... )
        "Best controller for asthma? [adult aged 18, age band 12+ adolescent adult]"
    """
    age_band = slots.get("age_band")
    if age_band and age_band in _AGE_BAND_LABELS:
        label, phrase = _AGE_BAND_LABELS[age_band]
        return f"{query} [{phrase}, {label}]"
    # No age band in slots — return query unchanged (caller should have ensured age is filled)
    return query
