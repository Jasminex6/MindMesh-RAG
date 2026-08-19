"""Grounded generation service: LLM as evidence formatter, not source of truth.

Day 3 addition. The LLM receives ONLY retrieved guideline chunks and must
ground every claim in that evidence. If evidence is missing, weak, or
ambiguous, the system refuses rather than hallucinate.

Ownership:
    - Grounding prompt construction
    - Structured answer parsing
    - Citation ↔ retrieved-chunk validation
    - Confidence labeling (based on retrieval quality, NOT LLM self-confidence)
    - Safety/refusal gating
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .models import SearchResult
from .config import RELEVANCE_THRESHOLD

OUT_OF_SCOPE_RESPONSE = (
    "OUT OF SCOPE\n\n"
    "This question is outside the scope of the supported asthma/childhood-asthma clinical "
    "guideline knowledge base. Top retrieval score is below the minimum relevance threshold (0.40), "
    "or No relevant guideline evidence was retrieved. Please ask a question related to asthma "
    "symptoms, diagnosis, management, treatment, or childhood asthma."
)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """One citation linking a claim to a specific retrieved chunk."""
    claim: str
    chunk_id: str
    document: str
    section: str
    page: str
    score: float
    verified: bool = False          # True after post-generation grounding check


@dataclass
class GeneratedAnswer:
    """Structured answer returned by the generation pipeline."""
    query: str
    recommendation: str
    supporting_evidence: str
    citations: list[Citation]
    confidence: str                 # High | Medium | Low | Insufficient Evidence
    safety_note: str
    refused: bool = False
    refusal_reason: str = ""
    raw_llm_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Confidence logic — based on retrieval quality, not LLM opinion
# ---------------------------------------------------------------------------

def assess_confidence(results: list[SearchResult],
                      min_high: float = 0.50,
                      min_medium: float = 0.40,
                      min_low: float = 0.40) -> str:
    """Derive confidence label from retrieval scores.

    Rules (calibrated for nomic-embed-text cosine relevance scores):
        top_score >= min_high   AND  >=2 results above min_medium  → High
        top_score >= min_medium AND  >=1 result  above min_low     → Medium
        otherwise                                               → Insufficient Evidence
    """
    if not results:
        return "Insufficient Evidence"

    top_score = results[0].score
    above_medium = sum(1 for r in results if r.score >= min_medium)
    above_low = sum(1 for r in results if r.score >= min_low)

    if top_score >= min_high and above_medium >= 2:
        return "High"
    if top_score >= min_medium and above_low >= 1:
        return "Medium"
    return "Insufficient Evidence"


# ---------------------------------------------------------------------------
# Refusal gate — runs BEFORE calling the LLM
# ---------------------------------------------------------------------------


def check_refusal(query: str,
                  results: list[SearchResult],
                  confidence: str,
                  intent_category: str | Any | None = None,
                  slots: dict[str, Any] | None = None,
                  min_results: int = 1,
                  min_top_score: float = RELEVANCE_THRESHOLD) -> tuple[bool, str]:
    """Decide whether to refuse answering.

    Obeys the single intent_category passed from IntentClassifier upstream.
    Does NOT re-classify the query string using regex matching.
    """
    # 1. Intent category check from upstream single classifier
    if intent_category is not None:
        cat_str = intent_category.value if hasattr(intent_category, "value") else str(intent_category)
        if cat_str in ("prompt_injection", "emergency", "out_of_scope", "personal_dosing_decision"):
            if cat_str == "out_of_scope":
                return True, OUT_OF_SCOPE_RESPONSE
            elif cat_str == "prompt_injection":
                return True, (
                    "Prompt injection or system override attempt detected. "
                    "Requests attempting to bypass safety rules are rejected."
                )
            elif cat_str == "personal_dosing_decision":
                return True, (
                    "This system provides general guideline information only. "
                    "Personalized medication dosage decisions for immediate patient administration "
                    "must be performed by a qualified healthcare professional."
                )
            elif cat_str == "emergency":
                return True, (
                    "EMERGENCY WARNING: If a patient is experiencing severe breathing difficulty, "
                    "turning blue, or choking, seek immediate emergency medical care (call 911 or "
                    "go to the nearest emergency department) immediately."
                )
            return True, "Request refused per safety policy."

    # 2. Empty or low-quality retrieval check
    if not results or len(results) < min_results:
        return True, OUT_OF_SCOPE_RESPONSE

    top_score = results[0].score
    if top_score < min_top_score or all(r.score < min_top_score for r in results):
        return True, OUT_OF_SCOPE_RESPONSE

    if confidence == "Insufficient Evidence":
        return True, (
            "Retrieved evidence quality is too low to provide a reliable, "
            "grounded answer."
        )

    return False, ""


# ---------------------------------------------------------------------------
# Grounding prompt
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """\
You are a clinical guideline assistant. You MUST answer ONLY using the
retrieved guideline evidence provided below. You are NOT a doctor and must
NOT add medical knowledge from your training data.

RULES:
1. Recommendation: Answer ONLY the specific question asked based strictly on retrieved chunks. Do NOT substitute emergency treatment or unrelated conditions (e.g., bronchiolitis) if general symptoms/precautions were asked.
2. Supporting Evidence: Bullet points containing short excerpts from the exact retrieved chunks used.
3. Citations: Map each claim to its exact chunk_id from the evidence.
4. If the retrieved evidence chunks do not directly answer the specific question asked, set "recommendation": "Insufficient Evidence".
5. For GINA (GINA-Summary-Guide-2026-WEB-WMS) evidence, paraphrase recommendations in concise clinical language rather than reproducing long verbatim text.
6. CRITICAL SAFETY CONSTRAINT: NEVER generate a direct personalized medical instruction such as "take X medication at Y dose". ALWAYS phrase recommendations objectively as guideline recommendations for that patient population (e.g. "For people in this age group, GINA guidelines recommend considering...").
7. AGE BAND PRIORITY: When "AGE BAND CONTEXT" is present in the user prompt, cite ONLY from guideline sections that apply to that age band. If retrieved evidence spans multiple age bands, state explicitly which band your recommendation covers and do NOT present guidance for a different age band as if it applies here.
8. CONVERSATIONAL FOLLOW-UPS & GROUNDING: Answer follow-up questions using the provided context and conversation history. Do not trigger a refusal if the user asks for explanations, examples, or clarifications of previous points, provided they map back to the retrieved chunks and history.

Respond ONLY in the following JSON structure (no markdown fences, no extra text):
{
  "recommendation": "Short direct recommendation answering the specific question",
  "supporting_evidence": [
    "Short excerpt from chunk supporting the recommendation"
  ],
  "citations": [
    {
      "claim": "brief claim text",
      "chunk_id": "exact chunk_id from evidence"
    }
  ],
  "safety_note": "Grounded in official guideline evidence. Clinical judgment required."
}
"""


def build_evidence_block(results: list[SearchResult], min_chunk_score: float = 0.32) -> str:
    """Format retrieved chunks into the evidence block for the prompt, filtering weak chunks."""
    lines = []
    top_score = results[0].score if results else 0.0
    for r in results:
        # Filter out chunks that are below min threshold or significantly lower than top score
        if r.score < min_chunk_score or (top_score - r.score > 0.15):
            continue
        meta = r.metadata
        lines.append(f"--- EVIDENCE CHUNK ---")
        lines.append(f"chunk_id: {meta.get('chunk_id', 'unknown')}")
        lines.append(f"document: {meta.get('document', 'unknown')}")
        lines.append(f"section: {meta.get('section', 'unknown')}")
        lines.append(f"page: {meta.get('page_start', '?')}-{meta.get('page_end', '?')}")
        lines.append(f"score: {r.score:.4f}")
        lines.append(f"text:\n{r.text}")
        lines.append("")
    return "\n".join(lines)


def build_user_prompt(
    query: str,
    evidence_block: str,
    age_band: str | None = None,
    chat_history: list[dict[str, str]] | list[tuple[str, str]] | None = None,
) -> str:
    """Build the user-facing prompt for the LLM.

    Args:
        query:          The clinical question (may already be the enriched query
                        from ``summarize_for_retrieval()``).
        evidence_block: Pre-formatted evidence text from ``build_evidence_block()``.
        age_band:       Canonical age band key (e.g. ``"under_6"``, ``"children_6_11"``,
                        ``"adults_adolescents"``). When provided, a mandatory context
                        header is prepended so the LLM cites from the correct section
                        (enforces rule 7 of ``_SYSTEM_PROMPT``).
        chat_history:   Optional conversation history for context-aware generation.
    """
    _AGE_BAND_DISPLAY = {
        "under_6":            "under 6 years old (children under 5 / under-5 guideline section)",
        "children_6_11":      "6–11 years old (children aged 6 to 11)",
        "adults_adolescents": "12 years and older (adolescents and adults)",
    }
    age_header = ""
    if age_band and age_band in _AGE_BAND_DISPLAY:
        age_header = (
            f"AGE BAND CONTEXT: The patient/person this question is about is "
            f"{_AGE_BAND_DISPLAY[age_band]}.\n"
            "Prioritize evidence from guideline sections applicable to this age band. "
            "Do NOT present guidance for a different age band as if it applies here.\n\n"
        )
    history_block = ""
    if chat_history:
        lines = []
        for item in chat_history[-6:]:
            if isinstance(item, dict):
                role = "User" if item.get("role") in ("user", "human") else "Assistant"
                lines.append(f"{role}: {item.get('content', '').strip()}")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                role = "User" if str(item[0]) in ("user", "human") else "Assistant"
                lines.append(f"{role}: {str(item[1]).strip()}")
        if lines:
            history_block = "RECENT CONVERSATION HISTORY:\n" + "\n".join(lines) + "\n\n"

    return (
        f"{age_header}"
        f"{history_block}"
        f"EVIDENCE CHUNKS:\n{evidence_block}\n\n"
        f"CLINICAL QUESTION:\n{query}\n\n"
        "Answer using ONLY the evidence above. Follow the JSON structure exactly."
    )


# ---------------------------------------------------------------------------
# LLM call (Ollama via langchain-ollama, matching existing stack)
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str,
             model: str = "llama3.2", temperature: float = 0.1) -> str:
    """Call Ollama LLM and return raw text output."""
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=model, temperature=temperature)
        messages = [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
        response = llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return json.dumps({
            "recommendation": "Unable to invoke LLM model for generation.",
            "supporting_evidence": "Ensure Ollama is running locally and model is pulled.",
            "citations": [],
            "safety_note": f"LLM error: {e}",
        })


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_llm_response(raw: str) -> dict[str, Any]:
    """Extract the JSON object from the LLM output, tolerating minor noise."""
    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Regex search for first JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "recommendation": raw,
        "supporting_evidence": "",
        "citations": [],
        "safety_note": "Could not parse structured JSON from LLM output.",
    }


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------

def verify_citations(citations: list[Citation],
                     results: list[SearchResult]) -> list[Citation]:
    """Verify each citation against the retrieved chunks."""
    retrieved_map = {
        str(r.metadata.get("chunk_id", "")): r for r in results
    }

    verified_list = []
    for c in citations:
        chunk_id = str(c.chunk_id).strip()
        if chunk_id in retrieved_map:
            match = retrieved_map[chunk_id]
            meta = match.metadata
            verified_list.append(Citation(
                claim=c.claim,
                chunk_id=chunk_id,
                document=str(meta.get("document", "")),
                section=str(meta.get("section", "")),
                page=str(meta.get("page_start", meta.get("page", ""))),
                score=match.score,
                verified=True,
            ))
        else:
            verified_list.append(Citation(
                claim=c.claim,
                chunk_id=chunk_id,
                document=c.document,
                section=c.section,
                page=c.page,
                score=c.score,
                verified=False,
            ))
    return verified_list


# ---------------------------------------------------------------------------
# Post-generation safety check
# ---------------------------------------------------------------------------

def post_generation_safety_check(answer: GeneratedAnswer) -> GeneratedAnswer:
    """Post-generation safety verification."""
    if answer.refused:
        return answer

    if not answer.citations:
        answer.confidence = "Low"
        answer.safety_note = (
            "WARNING: No citations provided by LLM. Answer is unverified against guidelines."
        )
        return answer

    unverified_count = sum(1 for c in answer.citations if not c.verified)
    if unverified_count > 0 and len(answer.citations) == unverified_count:
        answer.confidence = "Low"
        answer.safety_note = (
            "WARNING: All cited chunk IDs were invalid (hallucinated citations). "
            "Refusing output due to failed grounding check."
        )
        dosage_pattern = r"\b\d+\s*(mg|mcg|µg|ml|units?)\b"
        if re.search(dosage_pattern, answer.recommendation, re.IGNORECASE):
            answer.safety_note += (
                " CAUTION: Dosage information present but no verified citations. "
                "Verify against original guideline."
            )
        answer.refused = True
        answer.recommendation = "Refused: Output could not be verified against retrieved evidence."
        answer.supporting_evidence = "(All citations were unverified hallucinated IDs)"
        return answer

    verified_count = sum(1 for c in answer.citations if c.verified)
    
    if not answer.safety_note or answer.safety_note.strip().lower() in ("none", "insufficient evidence", "null"):
        answer.safety_note = "Grounded in official guideline evidence. Clinical decisions require professional medical judgment."

    # Check for unsupported dosage claims
    dosage_pattern = r"\b\d+\s*(mg|mcg|µg|ml|units?)\b"
    if re.search(dosage_pattern, answer.recommendation, re.IGNORECASE):
        if verified_count == 0:
            answer.safety_note += (
                " CAUTION: Dosage information present but no verified citations. "
                "Verify against original guideline."
            )

    return answer


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

class GenerationService:
    """Coordinate: refusal check → prompt → LLM → parse → verify → safety.

    This service does NOT own retrieval. It receives already-retrieved chunks.
    """

    def __init__(self, model: str = "llama3.2", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def generate(self, query: str,
                 results: list[SearchResult],
                 skip_llm: bool = False,
                 age_band: str | None = None,
                 intent_category: str | None = None,
                 slots: dict | None = None,
                 chat_history: list[dict[str, str]] | list[tuple[str, str]] | None = None) -> GeneratedAnswer:
        """Full generation pipeline.

        Args:
            query:           The clinical question.
            results:         Pre-retrieved SearchResult list from retrieval layer.
            skip_llm:        If True, build answer structure without calling LLM.
            age_band:        Canonical age band key ("under_6", "children_6_11", "adults_adolescents").
            intent_category: Intent category computed by IntentClassifier upstream.
            slots:           Slots dict containing context values.
            chat_history:    Optional conversation history for context-aware prompt generation.
        """
        # 1. Assess confidence from retrieval quality
        confidence = assess_confidence(results)

        # 2. Refusal gate (obeys upstream intent_category, no re-classification)
        should_refuse, reason = check_refusal(
            query, results, confidence,
            intent_category=intent_category,
            slots=slots,
        )
        if should_refuse:
            return GeneratedAnswer(
                query=query,
                recommendation="",
                supporting_evidence="",
                citations=[],
                confidence="Insufficient Evidence",
                safety_note=reason,
                refused=True,
                refusal_reason=reason,
            )

        # 3. Build prompt
        evidence_block = build_evidence_block(results)
        user_prompt = build_user_prompt(query, evidence_block, age_band=age_band, chat_history=chat_history)

        if skip_llm:
            return GeneratedAnswer(
                query=query,
                recommendation="[LLM call skipped — testing mode]",
                supporting_evidence=evidence_block,
                citations=[],
                confidence=confidence,
                safety_note="LLM was not called (skip_llm=True).",
            )

        # 4. Call LLM
        raw_output = call_llm(
            _SYSTEM_PROMPT, user_prompt,
            model=self.model, temperature=self.temperature,
        )

        # 5. Parse response
        parsed = parse_llm_response(raw_output)
        if not isinstance(parsed, dict):
            parsed = {"recommendation": str(parsed), "supporting_evidence": "", "citations": []}

        # Safely normalize recommendation and supporting_evidence FIRST
        rec = parsed.get("recommendation", "")
        if isinstance(rec, (dict, list)):
            rec = json.dumps(rec)
        elif rec is None:
            rec = ""
        else:
            rec = str(rec)

        supp_ev = parsed.get("supporting_evidence", "")
        if isinstance(supp_ev, list):
            supp_ev = "\n".join(f"• {str(x).strip()}" for x in supp_ev if str(x).strip())
        elif isinstance(supp_ev, dict):
            supp_ev = str(supp_ev.get("text", supp_ev.get("supporting_evidence", str(supp_ev))))
        elif supp_ev is None:
            supp_ev = ""
        elif isinstance(supp_ev, str) and supp_ev.strip().startswith("{"):
            try:
                d = json.loads(supp_ev)
                if isinstance(d, dict) and "text" in d:
                    supp_ev = str(d["text"])
                elif isinstance(d, list):
                    supp_ev = "\n".join(f"• {str(x).strip()}" for x in d if str(x).strip())
            except json.JSONDecodeError:
                pass
        else:
            supp_ev = str(supp_ev)

        # 6. Build citation objects
        raw_citations = parsed.get("citations", [])
        if not isinstance(raw_citations, list):
            raw_citations = []

        citations = [
            Citation(
                claim=str(c.get("claim", "")) if isinstance(c, dict) else "",
                chunk_id=str(c.get("chunk_id", "")) if isinstance(c, dict) else "",
                document="",
                section="",
                page="",
                score=0.0,
            )
            for c in raw_citations
            if isinstance(c, dict)
        ]

        # Fallback: if JSON citations array was empty, check if chunk_ids were mentioned inline in text
        if not citations:
            rec_text = rec + " " + supp_ev
            for r in results:
                cid = str(r.metadata.get("chunk_id", ""))
                if cid and cid in rec_text:
                    citations.append(Citation(
                        claim=rec[:150],
                        chunk_id=cid,
                        document="",
                        section="",
                        page="",
                        score=0.0,
                    ))

        # 7. Verify citations against retrieved chunks
        citations = verify_citations(citations, results)

        # 8. Build answer
        safety_note = parsed.get("safety_note", "")
        if safety_note is None or isinstance(safety_note, (dict, list)):
            safety_note = str(safety_note or "")

        answer = GeneratedAnswer(
            query=query,
            recommendation=rec,
            supporting_evidence=supp_ev,
            citations=citations,
            confidence=confidence,
            safety_note=str(safety_note),
            raw_llm_output=raw_output,
        )

        # 9. Post-generation safety check
        answer = post_generation_safety_check(answer)

        return answer
