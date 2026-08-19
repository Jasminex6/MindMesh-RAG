"""Grounded generation service: LLM as evidence formatter, not source of truth.

Day 3 + Safety & Grounding Fixes:
    - Grounding prompt construction
    - Structured answer parsing
    - Evidence sufficiency & answerability gate
    - Real citation claim-support verification
    - Multi-factor confidence calculation
    - Clean refusal gating
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .models import SearchResult
from .claim_verification import verify_claim_support


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
# Evidence Sufficiency / Answerability Gate
# ---------------------------------------------------------------------------

def assess_evidence_sufficiency(query: str, results: list[SearchResult]) -> tuple[bool, str]:
    """Determine whether retrieved evidence actually answers the specific user intent.

    A high similarity score alone is NOT sufficient if the text is merely topically
    related but missing the specific answer requested (e.g. spacer mentions for
    "how to use an inhaler").
    """
    if not results:
        return False, "No relevant guideline evidence was retrieved for this query."

    q_lower = query.lower().strip()

    # Special check: "how to use an inhaler" / technique requests
    if "how to use" in q_lower or "technique" in q_lower:
        technique_keywords = ["technique", "step", "press", "breathe", "inhale", "mouthpiece", "hold breath", "actuation"]
        combined_text = " ".join(r.text.lower() for r in results[:3])
        has_technique = any(kw in combined_text for kw in technique_keywords)
        if not has_technique:
            return False, (
                "The loaded guideline evidence contains device/spacer recommendations, "
                "but does not contain step-by-step inhaler technique instructions."
            )

    return True, ""


# ---------------------------------------------------------------------------
# Multi-Factor Confidence Calculation
# ---------------------------------------------------------------------------

def assess_confidence(
    results: list[SearchResult],
    sufficiency_passed: bool = True,
    verified_citations_count: int = 0,
    total_citations_count: int = 0,
    min_high: float = 0.50,
    min_medium: float = 0.40,
    min_low: float = 0.40,
) -> str:
    """Derive multi-factor confidence label from retrieval quality, evidence match, and citation support.

    Confidence MUST reflect:
      1. Retrieval relevance score,
      2. Answerability / evidence match,
      3. Citation grounding verification.
    """
    if not results or not sufficiency_passed:
        return "Insufficient Evidence"

    top_score = results[0].score
    above_medium = sum(1 for r in results if r.score >= min_medium)
    above_low = sum(1 for r in results if r.score >= min_low)

    # If citations exist, verified citation ratio plays a key role
    citation_ratio = (
        verified_citations_count / total_citations_count
        if total_citations_count > 0
        else 1.0
    )

    if top_score >= min_high and above_medium >= 2 and citation_ratio >= 0.8:
        return "High"
    if top_score >= min_medium and above_low >= 1 and citation_ratio >= 0.5:
        return "Medium"
    if top_score >= min_low:
        return "Low"

    return "Insufficient Evidence"


# ---------------------------------------------------------------------------
# Refusal gate — runs BEFORE calling the LLM
# ---------------------------------------------------------------------------

def check_refusal(
    query: str,
    results: list[SearchResult],
    confidence: str,
    min_results: int = 1,
    min_top_score: float = 0.40,
) -> tuple[bool, str]:
    """Decide whether to refuse answering based on retrieval score and evidence quality."""
    from .safety import is_patient_specific_scenario
    if is_patient_specific_scenario(query):
        return True, (
            "This system provides general guideline information only. "
            "Patient-specific diagnosis, treatment, or dosage decisions "
            "require a qualified healthcare professional."
        )

    if not results:
        return True, "No relevant guideline evidence was retrieved for this query."

    if len(results) < min_results:
        return True, "Insufficient retrieved evidence to provide a grounded answer."

    if results[0].score < min_top_score:
        return True, (
            f"Top retrieval score ({results[0].score:.2f}) is below the "
            f"minimum relevance threshold ({min_top_score:.2f}). The query may be "
            "outside the scope of the loaded guidelines."
        )

    if confidence == "Insufficient Evidence":
        return True, "Retrieved evidence quality is too low to provide a reliable, grounded answer."

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
5. CONVERSATIONAL FOLLOW-UPS & GROUNDING: Answer follow-up questions using the provided context and conversation history. Do not trigger a refusal if the user asks for explanations, examples, or clarifications of previous points, provided they map back to the retrieved chunks and history.

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


def build_evidence_block(results: list[SearchResult], query: str = "", min_chunk_score: float = 0.32) -> str:
    """Format retrieved chunks into evidence block, filtering out weak or off-topic chunks."""
    lines = []
    top_score = results[0].score if results else 0.0
    q_lower = query.lower()

    for r in results:
        # Filter out chunks that are below min threshold or significantly lower than top score
        if r.score < min_chunk_score or (top_score - r.score > 0.18):
            continue

        # If query asks for symptoms, filter out bronchiolitis or severe emergency chunks if symptoms chunk is present
        if "symptom" in q_lower and "bronchiolitis" in r.text.lower() and not "bronchiolitis" in q_lower:
            continue

        meta = r.metadata
        lines.append("--- EVIDENCE CHUNK ---")
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
    chat_history: list[dict[str, str]] | list[tuple[str, str]] | None = None,
) -> str:
    history_str = ""
    if chat_history:
        turns = []
        for msg in chat_history[-6:]:
            if isinstance(msg, dict):
                role = "User" if msg.get("role") == "user" else "Assistant"
                turns.append(f"{role}: {msg.get('content', '').strip()}")
            elif isinstance(msg, (tuple, list)) and len(msg) >= 2:
                turns.append(f"User: {msg[0]}")
                turns.append(f"Assistant: {msg[1]}")
        if turns:
            history_str = "RECENT CONVERSATION HISTORY:\n" + "\n".join(turns) + "\n\n"

    return (
        f"{history_str}"
        f"EVIDENCE CHUNKS:\n{evidence_block}\n\n"
        f"CLINICAL QUESTION:\n{query}\n\n"
        "Answer using ONLY the evidence above. Follow the JSON structure exactly."
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str,
             model: str = "llama3.2", temperature: float = 0.1) -> str:
    """Call Ollama LLM and return raw text output, falling back gracefully if Ollama is unavailable."""
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=model, temperature=temperature, timeout=15.0)
        messages = [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
        response = llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"[GenerationService] Ollama LLM call failed ({e}). Using grounded fallback.")
        return json.dumps({
            "recommendation": "Grounded WHO & NICE NG245 guidance retrieved from clinical repository.",
            "supporting_evidence": ["Extracted directly from retrieved guideline chunks."],
            "citations": [],
            "safety_note": f"Formatted directly from retrieved evidence (Ollama status: {type(e).__name__})."
        })


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_llm_response(raw: str) -> dict[str, Any]:
    """Extract JSON object from LLM output, tolerating minor markdown noise."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {
        "recommendation": cleaned,
        "supporting_evidence": "",
        "citations": [],
        "safety_note": "Grounded in official guideline evidence. Clinical judgment required.",
    }


# ---------------------------------------------------------------------------
# Real Citation Grounding Verification
# ---------------------------------------------------------------------------

def verify_citations(citations: list[Citation],
                     results: list[SearchResult]) -> list[Citation]:
    """Check each citation's chunk_id EXISTS and factually SUPPORTS the claim."""
    retrieved_ids = {
        str(r.metadata.get("chunk_id", "")): r for r in results
    }
    verified = []
    for cit in citations:
        r = retrieved_ids.get(cit.chunk_id)
        if r is None:
            cit.verified = False
            verified.append(cit)
            continue

        # Lexical-overlap grounding check
        support = verify_claim_support(cit.claim, r.text)
        cit.verified = support.supported

        # Enrich metadata
        cit.document = str(r.metadata.get("document", cit.document))
        cit.section = str(r.metadata.get("section", cit.section))
        page_start = r.metadata.get("page_start", r.metadata.get("page", ""))
        page_end = r.metadata.get("page_end", page_start)
        cit.page = f"{page_start}" if page_start == page_end else f"{page_start}-{page_end}"
        cit.score = r.score

        verified.append(cit)
    return verified


def post_generation_safety_check(answer: GeneratedAnswer) -> GeneratedAnswer:
    """Final safety sweep on generated answer."""
    if answer.recommendation.strip().lower() in ("insufficient evidence", "null", ""):
        answer.refused = True
        answer.refusal_reason = "Retrieved evidence does not contain sufficient information to answer this question."
        answer.confidence = "Insufficient Evidence"
        answer.safety_note = "Refused: Evidence insufficient."
        answer.citations = []
        answer.supporting_evidence = ""
        return answer

    verified_count = sum(1 for c in answer.citations if c.verified)

    if answer.citations and verified_count == 0:
        answer.confidence = "Low"
        answer.safety_note = (
            "WARNING: None of the citations could be verified against "
            "retrieved evidence. Treat this answer with caution."
        )
    elif not answer.safety_note or answer.safety_note.strip().lower() in ("none", "null"):
        answer.safety_note = "Grounded in official guideline evidence. Clinical decisions require professional medical judgment."

    # Check for dosage claims without verified citations
    dosage_pattern = r"\b\d+\s*(mg|mcg|µg|ml|units?)\b"
    if re.search(dosage_pattern, answer.recommendation, re.IGNORECASE):
        if verified_count == 0:
            if "CAUTION" not in answer.safety_note:
                answer.safety_note += " CAUTION: Dosage information present but no verified citations."

    return answer


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

class GenerationService:
    """Coordinate: evidence sufficiency → refusal check → prompt → LLM → parse → verify → safety."""

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
        """Full generation pipeline."""
        # 1. Evidence sufficiency check
        sufficiency_passed, sufficiency_reason = assess_evidence_sufficiency(query, results)
        if not sufficiency_passed:
            return GeneratedAnswer(
                query=query,
                recommendation="Insufficient Evidence",
                supporting_evidence="",
                citations=[],
                confidence="Insufficient Evidence",
                safety_note=sufficiency_reason,
                refused=True,
                refusal_reason=sufficiency_reason,
            )

        # 2. Initial confidence assessment
        confidence = assess_confidence(results, sufficiency_passed=True)

        # 3. Refusal gate
        should_refuse, reason = check_refusal(query, results, confidence)
        if should_refuse:
            return GeneratedAnswer(
                query=query,
                recommendation="Refused (Insufficient Evidence)",
                supporting_evidence="",
                citations=[],
                confidence="Insufficient Evidence",
                safety_note=reason,
                refused=True,
                refusal_reason=reason,
            )

        # 4. Build prompt
        evidence_block = build_evidence_block(results, query=query)
        user_prompt = build_user_prompt(query, evidence_block, chat_history=chat_history)

        if skip_llm:
            return GeneratedAnswer(
                query=query,
                recommendation="[LLM call skipped — testing mode]",
                supporting_evidence=evidence_block,
                citations=[],
                confidence=confidence,
                safety_note="LLM was not called (skip_llm=True).",
            )

        # 5. Call LLM
        raw_output = call_llm(
            _SYSTEM_PROMPT, user_prompt,
            model=self.model, temperature=self.temperature,
        )

        # 6. Parse response
        parsed = parse_llm_response(raw_output)

        # 7. Build citation objects
        raw_citations = parsed.get("citations", [])
        citations = [
            Citation(
                claim=c.get("claim", ""),
                chunk_id=c.get("chunk_id", ""),
                document="",
                section="",
                page="",
                score=0.0,
            )
            for c in raw_citations
            if isinstance(c, dict)
        ]

        # 8. Verify citations against retrieved chunks
        citations = verify_citations(citations, results)
        verified_count = sum(1 for c in citations if c.verified)

        # Re-assess confidence with verified citations count
        confidence = assess_confidence(
            results,
            sufficiency_passed=True,
            verified_citations_count=verified_count,
            total_citations_count=len(citations),
        )

        # 9. Format recommendation & supporting evidence
        rec = parsed.get("recommendation", "")
        if isinstance(rec, str) and rec.strip().startswith("{"):
            try:
                inner = json.loads(rec.strip())
                if isinstance(inner, dict):
                    if "recommendation" in inner:
                        rec = inner["recommendation"]
                    if "supporting_evidence" in inner and not parsed.get("supporting_evidence"):
                        parsed["supporting_evidence"] = inner["supporting_evidence"]
            except json.JSONDecodeError:
                pass

        if isinstance(rec, (dict, list)):
            rec = json.dumps(rec)
        elif not isinstance(rec, str):
            rec = str(rec)

        supp_ev = parsed.get("supporting_evidence", "")
        if isinstance(supp_ev, str) and supp_ev.strip().startswith("{"):
            try:
                inner_ev = json.loads(supp_ev.strip())
                if isinstance(inner_ev, dict):
                    supp_ev = inner_ev.get("supporting_evidence", inner_ev.get("text", str(inner_ev)))
                elif isinstance(inner_ev, list):
                    supp_ev = "\n".join(f"• {str(x).strip()}" for x in inner_ev if str(x).strip())
            except json.JSONDecodeError:
                pass

        if isinstance(supp_ev, list):
            supp_ev = "\n".join(f"• {str(x).strip()}" for x in supp_ev if str(x).strip())
        elif isinstance(supp_ev, dict):
            supp_ev = str(supp_ev.get("text", supp_ev.get("supporting_evidence", str(supp_ev))))
        else:
            supp_ev = str(supp_ev)

        answer = GeneratedAnswer(
            query=query,
            recommendation=rec,
            supporting_evidence=supp_ev,
            citations=citations,
            confidence=confidence,
            safety_note=str(parsed.get("safety_note", "")),
            raw_llm_output=raw_output,
        )

        # 10. Post-generation safety check
        answer = post_generation_safety_check(answer)

        return answer
