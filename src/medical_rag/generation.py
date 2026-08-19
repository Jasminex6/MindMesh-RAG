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

_PATIENT_SPECIFIC_PATTERNS = [
    r"\bmy\s+(child|son|daughter|patient|baby|kid|boy|girl|father|mother|relative)\b",
    r"\b(prescribe|dose|dosage|how\s+much)\s+(for|of|should|to|can|i)\b",
    r"\bshould\s+i\s+(take|give|start|stop|use|increase|decrease)\b",
    r"\bdiagnos(e|is)\s+(me|my|this|a\s+patient)\b",
    r"\bwhat\s+(medication|drug|treatment|dose)\s+should\s+(i|my|we)\b",
    r"\b(years?\s+old|yo|y/o)\b",
    r"\bweighing\s+\d+\s*(kg|lbs?)\b",
    r"\b\d+\s*(kg|lbs)\b",
    r"\bpatient\s+presenting\s+with\b",
]


def _is_patient_specific(query: str) -> bool:
    q = query.lower()
    return any(re.search(pat, q) for pat in _PATIENT_SPECIFIC_PATTERNS)


def check_refusal(query: str,
                  results: list[SearchResult],
                  confidence: str,
                  min_results: int = 1,
                  min_top_score: float = 0.40) -> tuple[bool, str]:
    """Decide whether to refuse answering.

    Returns (should_refuse, reason).
    """
    if _is_patient_specific(query):
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


def build_user_prompt(query: str, evidence_block: str) -> str:
    return (
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
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=model, temperature=temperature)
    messages = [
        ("system", system_prompt),
        ("human", user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


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

    # Try to find a JSON object in the text
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return raw text as recommendation
    return {
        "recommendation": cleaned,
        "supporting_evidence": "",
        "citations": [],
        "safety_note": "Grounded in official guideline evidence. Clinical judgment required.",
    }


# ---------------------------------------------------------------------------
# Post-generation grounding check
# ---------------------------------------------------------------------------

def verify_citations(citations: list[Citation],
                     results: list[SearchResult]) -> list[Citation]:
    """Check each citation's chunk_id exists AND that the cited chunk text
    actually supports the claim (not just topically related to it)."""
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

        # NEW: existence check is not enough — verify the claim is actually
        # supported by this chunk's text, not just that the chunk_id exists.
        support = verify_claim_support(cit.claim, r.text)
        cit.verified = support.supported

        # Enrich with full metadata from the actual retrieved chunk
        cit.document = str(r.metadata.get("document", cit.document))
        cit.section = str(r.metadata.get("section", cit.section))
        page_start = r.metadata.get("page_start", r.metadata.get("page", ""))
        page_end = r.metadata.get("page_end", page_start)
        cit.page = f"{page_start}" if page_start == page_end else f"{page_start}-{page_end}"
        cit.score = r.score

        verified.append(cit)
    return verified

def post_generation_safety_check(answer: GeneratedAnswer) -> GeneratedAnswer:
    """Final safety sweep on the generated answer.

    Flags or downgrades if:
    - LLM returns 'Insufficient Evidence' as recommendation
    - No citations verified
    - Contains dosage/prescription language without citation
    """
    if answer.recommendation.strip().lower() == "insufficient evidence":
        answer.refused = True
        answer.refusal_reason = "Retrieved evidence does not contain sufficient information to answer this question."
        answer.confidence = "Insufficient Evidence"
        answer.safety_note = "Refused: Evidence insufficient."
        return answer

    verified_count = sum(1 for c in answer.citations if c.verified)

    if answer.citations and verified_count == 0:
        answer.confidence = "Low"
        answer.safety_note = (
            "WARNING: None of the citations could be verified against "
            "retrieved evidence. Treat this answer with caution."
        )
    elif not answer.safety_note or answer.safety_note.strip().lower() in ("none", "insufficient evidence", "null"):
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
                 skip_llm: bool = False) -> GeneratedAnswer:
        """Full generation pipeline.

        Args:
            query: the clinical question
            results: pre-retrieved SearchResult list from retrieval layer
            skip_llm: if True, build the answer structure without calling LLM
                      (useful for testing refusal/confidence logic)
        """
        # 1. Assess confidence from retrieval quality
        confidence = assess_confidence(results)

        # 2. Refusal gate
        should_refuse, reason = check_refusal(query, results, confidence)
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
        user_prompt = build_user_prompt(query, evidence_block)

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

        # 6. Build citation objects
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

        # Fallback: if JSON citations array was empty, check if chunk_ids were mentioned inline in text
        if not citations:
            rec_text = parsed.get("recommendation", "") + " " + parsed.get("supporting_evidence", "")
            for r in results:
                cid = str(r.metadata.get("chunk_id", ""))
                if cid and cid in rec_text:
                    citations.append(Citation(
                        claim=parsed.get("recommendation", "")[:150],
                        chunk_id=cid,
                        document="",
                        section="",
                        page="",
                        score=0.0,
                    ))

        # 7. Verify citations against retrieved chunks
        citations = verify_citations(citations, results)

        # 8. Build answer
        rec = parsed.get("recommendation", "")
        if isinstance(rec, (dict, list)):
            rec = json.dumps(rec)
        elif not isinstance(rec, str):
            rec = str(rec)

        supp_ev = parsed.get("supporting_evidence", "")
        if isinstance(supp_ev, list):
            supp_ev = "\n".join(f"• {str(x).strip()}" for x in supp_ev if str(x).strip())
        elif isinstance(supp_ev, dict):
            supp_ev = str(supp_ev.get("text", supp_ev.get("supporting_evidence", str(supp_ev))))
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

        answer = GeneratedAnswer(
            query=query,
            recommendation=rec,
            supporting_evidence=supp_ev,
            citations=citations,
            confidence=confidence,
            safety_note=str(parsed.get("safety_note", "")),
            raw_llm_output=raw_output,
        )

        # 9. Post-generation safety check
        answer = post_generation_safety_check(answer)

        return answer
