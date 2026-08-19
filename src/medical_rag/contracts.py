"""Canonical Shared Data Contracts for RAG Service & UI Consumers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CitationContract:
    claim: str
    chunk_id: str
    document: str
    section: str
    page: str
    score: float
    verified: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": self.document or "WHO 2026 / NICE NG245",
            "section": self.section or "",
            "page": self.page or "",
            "score": round(float(self.score), 4),
            "claim": self.claim,
            "chunk_id": self.chunk_id,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class EvidenceContract:
    chunk_id: str
    text: str
    document_name: str
    section_title: str
    page_number: str
    retrieval_score: float
    verification_status: bool


@dataclass(frozen=True)
class RAGResponse:
    status: str                       # SUCCESS | EMERGENCY | OUT_OF_SCOPE | NO_EVIDENCE | CLARIFY
    language: str                     # "en" | "ar"
    query: str
    resolved_query: str
    recommendation: str
    evidence: tuple[EvidenceContract, ...]
    citations: tuple[CitationContract, ...]
    confidence: str                   # High | Moderate | Ungrounded
    safety_message: str
    clarification_question: str | None = None
    video_url: str | None = None
    video_title: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["content"] = self.recommendation
        d["confidence_level"] = self.confidence if self.confidence in ("High", "Moderate") else "Ungrounded"
        d["is_verified"] = any(c.verified for c in self.citations) if self.citations else (self.status == "SUCCESS")
        payload_citations = []
        for c in self.citations:
            if isinstance(c, CitationContract):
                payload_citations.append(c.to_payload())
            elif isinstance(c, dict):
                payload_citations.append(c)
        d["citations_payload"] = payload_citations
        return d
