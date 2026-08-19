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
    status: str                       # ANSWER | REFUSAL | NEEDS_CLARIFICATION | ERROR
    language: str                     # "en" | "ar"
    query: str
    resolved_query: str
    recommendation: str
    evidence: tuple[EvidenceContract, ...]
    citations: tuple[CitationContract, ...]
    confidence: str                   # High | Medium | Low | Insufficient Evidence
    safety_message: str
    clarification_question: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
