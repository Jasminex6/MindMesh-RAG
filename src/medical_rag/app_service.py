"""Unified RAG Application Service Boundary."""

from __future__ import annotations

import time
from typing import Any

from .canonicalization import canonicalize_query
from .contracts import CitationContract, EvidenceContract, RAGResponse
from .generation import GeneratedAnswer, GenerationService
from .hybrid_retrieval import UnifiedRetriever, format_retrieval_query, normalize_medical_typos
from .intent import QueryIntent, classify_query_intent


class RagApplicationService:
    """Application facade coordinating safety, retrieval, generation, contracts, memory, and persistence."""

    def __init__(
        self,
        retriever: UnifiedRetriever,
        generation_service: GenerationService,
        persistence_store: Any | None = None,
        profiler: Any | None = None,
    ):
        self.retriever = retriever
        self.gen_service = generation_service
        self.persistence = persistence_store
        self.profiler = profiler

    def ask(
        self,
        query: str,
        conversation_id: str | None = None,
        strategy: str = "hybrid_rerank",
        language: str = "en",
        chat_history: list[dict] | None = None,
    ) -> RAGResponse:
        """Execute clinical RAG pipeline and return canonical RAGResponse contract."""
        start_time = time.perf_counter()
        q_raw = query.strip()
        timing: dict[str, float] = {}

        # Load chat history from persistence if conversation_id is provided and history not passed explicitly
        if self.persistence and conversation_id and not chat_history:
            chat_history = self.persistence.get_conversation_messages(conversation_id)

        # 1. Early Safety & Out-of-Scope Gate
        gate_start = time.perf_counter()
        intent = classify_query_intent(q_raw)
        timing["safety_gate_ms"] = (time.perf_counter() - gate_start) * 1000.0

        if intent in (QueryIntent.OUT_OF_SCOPE, QueryIntent.PATIENT_SPECIFIC):
            refusal_reason = (
                "Patient-specific dosage or prescribing requests require direct clinical calculation. Clinical consultation required."
                if intent == QueryIntent.PATIENT_SPECIFIC
                else "This query is out of scope for the Pediatric Asthma Clinical Decision Support system (zero guideline retrieval performed)."
            )
            response = RAGResponse(
                status="REFUSAL",
                language=language,
                query=q_raw,
                resolved_query=q_raw,
                recommendation="Refused (Safety Guardrail Triggered)",
                evidence=(),
                citations=(),
                confidence="Insufficient Evidence",
                safety_message=refusal_reason,
                timing_ms={"total_ms": (time.perf_counter() - start_time) * 1000.0},
            )
            if self.persistence and conversation_id:
                self.persistence.add_message(conversation_id, "user", q_raw)
                self.persistence.add_message(conversation_id, "assistant", response.recommendation, response.to_dict())
            return response

        # 2. Preprocessing & Canonicalization (with Multi-Turn Follow-Up Resolution)
        prep_start = time.perf_counter()
        q_normalized = normalize_medical_typos(q_raw)
        canonical_q = canonicalize_query(q_normalized, chat_history=chat_history)
        search_q = format_retrieval_query(canonical_q)
        timing["prep_ms"] = (time.perf_counter() - prep_start) * 1000.0

        # 3. Retrieval
        ret_start = time.perf_counter()
        results = self.retriever.search(search_q, strategy=strategy, top_k=5)
        timing["retrieval_ms"] = (time.perf_counter() - ret_start) * 1000.0

        # 4. Generation & Grounding
        gen_start = time.perf_counter()
        answer: GeneratedAnswer = self.gen_service.generate(canonical_q, results)
        timing["generation_ms"] = (time.perf_counter() - gen_start) * 1000.0

        # 5. Convert to RAGResponse contract
        evidence_list = []
        for r in results:
            evidence_list.append(
                EvidenceContract(
                    chunk_id=str(r.metadata.get("chunk_id", "")),
                    text=r.text,
                    document_name=str(r.metadata.get("document", "")),
                    section_title=str(r.metadata.get("section", "")),
                    page_number=str(r.metadata.get("page", "")),
                    retrieval_score=float(r.score),
                    verification_status=any(c.chunk_id == str(r.metadata.get("chunk_id", "")) and c.verified for c in answer.citations),
                )
            )

        citation_list = []
        for c in answer.citations:
            citation_list.append(
                CitationContract(
                    claim=c.claim,
                    chunk_id=c.chunk_id,
                    document=c.document,
                    section=c.section,
                    page=c.page,
                    score=c.score,
                    verified=c.verified,
                )
            )

        timing["total_ms"] = (time.perf_counter() - start_time) * 1000.0

        response = RAGResponse(
            status="REFUSAL" if answer.refused else "ANSWER",
            language=language,
            query=q_raw,
            resolved_query=canonical_q,
            recommendation=answer.recommendation,
            evidence=tuple(evidence_list),
            citations=tuple(citation_list),
            confidence=answer.confidence,
            safety_message=answer.safety_note,
            timing_ms=timing,
        )

        if self.persistence and conversation_id:
            self.persistence.add_message(conversation_id, "user", q_raw)
            self.persistence.add_message(conversation_id, "assistant", response.recommendation, response.to_dict())

        return response
