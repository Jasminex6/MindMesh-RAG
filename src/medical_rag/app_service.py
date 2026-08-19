"""Unified RAG Application Service Boundary."""

from __future__ import annotations

import time
from typing import Any

from .canonicalization import canonicalize_query
from .contracts import CitationContract, EvidenceContract, RAGResponse
from .generation import GeneratedAnswer, GenerationService
from .hybrid_retrieval import UnifiedRetriever, format_retrieval_query, normalize_medical_typos
from .intent_classifier import IntentClassifier, AGE_MANDATORY_INTENTS, summarize_for_retrieval
from .query_decomposition import is_compound_query, retrieve_multi_question
from .router import route_query, RoutingDecision


class RagApplicationService:
    """Application facade coordinating safety router, retrieval, generation, contracts, memory, and persistence."""

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
        slots: dict | None = None,
        chat_history: list[dict] | None = None,
    ) -> RAGResponse:
        """Execute clinical RAG pipeline and return canonical RAGResponse contract."""
        start_time = time.perf_counter()
        q_raw = query.strip()
        timing: dict[str, float] = {}

        if slots is None:
            slots = {}

        if self.persistence and conversation_id and not chat_history:
            chat_history = self.persistence.get_conversation_messages(conversation_id)

        # 1. Preprocessing & Multi-Turn Conversational Query Rewriting
        prep_start = time.perf_counter()
        q_normalized = normalize_medical_typos(q_raw)
        canonical_q = canonicalize_query(q_normalized, chat_history=chat_history)
        timing["prep_ms"] = (time.perf_counter() - prep_start) * 1000.0

        # 2. Pre-Flight Safety Gate & Router (evaluated on canonical query)
        gate_start = time.perf_counter()
        decision: RoutingDecision = route_query(canonical_q)
        timing["safety_gate_ms"] = (time.perf_counter() - gate_start) * 1000.0

        if decision.status == "BLOCKED":
            safety_msg = (
                decision.safety_message_ar if language == "ar" and decision.safety_message_ar
                else decision.safety_message_en
            )
            response = RAGResponse(
                status="REFUSAL",
                language=language,
                query=q_raw,
                resolved_query=canonical_q,
                recommendation=f"Refused ({decision.category} Guardrail Triggered)",
                evidence=(),
                citations=(),
                confidence="Insufficient Evidence",
                safety_message=safety_msg,
                timing_ms={"total_ms": (time.perf_counter() - start_time) * 1000.0},
            )
            if self.persistence and conversation_id:
                self.persistence.add_message(conversation_id, "user", q_raw)
                self.persistence.add_message(conversation_id, "assistant", response.recommendation, response.to_dict())
            return response

        if decision.status == "CLARIFY":
            response = RAGResponse(
                status="CLARIFY",
                language=language,
                query=q_raw,
                resolved_query=canonical_q,
                recommendation=decision.clarification_question or "Please clarify your question.",
                evidence=(),
                citations=(),
                confidence="Needs Clarification",
                safety_message="Age group or population context required before guideline retrieval.",
                timing_ms={"total_ms": (time.perf_counter() - start_time) * 1000.0},
            )
            if self.persistence and conversation_id:
                self.persistence.add_message(conversation_id, "user", q_raw)
                self.persistence.add_message(conversation_id, "assistant", response.recommendation, response.to_dict())
            return response

        # 3. Intent Classification & Hard Refusal Check
        classifier = IntentClassifier()
        intent = classifier.classify(canonical_q)

        if intent.requires_refusal or intent.requires_emergency:
            refusal_reason = intent.refusal_reason or intent.emergency_response or "Request refused per safety policy."
            response = RAGResponse(
                status="REFUSAL",
                language=language,
                query=q_raw,
                resolved_query=canonical_q,
                recommendation="Refused (Safety Protocol Triggered)",
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

        # 4. Mandatory Age Slot Guard
        if intent.category in AGE_MANDATORY_INTENTS and not slots.get("age_band"):
            # Inspect canonical query string for implied age
            q_low = canonical_q.lower()
            if "under 6" in q_low or "under-5" in q_low or "infant" in q_low or "toddler" in q_low:
                slots["age_band"] = "under_6"
            elif "6-11" in q_low or "6 to 11" in q_low:
                slots["age_band"] = "children_6_11"
            elif "12+" in q_low or "adult" in q_low or "adolescent" in q_low:
                slots["age_band"] = "adults_adolescents"

            if not slots.get("age_band"):
                response = RAGResponse(
                    status="CLARIFY",
                    language=language,
                    query=q_raw,
                    resolved_query=canonical_q,
                    recommendation="Are you asking about children under 6, children aged 6–11, or adolescents/adults (12+)? Asthma guidance differs by age group.",
                    evidence=(),
                    citations=(),
                    confidence="Needs Clarification",
                    safety_message="Age group or population context required before guideline retrieval.",
                    timing_ms={"total_ms": (time.perf_counter() - start_time) * 1000.0},
                )
                if self.persistence and conversation_id:
                    self.persistence.add_message(conversation_id, "user", q_raw)
                    self.persistence.add_message(conversation_id, "assistant", response.recommendation, response.to_dict())
                return response

        # 5. Query Enrichment & Multi-Question Retrieval
        enriched_query = summarize_for_retrieval(canonical_q, slots, intent.category)
        age_band = slots.get("age_band")

        ret_start = time.perf_counter()
        if is_compound_query(enriched_query):
            multi_res = retrieve_multi_question(
                enriched_query,
                retriever=self.retriever,
                strategy=strategy,
                top_k_per_question=5,
                final_top_k=5,
            )
            results = multi_res.merged_results
        else:
            search_q = format_retrieval_query(enriched_query)
            results = self.retriever.search(search_q, strategy=strategy, top_k=5)
        timing["retrieval_ms"] = (time.perf_counter() - ret_start) * 1000.0

        # 6. Generation & Grounding
        gen_start = time.perf_counter()
        answer: GeneratedAnswer = self.gen_service.generate(
            canonical_q,
            results,
            age_band=age_band,
            intent_category=intent.category,
            slots=slots,
            chat_history=chat_history,
        )
        timing["generation_ms"] = (time.perf_counter() - gen_start) * 1000.0

        # 7. Clean output handling on Refusal / Insufficient Evidence
        if answer.refused:
            timing["total_ms"] = (time.perf_counter() - start_time) * 1000.0
            response = RAGResponse(
                status="REFUSAL",
                language=language,
                query=q_raw,
                resolved_query=canonical_q,
                recommendation=answer.recommendation or "Refused (Insufficient Evidence)",
                evidence=(),
                citations=(),
                confidence="Insufficient Evidence",
                safety_message=answer.safety_note or answer.refusal_reason,
                timing_ms=timing,
            )
            if self.persistence and conversation_id:
                self.persistence.add_message(conversation_id, "user", q_raw)
                self.persistence.add_message(conversation_id, "assistant", response.recommendation, response.to_dict())
            return response

        # 8. Convert to RAGResponse contract for successful answer
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
            status="ANSWER",
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
