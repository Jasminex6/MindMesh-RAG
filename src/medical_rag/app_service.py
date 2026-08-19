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

        # 1. Stage 1: Bilingual NLP Query Processing & Reconstruction
        prep_start = time.perf_counter()
        from .nlp_processor import NlpQueryProcessor
        nlp_proc = NlpQueryProcessor()
        nlp_res = nlp_proc.process(q_raw, chat_history=chat_history)
        
        language = nlp_res.language
        canonical_q = nlp_res.reconstructed_query or q_raw
        if nlp_res.implied_age_band and not slots.get("age_band"):
            slots["age_band"] = nlp_res.implied_age_band
        timing["prep_ms"] = (time.perf_counter() - prep_start) * 1000.0

        # Video Trigger Detection
        v_url, v_title = None, None
        try:
            try:
                from .video_triggers import detect_video
            except ImportError:
                from video_triggers import detect_video

            video_trig = detect_video(q_raw) or detect_video(canonical_q)
            if video_trig:
                v_url = str(video_trig.video_path)
                v_title = str(video_trig.title)
            print(f"[DEBUG_ASK] q_raw={q_raw} canonical_q={canonical_q} v_url={v_url}")
        except Exception as e:
            print(f"[app_service] Video trigger notice: {e}")

        # 2. Pre-Flight Safety Gate & Router (evaluated on reconstructed query)
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
                video_url=v_url,
                video_title=v_title,
                timing_ms={"total_ms": (time.perf_counter() - start_time) * 1000.0},
            )
            if self.persistence and conversation_id:
                self.persistence.add_message(conversation_id, "user", q_raw)
                self.persistence.add_message(conversation_id, "assistant", response.recommendation, response.to_dict())
            return response

        if decision.status == "CLARIFY" and not v_url:
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
                video_url=v_url,
                video_title=v_title,
                timing_ms={"total_ms": (time.perf_counter() - start_time) * 1000.0},
            )
            if self.persistence and conversation_id:
                self.persistence.add_message(conversation_id, "user", q_raw)
                self.persistence.add_message(conversation_id, "assistant", response.recommendation, response.to_dict())
            return response

        # 3. Intent Classification & Hard Refusal Check
        classifier = IntentClassifier()
        intent = classifier.classify(canonical_q)

        if intent.requires_emergency or intent.category == "emergency":
            emergency_card = intent.emergency_response or intent.refusal_reason or (
                "🚨 **EMERGENCY: Seek Immediate Medical Care**\n\n"
                "**Call your local emergency services (123, 911, 999) or transport the child to the nearest emergency department immediately.** Cyanosis (turning blue) and severe respiratory distress are life-threatening signs."
            )
            response = RAGResponse(
                status="EMERGENCY",
                language=language,
                query=q_raw,
                resolved_query=canonical_q,
                recommendation=emergency_card,
                evidence=(),
                citations=(),
                confidence="Ungrounded",
                safety_message=emergency_card,
                timing_ms={"total_ms": (time.perf_counter() - start_time) * 1000.0},
            )
            if self.persistence and conversation_id:
                self.persistence.add_message(conversation_id, "user", q_raw)
                self.persistence.add_message(conversation_id, "assistant", response.recommendation, response.to_dict())
            return response

        if (intent.requires_refusal or intent.category == "out_of_scope") and not v_url:
            out_card = intent.refusal_reason or (
                "I am a clinical decision support assistant specialized in pediatric asthma and acute bronchiolitis management based on WHO and NICE NG245 guidelines.\n\n"
                "I can only assist with medical questions regarding pediatric respiratory symptoms, diagnostic pathways, medication stepping (MART/ICS), and acute exacerbation protocols. Please enter a clinical question."
            )
            response = RAGResponse(
                status="OUT_OF_SCOPE",
                language=language,
                query=q_raw,
                resolved_query=canonical_q,
                recommendation=out_card,
                evidence=(),
                citations=(),
                confidence="Ungrounded",
                safety_message=out_card,
                video_url=v_url,
                video_title=v_title,
                timing_ms={"total_ms": (time.perf_counter() - start_time) * 1000.0},
            )
            if self.persistence and conversation_id:
                self.persistence.add_message(conversation_id, "user", q_raw)
                self.persistence.add_message(conversation_id, "assistant", response.recommendation, response.to_dict())
            return response

        # 4. Mandatory Age Slot Guard
        if intent.category in AGE_MANDATORY_INTENTS and not slots.get("age_band") and not v_url:
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
                    confidence="Ungrounded",
                    safety_message="Age group or population context required before guideline retrieval.",
                    video_url=v_url,
                    video_title=v_title,
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

        # 7. Clean output handling on Refusal / Insufficient Evidence (NO_EVIDENCE)
        if answer.refused:
            timing["total_ms"] = (time.perf_counter() - start_time) * 1000.0
            if v_url:
                no_evidence_card = (
                    f"**فيديو توضيحي: {v_title}**\n\nإليك فيديو تدريبي توضيحي لإرشادات تقنيات التنفس واستخدام البخاخ."
                    if language == "ar" else
                    f"**Demonstration Video: {v_title}**\n\nHere is an instructional demonstration video for breathing exercises and asthma management."
                )
            else:
                no_evidence_card = (
                    "**لم يتم العثور على إرشادات محددة في الدليل**\n\n"
                    "لا تحتوي إرشادات منظمة الصحة العالمية وNICE NG245 المحملة على أدلة سريرية كافية للإجابة على هذا السيناريو السريري.\n\n"
                    "**الخطوات التالية الموصى بها:**\n"
                    "- إعادة صياغة السؤال باستخدام مصطلحات سريرية قياسية (مثل عمر المريض، شدة الأعراض، أو الفئة الدوائية).\n"
                    "- استشارة أخصائي أمراض الصدر للأطفال."
                    if language == "ar" else
                    "**No Specific Guideline Guidance Found**\n\n"
                    "The loaded WHO and NICE NG245 pediatric guidelines do not contain sufficiently specific evidence to answer this particular clinical scenario.\n\n"
                    "**Recommended Next Steps:**\n"
                    "- Rephrase the query with standard clinical terms (e.g., patient age, specific symptom severity, or drug class).\n"
                    "- Consult local hospital formularies or refer to a pediatric respiratory specialist."
                )
            response = RAGResponse(
                status="NO_EVIDENCE",
                language=language,
                query=q_raw,
                resolved_query=canonical_q,
                recommendation=no_evidence_card,
                evidence=(),
                citations=(),
                confidence="Ungrounded",
                safety_message=no_evidence_card,
                video_url=v_url,
                video_title=v_title,
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

        # Use precomputed v_url and v_title from top of ask()

        response = RAGResponse(
            status="SUCCESS",
            language=language,
            query=q_raw,
            resolved_query=canonical_q,
            recommendation=answer.recommendation,
            evidence=tuple(evidence_list),
            citations=tuple(citation_list),
            confidence=answer.confidence,
            safety_message=answer.safety_note,
            video_url=v_url,
            video_title=v_title,
            timing_ms=timing,
        )

        if self.persistence and conversation_id:
            self.persistence.add_message(conversation_id, "user", q_raw)
            self.persistence.add_message(conversation_id, "assistant", response.recommendation, response.to_dict())

        return response
