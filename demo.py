"""Interactive Clinical Decision Support RAG CLI."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.config import default_config
from medical_rag.pipeline import CorpusPipeline
from medical_rag.vector_repository import ChromaVectorRepository
from medical_rag.hybrid_retrieval import UnifiedRetriever
from medical_rag.generation import GenerationService, GeneratedAnswer
from medical_rag.query_rewriter import ConversationalQueryRewriter
from medical_rag.intent_classifier import IntentClassifier, AGE_MANDATORY_INTENTS, AGE_BAND_OPTIONS, summarize_for_retrieval
from medical_rag.router import route_query
from langchain_ollama import OllamaEmbeddings


def display_answer(answer: GeneratedAnswer) -> None:
    """Format and display a GeneratedAnswer according to calibrated hackathon specifications."""
    divider = "=" * 70
    print("\n" + divider)
    print(f"CLINICAL QUESTION: {answer.query}")
    print(divider)

    if answer.refused:
        print("\n1. RECOMMENDATION:")
        print("   Refused (Safety / Grounding Guardrail Triggered)")

        print("\n2. SUPPORTING EVIDENCE:")
        print("   (No evidence used - query refused)")

        print("\n3. CITATIONS:")
        print("   (None)")

        print("\n4. CONFIDENCE & SAFETY:")
        print(f"   Confidence: {answer.confidence}")
        print(f"   Safety Disclaimer: {answer.refusal_reason}")
        print("\n" + divider + "\n")
        return

    # 1. RECOMMENDATION
    print("\n1. RECOMMENDATION:")
    print(f"   {answer.recommendation.strip()}")

    # 2. SUPPORTING EVIDENCE
    print("\n2. SUPPORTING EVIDENCE:")
    ev_text = str(answer.supporting_evidence).strip() if answer.supporting_evidence else ""
    if ev_text:
        for line in ev_text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("•"):
                line_str = "- " + line_str[1:].strip()
            elif not line_str.startswith("-"):
                line_str = "- " + line_str
            print(f"   {line_str}")
    else:
        print("   (Short excerpts from cited guideline chunks below)")

    # 3. CITATIONS
    print(f"\n3. CITATIONS ({len(answer.citations)} cited claims):")
    if not answer.citations:
        print("   (No explicit citations extracted)")
    for i, c in enumerate(answer.citations, 1):
        status = "[VERIFIED]" if c.verified else "[UNVERIFIED]"
        print(f"   [{i}] {status}")
        print(f"       Claim: {c.claim}")
        print(f"       Chunk ID: {c.chunk_id}")
        if c.verified:
            print(f"       Document: {c.document}")
            print(f"       Section: {c.section}")
            print(f"       Page: {c.page}")
            print(f"       Retrieval Score: {c.score:.4f}")
        else:
            print("       Warning: Cited chunk ID was NOT found in retrieved evidence.")

    # 4. CONFIDENCE & SAFETY
    print("\n4. CONFIDENCE & SAFETY:")
    print(f"   Confidence: {answer.confidence} (Derived from retrieval quality)")
    safety_msg = answer.safety_note if answer.safety_note else "Grounded in official guideline evidence. Clinical judgment required."
    print(f"   Safety Disclaimer: {safety_msg}")

    print("\n" + divider + "\n")


def setup_pipeline():
    """Initialize the RAG pipeline components."""
    print("\n[Initializing pipeline...]")
    config = default_config(ROOT)
    build = CorpusPipeline(config).build()

    emb_fn = OllamaEmbeddings(model=config.embedding_model)
    collection_name = f"demo_collection_{build.corpus_fingerprint}"
    repo = ChromaVectorRepository(config.chroma_dir, collection_name, emb_fn)
    repo.upsert(build.chunks, batch_size=32)
    retriever = UnifiedRetriever(repo, build.chunks)
    gen_service = GenerationService(model="llama3.2", temperature=0.1)

    print(f"[Pipeline Ready - {len(build.chunks)} chunks loaded from WHO & NICE guidelines]\n")
    return retriever, gen_service


def split_into_questions(query: str) -> list[str]:
    """Split a multi-question query into individual sub-questions if present."""
    q_str = query.strip()
    if not q_str:
        return []

    # Check for numbered list pattern: 1. ... 2. ...
    numbered_parts = re.split(r"(?:\r?\n|\s+)(?=\d+[\.\)]\s+)", q_str)
    numbered_parts = [re.sub(r"^\d+[\.\)]\s*", "", p).strip() for p in numbered_parts if p.strip()]
    if len(numbered_parts) > 1:
        return numbered_parts

    # Check for multiple question marks: Q1? Q2?
    if q_str.count("?") > 1:
        parts = [p.strip() + "?" for p in q_str.split("?") if p.strip()]
        parts = [p[:-1] if p.endswith("??") else p for p in parts if p.strip()]
        if len(parts) > 1:
            return parts

    # Check for multiple lines
    lines = [line.strip() for line in q_str.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    return [q_str]


def ask_question(
    query: str,
    retriever: UnifiedRetriever,
    gen_service: GenerationService,
    slots: dict | None = None,
    chat_history: list[dict[str, str]] | list[tuple[str, str]] | None = None,
    skip_llm_rewriter: bool = False,
    interactive: bool = True,
) -> object:
    """Classify query, enforce mandatory slots, run retrieval and generation."""
    if slots is None:
        slots = {}

    # 1. Stage 1: Bilingual NLP Query Processing & Topic Shift Detection
    from medical_rag.nlp_processor import NlpQueryProcessor
    nlp_proc = NlpQueryProcessor()
    nlp_res = nlp_proc.process(query, chat_history=chat_history)

    processed_query = nlp_res.reconstructed_query or query
    if nlp_res.implied_age_band and not slots.get("age_band"):
        slots["age_band"] = nlp_res.implied_age_band

    if processed_query != query:
        print(f"[NLP Query Processor] Reconstructed query:\n  Original: '{query}'\n  Reconstructed: '{processed_query}' (Language: {nlp_res.language}, Topic Shift: {nlp_res.is_topic_shift})")

    # 2. Sub-millisecond Pre-Flight Safety Gate & Router
    routing = route_query(processed_query)
    if routing.status == "BLOCKED":
        refusal_reason = routing.safety_message_en
        refused_answer = GeneratedAnswer(
            query=query,
            recommendation=f"Refused ({routing.category} Guardrail Triggered)",
            supporting_evidence="(RAG retrieval bypassed - safety refusal triggered)",
            citations=[],
            confidence="Insufficient Evidence",
            safety_note=refusal_reason,
            refused=True,
            refusal_reason=refusal_reason,
        )
        display_answer(refused_answer)
        if chat_history is not None and isinstance(chat_history, list):
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": refusal_reason})
        return refused_answer

    # 2. Intent Classifier & Hard Refusal
    classifier = IntentClassifier()
    intent = classifier.classify(processed_query)

    if intent.requires_refusal or intent.requires_emergency:
        refusal_reason = intent.refusal_reason or intent.emergency_response or "Request refused per safety policy."
        refused_answer = GeneratedAnswer(
            query=query,
            recommendation="Refused (Safety Protocol Triggered)",
            supporting_evidence="(RAG retrieval bypassed - safety refusal triggered)",
            citations=[],
            confidence="Insufficient Evidence",
            safety_note=refusal_reason,
            refused=True,
            refusal_reason=refusal_reason,
        )
        display_answer(refused_answer)
        if chat_history is not None and isinstance(chat_history, list):
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": refusal_reason})
        return refused_answer

    # 3. Mandatory Age Slot Guard (deterministic, not LLM-controlled)
    if intent.category in AGE_MANDATORY_INTENTS and not slots.get("age_band"):
        print(f"\n[Intent: '{intent.category}' — age is required for this query]")
        print("  What is the age group of the person this question is about?")
        print("    1. Under 6 years old")
        print("    2. 6–11 years old")
        print("    3. 12 years and older (adolescent/adult)")
        if interactive and sys.stdin.isatty():
            try:
                raw = input("  Enter 1, 2, or 3 > ").strip()
            except (EOFError, KeyboardInterrupt, OSError):
                raw = ""
            age_band = AGE_BAND_OPTIONS.get(raw.lower(), None)
            if age_band:
                slots["age_band"] = age_band
                print(f"  [Age band recorded: {age_band}]\n")
            else:
                print("  [Age band not recognised — proceeding without age filter]\n")

    # 4. Adaptive Clarification Prompting (non-age slots)
    if intent.requires_clarification:
        non_age_qs = [q for q in intent.adaptive_questions if q.id != "age_group"]
        if non_age_qs:
            print(f"[Intent: '{intent.category}' — adaptive clinical context / clarification]")
            for q in non_age_qs:
                if slots.get(q.id):
                    continue
                print(f"  • {q.question}")
                for idx, opt in enumerate(q.options, 1):
                    print(f"    {idx}. {opt}")
                if interactive and sys.stdin.isatty():
                    try:
                        ans = input("  Select option number (or press Enter to skip) > ").strip()
                        if ans.isdigit() and 1 <= int(ans) <= len(q.options):
                            chosen = q.options[int(ans) - 1]
                            slots[q.id] = chosen
                            print(f"  [Selected: {chosen}]\n")
                    except (EOFError, KeyboardInterrupt, OSError):
                        print()

    # 5. Build enriched query string for retrieval
    extra_context = []
    for k, v in slots.items():
        if k != "age_band" and v:
            extra_context.append(f"{k}: {v}")
    if extra_context:
        processed_query = f"{processed_query} [{', '.join(extra_context)}]"

    enriched_query = summarize_for_retrieval(processed_query, slots, intent.category)
    age_band = slots.get("age_band")

    # 6. Sub-question decomposition → retrieval → generation
    sub_questions = split_into_questions(enriched_query)
    answers = []

    if len(sub_questions) > 1:
        print(f"\n[Detected multi-question query: processing {len(sub_questions)} sub-questions...]\n")

    for sub_q in sub_questions:
        print(f"[Searching guidelines for: '{sub_q}'...]")
        results = retriever.search(sub_q, strategy="hybrid_rerank", top_k=5)
        answer = gen_service.generate(
            sub_q, results, age_band=age_band, intent_category=intent.category, slots=slots, chat_history=chat_history
        )
        display_answer(answer)
        answers.append(answer)

    final_result = answers if len(answers) > 1 else (answers[0] if answers else None)

    if chat_history is not None and isinstance(chat_history, list):
        chat_history.append({"role": "user", "content": query})
        rec_text = ""
        if isinstance(final_result, GeneratedAnswer):
            rec_text = final_result.recommendation
        elif isinstance(final_result, list):
            rec_text = "\n".join(a.recommendation for a in final_result if isinstance(a, GeneratedAnswer))
        chat_history.append({"role": "assistant", "content": rec_text})

    return final_result


def main():
    retriever, gen_service = setup_pipeline()
    chat_history: list[dict[str, str]] = []

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        query = " ".join(sys.argv[1:])
        ask_question(query, retriever, gen_service, chat_history=chat_history)
        return

    print("Interactive Demo Mode. Type a clinical question or 'exit'/'quit' to stop.\n")
    sample_questions = [
        "1. What is the recommended first-line controller treatment for children with asthma?",
        "2. How should asthma control be monitored during follow-up?",
        "3. What dose of inhaled corticosteroid should I give my child who weighs 25kg?",
        "4. What is the first-line drug treatment for type 2 diabetes in adults?",
    ]
    print("Sample questions you can try:")
    for sq in sample_questions:
        print(f"  {sq}")
    print()

    while True:
        try:
            user_input = input("Enter clinical question > ").strip()
            if not user_input or user_input.lower() in ("exit", "quit", "q"):
                print("Exiting demo.")
                break
            ask_question(user_input, retriever, gen_service, chat_history=chat_history)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo.")
            break
        except Exception as e:
            print(f"\n[Error: {e}] Please try another question.\n")


if __name__ == "__main__":
    main()
