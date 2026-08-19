"""Interactive Clinical Decision Support RAG CLI."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.config import default_config
from medical_rag.pipeline import CorpusPipeline
from medical_rag.vector_repository import ChromaVectorRepository
from medical_rag.hybrid_retrieval import UnifiedRetriever
from medical_rag.generation import GenerationService, GeneratedAnswer
from medical_rag.router import route_query
from langchain_ollama import OllamaEmbeddings


def display_answer(answer: GeneratedAnswer) -> None:
    """Format and display a GeneratedAnswer according to hackathon specifications."""
    print("\n" + "=" * 80)
    print(f"CLINICAL QUESTION: {answer.query}")
    print("=" * 80)

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
        print("\n" + "=" * 80 + "\n")
        return

    # 1. RECOMMENDATION
    print("\n1. RECOMMENDATION:")
    print(f"   {answer.recommendation.strip()}")

    # 2. SUPPORTING EVIDENCE
    print("\n2. SUPPORTING EVIDENCE:")
    ev_text = str(answer.supporting_evidence).strip() if answer.supporting_evidence else ""
    if ev_text:
        for line in ev_text.splitlines():
            line_str = line.replace("•", "-")
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
    print(f"   Safety Disclaimer: {answer.safety_note if answer.safety_note else 'Grounded in official guideline evidence. Clinical judgment required.'}")

    print("\n" + "=" * 80 + "\n")


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


def ask_question(query: str, retriever: UnifiedRetriever, gen_service: GenerationService):
    # --- Safety / ambiguity gate — runs BEFORE retrieval ---
    decision = route_query(query)

    if decision.status == "BLOCKED":
        print("\n" + "=" * 80)
        print(f"CLINICAL QUESTION: {query}")
        print("=" * 80)
        print(f"\n[BLOCKED — {decision.category}]")
        print(f"   {decision.safety_message_en}")
        print(f"   {decision.safety_message_ar}")
        print("\n" + "=" * 80 + "\n")
        return None

    if decision.status == "CLARIFY":
        print("\n" + "=" * 80)
        print(f"CLINICAL QUESTION: {query}")
        print("=" * 80)
        print("\n[CLARIFICATION NEEDED]")
        print(f"   {decision.clarification_question}")
        print("\n" + "=" * 80 + "\n")
        return None

    # --- decision.status == "PROCEED" — continue as before ---
    print(f"[Searching guidelines for: '{query}'...]")
    results = retriever.search(query, strategy="hybrid_rerank", top_k=5)
    answer = gen_service.generate(query, results)
    display_answer(answer)
    return answer


def main():
    retriever, gen_service = setup_pipeline()

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        query = " ".join(sys.argv[1:])
        ask_question(query, retriever, gen_service)
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
            ask_question(user_input, retriever, gen_service)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo.")
            break


if __name__ == "__main__":
    main()
