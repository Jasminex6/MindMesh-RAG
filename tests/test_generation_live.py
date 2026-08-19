"""Day 3 live end-to-end test: real retrieval + real LLM generation.

Requires: Ollama running with nomic-embed-text and llama3.2 models.
Run from project root:  python tests/test_generation_live.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.config import default_config
from medical_rag.pipeline import CorpusPipeline
from medical_rag.vector_repository import ChromaVectorRepository
from medical_rag.hybrid_retrieval import UnifiedRetriever
from medical_rag.generation import GenerationService, GeneratedAnswer
from langchain_ollama import OllamaEmbeddings


def print_answer(label: str, answer: GeneratedAnswer) -> None:
    """Pretty-print a GeneratedAnswer."""
    print(f"\n{'='*80}")
    print(f"TEST: {label}")
    print(f"{'='*80}")
    print(f"Query: {answer.query}")
    print(f"Refused: {answer.refused}")
    print(f"Confidence: {answer.confidence}")

    if answer.refused:
        print(f"Refusal reason: {answer.refusal_reason}")
    else:
        print(f"\nRecommendation:\n  {answer.recommendation[:500]}")
        print(f"\nSupporting Evidence:\n  {answer.supporting_evidence[:300]}")
        print(f"\nCitations ({len(answer.citations)}):")
        for i, c in enumerate(answer.citations, 1):
            status = "[VERIFIED]" if c.verified else "[UNVERIFIED]"
            print(f"  [{i}] {status}")
            print(f"      Claim: {c.claim}")
            print(f"      Chunk: {c.chunk_id}")
            print(f"      Source: {c.document}, Section: {c.section}, Page: {c.page}")
            print(f"      Score: {c.score:.4f}")
        print(f"\nSafety Note: {answer.safety_note}")
    print(f"\n{'-'*80}")


def main():
    print("="*80)
    print("DAY 3: GROUNDED GENERATION LIVE TEST")
    print("="*80)

    # --- Setup ---
    config = default_config(ROOT)
    print(f"\n[Setup] Building corpus...")
    build = CorpusPipeline(config).build()
    print(f"[Setup] {len(build.chunks)} chunks from {len(build.documents)} documents")

    emb_fn = OllamaEmbeddings(model=config.embedding_model)
    collection = f"day3_live_test_{build.corpus_fingerprint}"
    repo = ChromaVectorRepository(config.chroma_dir, collection, emb_fn)
    repo.upsert(build.chunks, batch_size=32)
    retriever = UnifiedRetriever(repo, build.chunks)
    gen = GenerationService(model="llama3.2", temperature=0.1)

    print(f"[Setup] Ready. Collection: {collection}")

    # --- Test Cases ---
    test_cases = [
        {
            "label": "1. CLEARLY SUPPORTED - First-line controller therapy",
            "query": "What is the recommended first-line controller treatment for children with asthma?",
            "strategy": "hybrid_rerank",
        },
        {
            "label": "2. AMBIGUOUS - Monitoring asthma control",
            "query": "How should asthma control be assessed?",
            "strategy": "hybrid_rerank",
        },
        {
            "label": "3. MISSING EVIDENCE - Diabetes (out of scope)",
            "query": "What is the first-line drug treatment for type 2 diabetes in adults?",
            "strategy": "dense",
        },
        {
            "label": "4. OUT-OF-SCOPE - Appendicitis surgery",
            "query": "What is the recommended surgical management for acute appendicitis?",
            "strategy": "dense",
        },
        {
            "label": "5. PATIENT-SPECIFIC - Dosage request",
            "query": "What dose of inhaled corticosteroid should I give my child who weighs 25kg?",
            "strategy": "hybrid_rerank",
        },
        {
            "label": "6. CITATION VALIDITY - Exacerbation management",
            "query": "When is intravenous magnesium sulfate considered for a child with an acute asthma exacerbation?",
            "strategy": "hybrid_rerank",
        },
    ]

    results_log = []

    for tc in test_cases:
        query = tc["query"]
        print(f"\n[Retrieval] '{query[:60]}...'")

        # Retrieve
        search_results = retriever.search(query, strategy=tc["strategy"], top_k=5)
        print(f"[Retrieval] Got {len(search_results)} chunks, top score: "
              f"{search_results[0].score:.4f}" if search_results else "[Retrieval] No results")

        # Generate
        skip_llm = "--skip-llm" in sys.argv
        answer = gen.generate(query, search_results, skip_llm=skip_llm)
        print_answer(tc["label"], answer)

        # Log for JSON export
        results_log.append({
            "test": tc["label"],
            "query": query,
            "strategy": tc["strategy"],
            "num_retrieved": len(search_results),
            "top_score": search_results[0].score if search_results else 0.0,
            "refused": answer.refused,
            "confidence": answer.confidence,
            "num_citations": len(answer.citations),
            "verified_citations": sum(1 for c in answer.citations if c.verified),
            "recommendation_preview": answer.recommendation[:200] if not answer.refused else "",
            "refusal_reason": answer.refusal_reason,
        })

    # --- Summary ---
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for r in results_log:
        status = "REFUSED" if r["refused"] else f"ANSWERED ({r['confidence']})"
        cit_str = f"{r['verified_citations']}/{r['num_citations']} verified" if not r["refused"] else "N/A"
        print(f"  {r['test']}")
        print(f"    Status: {status}  |  Citations: {cit_str}")
    print()

    # Save
    out_path = ROOT / "artifacts" / "day3_live_test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results_log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
