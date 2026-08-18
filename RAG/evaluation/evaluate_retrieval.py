"""RAG Retrieval Evaluator script for benchmarking Precision@3/5, failure taxonomy, and refusal thresholding."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
src_dir = PROJECT_ROOT / "src"
if src_dir.is_dir() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

venv_site = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if venv_site.is_dir() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))

from medical_rag.config import default_config
from medical_rag.pipeline import CorpusPipeline
from medical_rag.vector_repository import ChromaVectorRepository
from medical_rag.hybrid_retrieval import UnifiedRetriever, tokenize
from evidence_panel import render_evidence_panel


def check_text_similarity(t1: str, t2: str) -> float:
    """Compute Jaccard token similarity between two text snippets."""
    toks1 = set(tokenize(t1))
    toks2 = set(tokenize(t2))
    if not toks1 or not toks2:
        return 0.0
    return len(toks1 & toks2) / len(toks1 | toks2)


def is_relevant(query_case: dict, chunk_text: str, chunk_meta: dict) -> bool:
    """Determine if a retrieved chunk is relevant to the query case."""
    if query_case["category"] == "out_of_scope":
        return False

    exp_id = query_case.get("expected_chunk_id")
    if exp_id and chunk_meta.get("chunk_id") == exp_id:
        return True

    text_lower = chunk_text.lower()
    keywords = query_case.get("expected_keywords", [])
    if not keywords:
        return False

    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    req_matches = max(1, min(2, len(keywords)))
    return matches >= req_matches


def detect_query_failures(query_case: dict, results: list[Any]) -> list[str]:
    """Detect failure modes for a query."""
    failures = []
    if query_case["category"] == "out_of_scope":
        if results:
            failures.append("unrefused_out_of_scope")
        return failures

    if not results:
        failures.append("wrong_topic")
        return failures

    # Check for duplicates
    seen_ids = set()
    seen_locs = set()
    has_dup = False
    for i, res in enumerate(results):
        meta = getattr(res, "metadata", {}) if hasattr(res, "metadata") else res.get("metadata", {})
        cid = meta.get("chunk_id")
        loc = (meta.get("document"), meta.get("section"), meta.get("page_start"))

        if cid and cid in seen_ids:
            has_dup = True
        if loc[0] and loc in seen_locs:
            has_dup = True

        if cid:
            seen_ids.add(cid)
        if loc[0]:
            seen_locs.add(loc)

        text_i = getattr(res, "text", "") if hasattr(res, "text") else res.get("text", "")
        for j in range(i):
            text_j = getattr(results[j], "text", "") if hasattr(results[j], "text") else results[j].get("text", "")
            if check_text_similarity(text_i, text_j) > 0.90:
                has_dup = True

    if has_dup:
        failures.append("duplicate_chunks")

    rel_flags = [is_relevant(query_case, getattr(r, "text", ""), getattr(r, "metadata", {})) for r in results]
    if not any(rel_flags):
        failures.append("wrong_topic")
    else:
        top_rel_idx = rel_flags.index(True)
        top_text = getattr(results[top_rel_idx], "text", "")
        if len(top_text.strip()) < 120:
            failures.append("missing_context")

    return failures


def run_evaluation(
    eval_set_path: Path | None = None,
    results_out_path: Path | None = None,
    top_k: int = 5,
    min_score_threshold: float = 0.0,
) -> dict[str, Any]:
    if eval_set_path is None:
        eval_set_path = PROJECT_ROOT / "RAG" / "evaluation" / "eval_set.json"
    if results_out_path is None:
        fname = "results_threshold.json" if min_score_threshold > 0 else "results.json"
        results_out_path = PROJECT_ROOT / "RAG" / "evaluation" / fname

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    config = default_config(PROJECT_ROOT)
    pipeline = CorpusPipeline(config)
    build = pipeline.build()

    from langchain_ollama import OllamaEmbeddings
    embedding_fn = OllamaEmbeddings(model=config.embedding_model)

    collection_name = f"{config.collection_prefix}_{build.corpus_fingerprint}"
    repository = ChromaVectorRepository(
        persist_directory=config.chroma_dir,
        collection_name=collection_name,
        embedding_function=embedding_fn,
    )
    repository.upsert(build.chunks, batch_size=32)
    retriever = UnifiedRetriever(repository, build.chunks)

    eval_results = []
    p3_list = []
    p5_list = []
    failure_counts = {
        "wrong_topic": 0,
        "missing_context": 0,
        "duplicate_chunks": 0,
        "unrefused_out_of_scope": 0,
    }

    mode_str = f"WITH REFUSAL THRESHOLD ({min_score_threshold})" if min_score_threshold > 0 else "RAW BASELINE (No Threshold)"
    print("\n" + "=" * 80)
    print(f"RUNNING RETRIEVAL EVALUATION: {mode_str}...")
    print("=" * 80 + "\n")

    for case in eval_cases:
        query = case["question"]
        results = retriever.search(query, strategy="dense", top_k=max(5, top_k), min_score_threshold=min_score_threshold)

        rel_flags = [
            is_relevant(case, getattr(r, "text", ""), getattr(r, "metadata", {}))
            for r in results
        ]

        if case["category"] != "out_of_scope":
            p3 = sum(rel_flags[:3]) / 3.0
            p5 = sum(rel_flags[:5]) / 5.0
            p3_list.append(p3)
            p5_list.append(p5)
        else:
            # Out of scope: If correctly refused (0 retrieved chunks), score = 1.0 (correct refusal)
            if not results:
                p3 = 1.0
                p5 = 1.0
            else:
                p3 = 0.0
                p5 = 0.0

        q_failures = detect_query_failures(case, results[:5])
        for fail_type in q_failures:
            if fail_type in failure_counts:
                failure_counts[fail_type] += 1

        retrieved_records = []
        for rank, res in enumerate(results[:5], start=1):
            meta = getattr(res, "metadata", {})
            retrieved_records.append({
                "rank": rank,
                "chunk_id": meta.get("chunk_id", "N/A"),
                "score": round(getattr(res, "score", 0.0), 4),
                "source": meta.get("document", "N/A"),
                "section": meta.get("section", "N/A"),
                "page": f"{meta.get('page_start', 'N/A')}",
                "text_excerpt": getattr(res, "text", "")[:300],
                "relevant": rel_flags[rank - 1] if rank <= len(rel_flags) else False,
                "human_relevance_label": "yes" if (rank <= len(rel_flags) and rel_flags[rank - 1]) else "no",
            })

        status_str = "REFUSED (Out of Scope)" if (case["category"] == "out_of_scope" and not results) else "PROCESSED"

        eval_results.append({
            "id": case["id"],
            "category": case["category"],
            "question": query,
            "status": status_str,
            "expected_source": case.get("expected_source_document"),
            "expected_section": case.get("expected_page_section"),
            "precision_at_3": round(p3, 4),
            "precision_at_5": round(p5, 4),
            "detected_failures": q_failures,
            "retrieved_chunks": retrieved_records,
            "evidence_panel": render_evidence_panel(results[:5]),
        })

    avg_p3 = sum(p3_list) / max(len(p3_list), 1)
    avg_p5 = sum(p5_list) / max(len(p5_list), 1)

    summary_payload = {
        "config": {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "embedding_model": config.embedding_model,
            "min_score_threshold": min_score_threshold,
            "collection_name": collection_name,
        },
        "metrics": {
            "questions_evaluated": len(eval_cases),
            "in_scope_evaluated": len([c for c in eval_cases if c['category'] != 'out_of_scope']),
            "average_precision_at_3": round(avg_p3, 4),
            "average_precision_at_5": round(avg_p5, 4),
            "failures": failure_counts,
        },
        "results": eval_results,
    }

    results_out_path.parent.mkdir(parents=True, exist_ok=True)
    results_out_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("-" * 50)
    print(f"Mode: {mode_str}")
    print(f"Questions evaluated: {len(eval_cases)}")
    print(f"Average Precision@3: {avg_p3:.4f}")
    print(f"Average Precision@5: {avg_p5:.4f}")
    print("\nFailures:")
    print(f"Wrong topic: {failure_counts['wrong_topic']}")
    print(f"Missing context: {failure_counts['missing_context']}")
    print(f"Duplicate chunks: {failure_counts['duplicate_chunks']}")
    print(f"Unrefused out-of-scope: {failure_counts['unrefused_out_of_scope']}")
    print("-" * 50)
    print(f"\nResults saved to: {results_out_path}")

    return summary_payload


if __name__ == "__main__":
    thresh = 0.72 if "--threshold" in sys.argv else 0.0
    run_evaluation(min_score_threshold=thresh)
