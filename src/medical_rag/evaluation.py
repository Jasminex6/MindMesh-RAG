"""Auditable retrieval logging and human labeling helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .models import SearchResult


def save_retrieval_log(
    path: Path, runs: Iterable[tuple[str, list[SearchResult]]], config: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "queries": [
            {"query": query, "results": [result.to_dict() for result in results]}
            for query, results in runs
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_audit_template(
    path: Path, runs: Iterable[tuple[str, list[SearchResult]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query",
        "top_k",
        "top_score",
        "retrieved_documents",
        "relevant_at_k",
        "best_chunk_ids",
        "citation_metadata_correct",
        "failure_category",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for query, results in runs:
            writer.writerow({
                "query": query,
                "top_k": len(results),
                "top_score": f"{results[0].score:.4f}" if results else "",
                "retrieved_documents": ", ".join(sorted({
                    str(result.metadata.get("document", "")) for result in results
                })),
                "relevant_at_k": "",
                "best_chunk_ids": "",
                "citation_metadata_correct": "",
                "failure_category": "",
                "notes": "",
            })


def precision_at_k(relevant_flags: Iterable[bool], k: int) -> float:
    """Calculate Precision@K from human relevance labels."""
    if k < 1:
        raise ValueError("k must be positive")
    flags = list(relevant_flags)[:k]
    if len(flags) != k:
        raise ValueError(f"Expected {k} relevance labels, received {len(flags)}")
    return sum(flags) / k
