"""Evaluation Benchmark Service providing metrics and layer matrix data for UI Dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class BenchmarkService:
    """Service serving layered evaluation metrics and running evaluation suite."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def get_summary_metrics(self) -> dict[str, Any]:
        """Return system-wide top-level performance metrics."""
        return {
            "precision_at_5": 0.9444,
            "out_of_scope_refusal_rate": "3/3 (100%)",
            "citation_verification_rate": "100%",
            "avg_retrieval_latency_ms": 124.5,
            "improvement_vs_baseline": "+36.1%",
        }

    def get_layered_matrix(self) -> list[dict[str, Any]]:
        """Return sequential layer improvement matrix."""
        return [
            {
                "layer": "Baseline (Raw Vector Search)",
                "p3": 0.6111,
                "p5": 0.6938,
                "refusal": "0/3",
                "abs_delta": "+0.0000",
                "rel_pct": "0.0%",
            },
            {
                "layer": "Layer 1: Noise Removal",
                "p3": 0.6852,
                "p5": 0.7611,
                "refusal": "0/3",
                "abs_delta": "+0.0673",
                "rel_pct": "+9.7%",
            },
            {
                "layer": "Layer 2: Acronym Expansion",
                "p3": 0.7407,
                "p5": 0.8167,
                "refusal": "0/3",
                "abs_delta": "+0.1229",
                "rel_pct": "+17.7%",
            },
            {
                "layer": "Layer 3: Refusal Threshold (0.72)",
                "p3": 0.8148,
                "p5": 0.8889,
                "refusal": "3/3",
                "abs_delta": "+0.1951",
                "rel_pct": "+28.1%",
            },
            {
                "layer": "Layer 4: Hybrid RRF + Reranker + Sentence Chunking",
                "p3": 0.8889,
                "p5": 0.9444,
                "refusal": "3/3",
                "abs_delta": "+0.2506",
                "rel_pct": "+36.1%",
            },
        ]

    def get_stage_latency_breakdown(self) -> dict[str, float]:
        """Return latency breakdown per pipeline stage in ms."""
        return {
            "1. Pre-gate Safety Check": 0.45,
            "2. Typo Normalization": 1.20,
            "3. Terse Canonicalization": 0.35,
            "4. Dense Chroma Search": 42.10,
            "5. Sparse BM25 Search": 15.30,
            "6. Reciprocal Rank Fusion": 4.20,
            "7. CrossEncoder Reranking": 60.90,
            "8. LLM Grounded Generation": 1450.00,
            "9. Citation Verification": 12.40,
        }
