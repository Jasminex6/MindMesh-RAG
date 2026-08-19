"""Evidence panel component for rendering top-k retrieved guideline passages before LLM generation."""

from __future__ import annotations

from typing import Iterable, Any


def render_evidence_panel(results: Iterable[Any], max_excerpt_len: int = 400) -> str:
    """Format top-k retrieved search results into a clean evidence panel string.
    
    Compatible with SearchResult instances or metadata dictionary objects.
    """
    res_list = list(results)
    if not res_list:
        return "=== EVIDENCE PANEL ===\nNo retrieved evidence passages found."

    lines = ["=== EVIDENCE PANEL (Top-K Retrieved Passages) ==="]
    for idx, item in enumerate(res_list, start=1):
        if hasattr(item, "metadata"):
            meta = item.metadata
            score = getattr(item, "score", 0.0)
            text = getattr(item, "text", "")
            rank = getattr(item, "rank", idx)
        elif isinstance(item, dict):
            meta = item.get("metadata", item)
            score = item.get("score", 0.0)
            text = item.get("text", item.get("page_content", ""))
            rank = item.get("rank", idx)
        else:
            meta = {}
            score = 0.0
            text = str(item)
            rank = idx

        doc = meta.get("document", meta.get("file", "Unknown Source"))
        sec = meta.get("section", "N/A")
        p_start = meta.get("page_start", meta.get("page", "N/A"))
        p_end = meta.get("page_end", p_start)
        page_str = f"{p_start}" if p_start == p_end else f"{p_start}-{p_end}"
        chunk_id = meta.get("chunk_id", "N/A")

        excerpt = text.strip()
        if len(excerpt) > max_excerpt_len:
            excerpt = excerpt[:max_excerpt_len] + "..."

        lines.append(f"\nChunk {rank}")
        lines.append(f"Chunk ID: {chunk_id}")
        lines.append(f"Score: {score:.4f}")
        lines.append(f"Source: {doc}")
        lines.append(f"Section: {sec}")
        lines.append(f"Page: {page_str}")
        lines.append(f"\n{excerpt}")
        lines.append("-" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test sample
    sample_res = [{
        "rank": 1,
        "score": 0.8421,
        "text": "Inhaled corticosteroids (ICS) are the preferred controller therapy for children with asthma.",
        "metadata": {
            "chunk_id": "who-guideline-p25-001",
            "document": "WHO asthma.pdf",
            "section": "Controller therapy",
            "page_start": 25,
            "page_end": 25,
        }
    }]
    print(render_evidence_panel(sample_res))
