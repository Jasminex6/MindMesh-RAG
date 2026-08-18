"""
Section-Aware Chunker Module for AI Clinical Decision Support Lite
Splits cleaned PDF text into 800-token chunks with 100-token overlap,
preserving section boundaries, subsections, original page numbers, and generating rich metadata schemas.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional

try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        # Fallback approximate token count (1 token ~= 4 chars or 0.75 words)
        return int(len(text.split()) * 1.3)


class SectionAwareChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100, min_tokens: int = 200):
        self.chunk_size = chunk_size        # Target chunk size (800 tokens)
        self.chunk_overlap = chunk_overlap  # Overlap size (100 tokens)
        self.min_tokens = min_tokens        # Minimum tokens before creating a new chunk on section boundary

    def _is_section_header(self, line: str) -> bool:
        """Heuristic to detect section headings in medical guidelines"""
        line_clean = line.strip()
        if not line_clean:
            return False

        # Numbered headings (e.g., '1. Introduction', '2.1 Diagnosis', 'Section 3')
        if re.match(r"^(\d+(\.\d+)*|Section\s+\d+|Part\s+\d+)\s+[:\-A-Z]", line_clean, re.IGNORECASE):
            return True

        # All-caps short line (e.g., 'DIAGNOSIS AND MANAGEMENT')
        if line_clean.isupper() and 4 <= len(line_clean) <= 60 and not line_clean.endswith("."):
            return True

        # Common clinical guideline keywords
        keywords = ["recommendation", "diagnosis", "management", "pharmacological", "monitoring", "algorithm", "evidence"]
        if any(kw in line_clean.lower() for kw in keywords) and len(line_clean) < 70 and not line_clean.endswith("."):
            return True

        return False

    def chunk_document(self, parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document pages into 800-token section-aware chunks with 100-token overlap & metadata
        """
        doc_name = parsed_doc["document_name"]
        file_path = parsed_doc["file_path"]
        chunks = []
        chunk_idx = 1

        current_section = "General / Overview"
        accumulated_units: List[Dict[str, Any]] = [] # list of {"text": line, "tokens": count, "page": page_num}
        current_tokens = 0
        start_page = 1

        for page in parsed_doc["pages"]:
            page_num = page["page_number"]
            page_text = page["cleaned_text"]
            lines = page_text.split("\n")

            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue

                line_tokens = count_tokens(line_str)
                is_header = self._is_section_header(line_str)

                # If new section header is encountered and we have accumulated enough content
                if is_header and current_tokens >= self.min_tokens:
                    chunk_text = "\n".join([u["text"] for u in accumulated_units]).strip()
                    end_page = accumulated_units[-1]["page"] if accumulated_units else page_num
                    chunks.append({
                        "chunk_id": f"{doc_name}_chunk_{chunk_idx:04d}",
                        "document_name": doc_name,
                        "file_path": file_path,
                        "section_title": current_section,
                        "start_page": accumulated_units[0]["page"] if accumulated_units else page_num,
                        "end_page": end_page,
                        "page_number": accumulated_units[0]["page"] if accumulated_units else page_num,
                        "token_count": current_tokens,
                        "text": chunk_text
                    })
                    chunk_idx += 1

                    # Retain 100 tokens of overlap
                    overlap_units = []
                    overlap_tokens_count = 0
                    for u in reversed(accumulated_units):
                        if overlap_tokens_count + u["tokens"] <= self.chunk_overlap:
                            overlap_units.insert(0, u)
                            overlap_tokens_count += u["tokens"]
                        else:
                            break

                    accumulated_units = overlap_units
                    current_tokens = overlap_tokens_count
                    current_section = line_str

                # Add current line unit
                accumulated_units.append({
                    "text": line_str,
                    "tokens": line_tokens,
                    "page": page_num
                })
                current_tokens += line_tokens

                # If total accumulated tokens reaches or exceeds chunk_size (800 tokens)
                if current_tokens >= self.chunk_size:
                    chunk_text = "\n".join([u["text"] for u in accumulated_units]).strip()
                    chunks.append({
                        "chunk_id": f"{doc_name}_chunk_{chunk_idx:04d}",
                        "document_name": doc_name,
                        "file_path": file_path,
                        "section_title": current_section,
                        "start_page": accumulated_units[0]["page"],
                        "end_page": accumulated_units[-1]["page"],
                        "page_number": accumulated_units[0]["page"],
                        "token_count": current_tokens,
                        "text": chunk_text
                    })
                    chunk_idx += 1

                    # Retain exact 100-token overlap for next chunk
                    overlap_units = []
                    overlap_tokens_count = 0
                    for u in reversed(accumulated_units):
                        if overlap_tokens_count + u["tokens"] <= self.chunk_overlap:
                            overlap_units.insert(0, u)
                            overlap_tokens_count += u["tokens"]
                        else:
                            break

                    accumulated_units = overlap_units
                    current_tokens = overlap_tokens_count

        # Flush final remaining units
        if accumulated_units:
            chunk_text = "\n".join([u["text"] for u in accumulated_units]).strip()
            if len(chunk_text) > 20:
                chunks.append({
                    "chunk_id": f"{doc_name}_chunk_{chunk_idx:04d}",
                    "document_name": doc_name,
                    "file_path": file_path,
                    "section_title": current_section,
                    "start_page": accumulated_units[0]["page"],
                    "end_page": accumulated_units[-1]["page"],
                    "page_number": accumulated_units[0]["page"],
                    "token_count": current_tokens,
                    "text": chunk_text
                })

        return chunks


def process_all_chunks(parsed_dir: str, output_dir: str) -> List[Dict[str, Any]]:
    """Process all parsed cleaned JSON files into 800-token chunked datasets with 100-token overlap"""
    os.makedirs(output_dir, exist_ok=True)
    chunker = SectionAwareChunker(chunk_size=800, chunk_overlap=100)
    
    json_files = [f for f in os.listdir(parsed_dir) if f.endswith("_cleaned.json")]
    all_chunks = []

    for jf in json_files:
        json_path = os.path.join(parsed_dir, jf)
        with open(json_path, "r", encoding="utf-8") as f:
            parsed_doc = json.load(f)

        doc_chunks = chunker.chunk_document(parsed_doc)
        out_chunk_path = os.path.join(output_dir, f"{parsed_doc['document_name']}_chunks.json")

        with open(out_chunk_path, "w", encoding="utf-8") as f:
            json.dump(doc_chunks, f, indent=2, ensure_ascii=False)

        print(f"Document: {parsed_doc['document_name']} -> Created {len(doc_chunks)} chunks (800 tokens, 100 overlap).")
        print(f"Saved to: {out_chunk_path}")
        all_chunks.extend(doc_chunks)

    return all_chunks


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parsed_directory = os.path.join(base_dir, "RAG", "parsed_data")
    chunks_directory = os.path.join(base_dir, "RAG", "chunks_data")
    
    if os.path.exists(parsed_directory):
        process_all_chunks(parsed_directory, chunks_directory)
    else:
        print(f"Parsed data directory '{parsed_directory}' not found yet. Run pdf_parser.py first.")
