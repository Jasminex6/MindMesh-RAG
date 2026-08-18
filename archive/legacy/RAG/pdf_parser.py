"""
PDF Parser & Text Cleaner Module for AI Clinical Decision Support
Parses PDF documents from Docs/Sources preserving page numbers, removing recurring 
headers/footers and extraction artifacts, and outputting clean structured JSON data.
Handles AES encrypted / restricted PDFs gracefully.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pypdf
except ImportError:
    pypdf = None


class PDFParser:
    def __init__(self, remove_headers_footers: bool = True):
        self.remove_headers_footers = remove_headers_footers

    def _extract_pages_fitz(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract pages using PyMuPDF (fitz)"""
        doc = fitz.open(pdf_path)
        pages = []
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            text = page.get_text("text")
            pages.append({
                "page_number": page_num,
                "raw_text": text
            })
        doc.close()
        return pages

    def _extract_pages_pypdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract pages using PyPDF with AES encryption handling"""
        reader = pypdf.PdfReader(pdf_path)
        
        # Handle encrypted or password-restricted PDFs
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:
                print(f"Notice: PDF is encrypted ({e}). Attempting extraction...")

        pages = []
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            try:
                text = page.extract_text() or ""
            except Exception as e:
                text = ""
                print(f"Warning on page {page_num}: {e}")

            pages.append({
                "page_number": page_num,
                "raw_text": text
            })
        return pages

    def detect_recurring_headers_footers(self, raw_pages: List[Dict[str, Any]]) -> Tuple[set, set]:
        """
        Identify top and bottom lines that recur across multiple pages (e.g., >30% of pages)
        """
        top_lines_count: Dict[str, int] = {}
        bottom_lines_count: Dict[str, int] = {}
        total_pages = len(raw_pages)

        if total_pages <= 2:
            return set(), set()

        for page in raw_pages:
            lines = [line.strip() for line in page["raw_text"].split("\n") if line.strip()]
            if not lines:
                continue

            # First 2 non-empty lines as potential header candidate
            top_candidates = lines[:2]
            for line in top_candidates:
                if len(line) > 3 and not re.match(r"^\d+$", line):
                    top_lines_count[line] = top_lines_count.get(line, 0) + 1

            # Last 2 non-empty lines as potential footer candidate
            bottom_candidates = lines[-2:]
            for line in bottom_candidates:
                if len(line) > 3 and not re.match(r"^\d+$", line):
                    bottom_lines_count[line] = bottom_lines_count.get(line, 0) + 1

        threshold = max(3, int(total_pages * 0.35))
        repeating_headers = {line for line, count in top_lines_count.items() if count >= threshold}
        repeating_footers = {line for line, count in bottom_lines_count.items() if count >= threshold}

        return repeating_headers, repeating_footers

    def clean_page_text(self, text: str, page_num: int, total_pages: int, headers: set, footers: set) -> str:
        """
        Clean raw text from a PDF page:
        - Strip known headers and footers
        - Remove page numbers (e.g., 'Page 4 of 50', '4 / 50', standalone '42')
        - Join hyphenated word splits across line breaks
        - Normalize whitespace and control characters
        """
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            if stripped in headers or stripped in footers:
                continue

            if re.match(r"^(Page\s+\d+(\s+of\s+\d+)?|\d+\s*/\s*\d+|\d+)$", stripped, re.IGNORECASE):
                continue

            cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines)
        cleaned_text = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", cleaned_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

        return cleaned_text.strip()

    def parse_document(self, pdf_path: str) -> Dict[str, Any]:
        """Full document parsing and cleaning pipeline with fallbacks"""
        filename = os.path.basename(pdf_path)
        doc_name = os.path.splitext(filename)[0]

        raw_pages = []
        errors = []

        # Try pypdf first
        if pypdf is not None:
            try:
                raw_pages = self._extract_pages_pypdf(pdf_path)
            except Exception as e:
                errors.append(f"pypdf extraction failed: {e}")

        # Fallback to PyMuPDF fitz if pypdf fails or not available
        if not raw_pages and fitz is not None:
            try:
                raw_pages = self._extract_pages_fitz(pdf_path)
            except Exception as e:
                errors.append(f"PyMuPDF fitz extraction failed: {e}")

        if not raw_pages:
            raise RuntimeError(f"Could not extract pages from '{filename}'. Errors: {'; '.join(errors)}")

        total_pages = len(raw_pages)
        headers, footers = self.detect_recurring_headers_footers(raw_pages)

        parsed_pages = []
        total_raw_chars = 0
        total_cleaned_chars = 0

        for page in raw_pages:
            p_num = page["page_number"]
            raw_text = page["raw_text"]
            cleaned_text = self.clean_page_text(raw_text, p_num, total_pages, headers, footers)

            total_raw_chars += len(raw_text)
            total_cleaned_chars += len(cleaned_text)

            parsed_pages.append({
                "page_number": p_num,
                "raw_char_count": len(raw_text),
                "cleaned_char_count": len(cleaned_text),
                "raw_text": raw_text,
                "cleaned_text": cleaned_text
            })

        return {
            "document_name": doc_name,
            "file_name": filename,
            "file_path": pdf_path,
            "total_pages": total_pages,
            "detected_headers": list(headers),
            "detected_footers": list(footers),
            "summary_stats": {
                "total_pages": total_pages,
                "total_raw_chars": total_raw_chars,
                "total_cleaned_chars": total_cleaned_chars,
                "reduction_ratio": round(1 - (total_cleaned_chars / max(1, total_raw_chars)), 4)
            },
            "pages": parsed_pages
        }


def process_all_sources(sources_dir: str, output_dir: str) -> List[Dict[str, Any]]:
    """Process all PDF files in sources_dir and save clean JSON outputs to output_dir"""
    os.makedirs(output_dir, exist_ok=True)
    parser = PDFParser()

    pdf_files = [f for f in os.listdir(sources_dir) if f.lower().endswith(".pdf")]
    results = []

    print(f"Found {len(pdf_files)} PDF file(s) in '{sources_dir}'")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(sources_dir, pdf_file)
        print(f"\n--- Parsing & Cleaning: {pdf_file} ---")

        try:
            doc_data = parser.parse_document(pdf_path)
            out_json_path = os.path.join(output_dir, f"{doc_data['document_name']}_cleaned.json")

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)

            print(f"Saved parsed result to: {out_json_path}")
            print(f"Pages: {doc_data['total_pages']}")
            print(f"Raw Chars: {doc_data['summary_stats']['total_raw_chars']:,}")
            print(f"Cleaned Chars: {doc_data['summary_stats']['total_cleaned_chars']:,}")
            results.append(doc_data)
        except Exception as e:
            print(f"ERROR processing '{pdf_file}': {e}. Skipping file and continuing...")

    return results


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sources_directory = os.path.join(base_dir, "Docs", "Sources")
    parsed_output_directory = os.path.join(base_dir, "RAG", "parsed_data")

    print(f"Sources Dir: {sources_directory}")
    print(f"Output Dir: {parsed_output_directory}")

    process_all_sources(sources_directory, parsed_output_directory)
