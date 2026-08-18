"""Page-preserving PDF extraction with conservative, auditable cleaning."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pymupdf

from .models import Page, ParsedGuideline, RemovalRecord, SourceSpec


ZONE_LINES = 3
MAX_RUNNING_LINE_LENGTH = 140
PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE
)
TOC_LEADER_RE = re.compile(
    r"^.{2,160}?(?:\.{3,}|\s\.{2,}|\s*[\u2026\u22ef]{1,})\s*[ivxlcdm\d-]{1,8}$", re.IGNORECASE
)
BIBLIOGRAPHY_CITATION_RE = re.compile(
    r"^(?:\d{1,3}\.\s+[A-Z][a-zA-Z\-]+(?:\s+[A-Z]{1,3})?|\bdoi\.org/|https?://(?:dx\.)?doi\.org/|"
    r"\bPaediatr Respir Rev\b|\bIndian Pediatr\b|\bEur J Emerg Med\b|\bPediatr Pulmonol\b|"
    r"\bN Engl J Med\b|\bLancet\b|\bBMJ\b|\bJ Allergy Clin Immunol\b).*",
    re.IGNORECASE,
)
PROTECTED_CLINICAL_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|kg|mL|L|IU|units?|mmol|mEq|mg/dL|"
    r"mg/kg|mmHg|bpm|%)\b|\brecommendation\s+\d+[a-z]?\b|\bgrade\s+[A-D]\b|"
    r"\blevel\s+(?:I{1,3}|IV|V)\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{4}\b)",
    re.IGNORECASE,
)
NAVIGATION_LINES = {
    "back to top",
    "click here",
    "home",
    "next",
    "previous",
    "return to recommendations",
}


def _nonempty_indexes(text: str) -> list[int]:
    return [index for index, line in enumerate(text.splitlines()) if line.strip()]


def _zone_indexes(text: str) -> set[int]:
    indexes = _nonempty_indexes(text)
    return set(indexes[:ZONE_LINES] + indexes[-ZONE_LINES:])


def _normalize_running_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line.strip().lower())
    return re.sub(r"\b\d+\b", "#", line)


class IngestionService:
    """Own PDF extraction, layout-noise detection, cleaning, and removal audit."""

    def __init__(self, recurring_ratio: float = 0.35):
        if not 0 < recurring_ratio <= 1:
            raise ValueError("recurring_ratio must be in (0, 1]")
        self.recurring_ratio = recurring_ratio

    def extract(self, pdf_path: Path) -> list[tuple[int, str]]:
        pages: list[tuple[int, str]] = []
        with pymupdf.open(pdf_path) as pdf:
            if pdf.needs_pass:
                raise ValueError(f"Encrypted PDF requires a password: {pdf_path.name}")
            for index, page in enumerate(pdf):
                pages.append((index + 1, page.get_text("text", sort=True) or ""))
        if not pages:
            raise ValueError(f"No pages were extracted from {pdf_path.name}")
        return pages

    def find_recurring_lines(self, raw_pages: list[tuple[int, str]]) -> set[str]:
        if len(raw_pages) < 4:
            return set()
        counts: Counter[str] = Counter()
        for _, text in raw_pages:
            lines = text.splitlines()
            seen = set()
            for index in _zone_indexes(text):
                line = lines[index].strip()
                if (
                    not line
                    or len(line) > MAX_RUNNING_LINE_LENGTH
                    or PROTECTED_CLINICAL_RE.search(line)
                ):
                    continue
                normalized = _normalize_running_line(line)
                if normalized not in seen:
                    counts[normalized] += 1
                    seen.add(normalized)
        threshold = max(3, round(len(raw_pages) * self.recurring_ratio))
        return {line for line, count in counts.items() if count >= threshold}

    def clean_page(
        self, raw_text: str, page_number: int, recurring_lines: set[str]
    ) -> tuple[str, list[RemovalRecord]]:
        raw_text = raw_text.replace("\u00ad", "").replace("\x00", "")
        lines = raw_text.splitlines()
        zone_indexes = _zone_indexes(raw_text)
        kept: list[str] = []
        audit: list[RemovalRecord] = []

        for index, original in enumerate(lines):
            line = re.sub(r"[ \t]+", " ", original).strip()
            if not line:
                kept.append("")
                continue
            reason = None
            if index in zone_indexes and PAGE_NUMBER_RE.fullmatch(line):
                reason = "page_number"
            elif (
                index in zone_indexes
                and _normalize_running_line(line) in recurring_lines
                and not PROTECTED_CLINICAL_RE.search(line)
            ):
                reason = "recurring_header_or_footer"
            elif TOC_LEADER_RE.fullmatch(line):
                reason = "table_of_contents_leader"
            elif BIBLIOGRAPHY_CITATION_RE.fullmatch(line) and ("doi.org" in line or "https://" in line or re.match(r"^\d{1,3}\.\s+[A-Z]", line)):
                reason = "bibliography_citation_reference"
            elif line.casefold() in NAVIGATION_LINES:
                reason = "navigation"

            if reason:
                audit.append(RemovalRecord(page_number, reason, line))
            else:
                kept.append(line)

        text = "\n".join(kept)
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip(), audit

    def parse(self, pdf_path: Path, source: SourceSpec) -> ParsedGuideline:
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        raw_pages = self.extract(pdf_path)
        recurring_lines = self.find_recurring_lines(raw_pages)
        pages = []
        audit = []
        for page_number, raw_text in raw_pages:
            cleaned_text, removed = self.clean_page(raw_text, page_number, recurring_lines)
            pages.append(Page(page_number, raw_text, cleaned_text))
            audit.extend(removed)
        return ParsedGuideline(
            source=source,
            file_path=str(pdf_path.resolve()),
            pages=tuple(pages),
            recurring_lines=tuple(sorted(recurring_lines)),
            removal_audit=tuple(audit),
        )
