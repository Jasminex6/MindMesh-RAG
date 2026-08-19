"""Deterministic section-aware and sentence-aware hybrid token chunking with page provenance."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import tiktoken

from .ingestion import TOC_LEADER_RE
from .models import Chunk, ParsedGuideline


@dataclass(frozen=True)
class _Unit:
    text: str
    page: int
    section: str
    tokens: int


def is_section_heading(line: str) -> bool:
    text = re.sub(r"\s+", " ", line.strip())
    if not text or len(text) > 180 or TOC_LEADER_RE.fullmatch(text):
        return False
    if text.startswith(("•", "-", "–", "—")):
        return False
    if re.match(r"^(?:table\s+\d+|box\s+\d+|step\s+[1-5])\b", text, re.IGNORECASE):
        return True
    if re.match(
        r"^(?:\d+(?:\.\d+)?\.?|section\s+\d+|chapter\s+\d+|part\s+[ivx\d]+)\s+\S",
        text,
        re.IGNORECASE,
    ):
        return True
    if re.match(r"^(?:recommendation|recommendations|appendix|annex)\b", text, re.I):
        return True
    if text.isupper() and 4 <= len(text) <= 100 and not text.endswith((".", ";")):
        return True
    return bool(
        text[0].isupper()
        and len(text.split()) <= 16
        and not text.endswith((".", ";"))
        and re.match(
            r"^(?:summary of recommendations|initial (?:clinical|management|treatment)|"
            r"diagnosing\b|monitoring\b|pharmacological\b|self-management\b|"
            r"medicines for\b|management of\b|treatment of\b|recommendations for research\b|"
            r"how the recommendations might affect practice\b|rationale and impact\b|"
            r"managing asthma in\b|pregnancy\b|elderly\b|obesity\b|gerd\b|aerd\b|occupational\b)",
            text,
            re.IGNORECASE,
        )
    )


def _derive_age_and_population(doc_name: str, section: str, text: str) -> tuple[str, bool, str]:
    is_gina = "gina" in doc_name.lower()
    covers_under_6 = not is_gina

    combined = (section + " " + text).lower()

    # Specific populations
    pops = {
        "pregnancy": ["pregnancy", "pregnant"],
        "elderly": ["elderly", "older adults"],
        "obesity": ["obesity", "obese"],
        "gerd": ["gerd", "gastroesophageal"],
        "aerd": ["aerd", "aspirin-exacerbated", "aspirin exacerbated"],
        "occupational": ["occupational asthma", "work-related"],
    }
    for pop_key, keywords in pops.items():
        if any(kw in combined for kw in keywords):
            return "specific_population", covers_under_6, pop_key

    # Table 3 / Adults & Adolescents
    if "table 3" in combined or "adults and adolescents" in combined or "adults & adolescents" in combined or "12+" in combined:
        return "adults_adolescents", covers_under_6, ""

    # Table 4 / Children 6-11
    if "table 4" in combined or "children 6-11" in combined or "children 6–11" in combined or "6 to 11" in combined or "6-11 years" in combined:
        return "children_6_11", covers_under_6, ""

    # General children 6-11 vs adults
    if "children" in combined and "6" in combined:
        return "children_6_11", covers_under_6, ""
    if "adult" in combined or "adolescent" in combined:
        return "adults_adolescents", covers_under_6, ""

    return "all_ages", covers_under_6, ""


def split_sentences(text: str) -> list[str]:
    """Split text into natural sentences using punctuation boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


class SectionAwareChunker:
    """Sentence-Aware Hybrid Chunker respecting section boundaries and sentence integrity."""

    def __init__(self, chunk_size: int = 700, overlap: int = 100):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _split_unit(self, text: str, page: int, section: str) -> list[_Unit]:
        tokens = self.encoder.encode(text)
        if len(tokens) <= self.chunk_size:
            return [_Unit(text, page, section, len(tokens))]
        
        sentences = split_sentences(text)
        units = []
        current_sentence_buffer: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            st_tokens = len(self.encoder.encode(sentence))
            if st_tokens > self.chunk_size:
                sub_tokens = self.encoder.encode(sentence)
                for start in range(0, len(sub_tokens), self.chunk_size):
                    part = sub_tokens[start : start + self.chunk_size]
                    units.append(_Unit(self.encoder.decode(part), page, section, len(part)))
                continue

            if current_tokens + st_tokens <= self.chunk_size:
                current_sentence_buffer.append(sentence)
                current_tokens += st_tokens
            else:
                if current_sentence_buffer:
                    chunk_str = " ".join(current_sentence_buffer)
                    units.append(_Unit(chunk_str, page, section, self.count_tokens(chunk_str)))
                current_sentence_buffer = [sentence]
                current_tokens = st_tokens

        if current_sentence_buffer:
            chunk_str = " ".join(current_sentence_buffer)
            units.append(_Unit(chunk_str, page, section, self.count_tokens(chunk_str)))

        return units if units else [_Unit(text, page, section, len(tokens))]

    def _overlap_tail(self, units: list[_Unit], limit: int) -> list[_Unit]:
        remaining = max(0, min(self.overlap, limit))
        tail: list[_Unit] = []
        for unit in reversed(units):
            if remaining <= 0:
                break
            if unit.tokens <= remaining:
                tail.insert(0, unit)
                remaining -= unit.tokens
            else:
                tokens = self.encoder.encode(unit.text)[-remaining:]
                tail.insert(0, _Unit(
                    self.encoder.decode(tokens), unit.page, unit.section, len(tokens)
                ))
                remaining = 0
        return tail

    def _make_chunk(
        self,
        document: ParsedGuideline,
        units: list[_Unit],
        sequence: int,
        boundary_reason: str,
        topic: str,
    ) -> Chunk:
        text = "\n".join(unit.text for unit in units).strip()
        pages = [unit.page for unit in units]
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        slug = re.sub(r"[^a-z0-9]+", "-", document.source.document.lower()).strip("-")[:48]
        section_name = units[0].section

        age_group, covers_under_6, pop_name = _derive_age_and_population(
            document.source.document, section_name, text
        )

        return Chunk(
            chunk_id=f"{slug}-{sequence:04d}-{digest}",
            text=text,
            document=document.source.document,
            publisher=document.source.publisher,
            source_url=document.source.source_url,
            topic=topic,
            section=section_name,
            page_start=min(pages),
            page_end=max(pages),
            token_count=self.count_tokens(text),
            boundary_reason=boundary_reason,
            age_group=age_group,
            covers_under_6=covers_under_6,
            specific_population_name=pop_name,
        )

    def chunk(self, document: ParsedGuideline, topic: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        section = "Front matter / section not yet detected"
        buffer: list[_Unit] = []
        pending_headings: list[tuple[str, int]] = []
        sequence = 1

        def flush(reason: str, keep_overlap: bool, next_tokens: int = 0) -> None:
            nonlocal buffer, sequence
            if not buffer:
                return
            chunks.append(self._make_chunk(document, buffer, sequence, reason, topic))
            sequence += 1
            limit = self.chunk_size - next_tokens
            buffer = self._overlap_tail(buffer, limit) if keep_overlap else []

        for page in document.pages:
            for raw_text in page.cleaned_text.splitlines():
                text = raw_text.strip()
                if not text:
                    continue
                if is_section_heading(text):
                    flush("section_boundary", keep_overlap=False)
                    pending_headings.append((text, page.number))
                    section = " > ".join(heading for heading, _ in pending_headings[-3:])
                    continue

                if pending_headings:
                    heading_text = "\n".join(heading for heading, _ in pending_headings)
                    heading_page = pending_headings[0][1]
                    heading_tokens = self.count_tokens(heading_text)
                    buffer.append(_Unit(heading_text, heading_page, section, heading_tokens))
                    pending_headings = []

                for unit in self._split_unit(text, page.number, section):
                    if buffer and buffer[0].section != unit.section:
                        flush("section_boundary", keep_overlap=False)
                    buffered_tokens = sum(item.tokens for item in buffer)
                    if buffer and buffered_tokens + unit.tokens > self.chunk_size:
                        flush("token_limit", keep_overlap=True, next_tokens=unit.tokens)
                    buffer.append(unit)

        flush("document_end", keep_overlap=False)
        return chunks
