"""Small, serializable data contracts shared by pipeline layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceSpec:
    file_name: str
    document: str
    publisher: str
    source_url: str
    usage_note: str


@dataclass(frozen=True)
class Page:
    number: int
    raw_text: str
    cleaned_text: str


@dataclass(frozen=True)
class RemovalRecord:
    page: int
    reason: str
    text: str


@dataclass(frozen=True)
class ParsedGuideline:
    source: SourceSpec
    file_path: str
    pages: tuple[Page, ...]
    recurring_lines: tuple[str, ...] = ()
    removal_audit: tuple[RemovalRecord, ...] = ()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    document: str
    publisher: str
    source_url: str
    topic: str
    section: str
    page_start: int
    page_end: int
    token_count: int
    boundary_reason: str

    def metadata(self) -> dict[str, str | int]:
        """Return scalar metadata accepted by Chroma and suitable for citations."""
        return {
            "chunk_id": self.chunk_id,
            "document": self.document,
            "publisher": self.publisher,
            "source_url": self.source_url,
            "topic": self.topic,
            "section": self.section,
            "page": self.page_start,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "token_count": self.token_count,
            "boundary_reason": self.boundary_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    text: str
    metadata: dict[str, str | int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
