"""
Data structures and schema definitions for Document Ingestion & Chunking
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class PageContent:
    page_number: int
    raw_text: str
    cleaned_text: str
    headers_footers_removed: List[str] = field(default_factory=list)

@dataclass
class ParsedDocument:
    document_id: str
    document_name: str
    file_path: str
    total_pages: int
    pages: List[PageContent] = field(default_factory=list)

@dataclass
class ChunkMetadata:
    chunk_id: str
    document_name: str
    file_path: str
    page_number: int
    start_page: int
    end_page: int
    section_title: str
    token_count: int
    source_url: Optional[str] = None

@dataclass
class TextChunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata
