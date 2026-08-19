"""Canonical medical-guideline RAG pipeline."""

from .chunking import SectionAwareChunker
from .config import RagConfig, default_config
from .contracts import CitationContract, EvidenceContract, RAGResponse
from .app_service import RagApplicationService
from .generation import Citation, GeneratedAnswer, GenerationService
from .ingestion import IngestionService
from .models import Chunk, ParsedGuideline, SourceSpec

__all__ = [
    "Citation",
    "CitationContract",
    "EvidenceContract",
    "RAGResponse",
    "RagApplicationService",
    "Chunk",
    "GeneratedAnswer",
    "GenerationService",
    "IngestionService",
    "ParsedGuideline",
    "RagConfig",
    "SectionAwareChunker",
    "SourceSpec",
    "default_config",
]
