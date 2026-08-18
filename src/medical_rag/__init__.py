"""Canonical medical-guideline RAG pipeline."""

from .chunking import SectionAwareChunker
from .config import RagConfig, default_config
from .generation import Citation, GeneratedAnswer, GenerationService
from .ingestion import IngestionService
from .models import Chunk, ParsedGuideline, SourceSpec

__all__ = [
    "Citation",
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
