"""Canonical medical-guideline RAG pipeline."""

from .chunking import SectionAwareChunker
from .config import RagConfig, default_config
from .generation import Citation, GeneratedAnswer, GenerationService
from .ingestion import IngestionService
from .models import Chunk, ParsedGuideline, SourceSpec
from .query_rewriter import ConversationalQueryRewriter, rewrite_conversational_query

__all__ = [
    "Citation",
    "Chunk",
    "ConversationalQueryRewriter",
    "GeneratedAnswer",
    "GenerationService",
    "IngestionService",
    "ParsedGuideline",
    "RagConfig",
    "SectionAwareChunker",
    "SourceSpec",
    "default_config",
    "rewrite_conversational_query",
]

