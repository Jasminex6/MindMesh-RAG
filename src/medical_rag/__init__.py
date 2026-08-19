"""Canonical medical-guideline RAG pipeline."""

from .chunking import SectionAwareChunker
from .config import RagConfig, default_config
from .contracts import CitationContract, EvidenceContract, RAGResponse
from .app_service import RagApplicationService
from .generation import Citation, GeneratedAnswer, GenerationService
from .ingestion import IngestionService
from .models import Chunk, ParsedGuideline, SourceSpec
from .query_rewriter import ConversationalQueryRewriter, rewrite_conversational_query
from .query_decomposition import is_compound_query, decompose_query, retrieve_multi_question
from .intent_classifier import IntentClassifier, IntentDecision, summarize_for_retrieval

__all__ = [
    "Citation",
    "CitationContract",
    "ConversationalQueryRewriter",
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
    "rewrite_conversational_query",
    "is_compound_query",
    "decompose_query",
    "retrieve_multi_question",
    "IntentClassifier",
    "IntentDecision",
    "summarize_for_retrieval",
]
