# Canonical Repository Contract

## Source of truth

- Human-facing entry point: `External/Asthma_RAG_LangChain.ipynb`
- Reusable implementation: `src/medical_rag/`
- Approved source PDFs: `Docs/Sources/WHO asthma.pdf` and `Docs/Sources/NICE Asthma.pdf`
- Generated outputs: `artifacts/` (rebuildable, not implementation source)
- Tests: `tests/`
- Recovery copies: `archive/legacy/` (never import, execute, or extend)

Do not add another parser, chunker, vector-store implementation, or competing notebook. Extend the canonical modules and keep the notebook thin.

## Ownership

| Responsibility | Owner |
|---|---|
| Paths, sources, experiment values | `src/medical_rag/config.py` |
| Data contracts | `src/medical_rag/models.py` |
| PDF extraction and cleaning audit | `src/medical_rag/ingestion.py` |
| Heading detection and section-aware token chunking | `src/medical_rag/chunking.py` |
| Corpus orchestration and manifest validation | `src/medical_rag/pipeline.py` |
| Ollama embeddings and Chroma persistence/search | `src/medical_rag/vector_repository.py` |
| BM25, hybrid, RRF, reranking retrieval strategies | `src/medical_rag/hybrid_retrieval.py` |
| Grounded generation, citations, refusal, and safety | `src/medical_rag/generation.py` |
| Retrieval logging and audit template | `src/medical_rag/evaluation.py` |

## Baseline

- Scope: asthma diagnosis and guideline-based management in children and adolescents
- Chunking: deterministic section-aware token chunks
- Target/max tokens: 700
- Overlap: 100 tokens within the same detected section only
- Embeddings: `nomic-embed-text` through local Ollama
- Vector database: Chroma with cosine distance
- Starting retrieval depth: top-5
- Generation: Ollama `llama3.2`, grounded in retrieved chunks only (Day 3)

These are baseline experiment values, not claims of optimality.

## Data flow

```text
PDF + SourceSpec
  -> IngestionService.parse()
  -> ParsedGuideline[Page]
  -> SectionAwareChunker.chunk()
  -> Chunk[] + complete provenance
  -> CorpusPipeline.write_manifest()
  -> ChromaVectorRepository.upsert()
  -> stored text + vector + scalar metadata + stable ID

Question
  -> OllamaEmbeddings query vector
  -> Chroma cosine search (or UnifiedRetriever with hybrid/rerank)
  -> top-k SearchResult[]
  -> evidence panel + JSON log + human audit CSV

Question + top-k SearchResult[]
  -> assess_confidence(results)        # retrieval-score-based
  -> check_refusal(query, results)     # safety gate BEFORE LLM
  -> build_evidence_block(results)     # format for grounding prompt
  -> call_llm(system_prompt, user)     # Ollama llama3.2
  -> parse_llm_response(raw)           # extract JSON
  -> verify_citations(citations, results)  # chunk_id validation
  -> post_generation_safety_check()    # final sweep
  -> GeneratedAnswer (structured, cited, grounded)
```

## Commands

```powershell
python -m unittest discover -s tests -v
python -m medical_rag.pipeline
```

Run the notebook after ensuring Ollama is running and `nomic-embed-text` is available.
