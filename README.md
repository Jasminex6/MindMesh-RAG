# Asthma medical RAG hackathon

This repository has one active implementation:

```text
External/Asthma_RAG_LangChain.ipynb   learning and demo entry point
src/medical_rag/                      reusable pipeline
tests/                                fast ingestion/chunking/evaluation tests
Docs/Sources/                         approved source PDFs
artifacts/                            rebuildable manifests and retrieval logs
.agents/skills/rag-hackathon-architecture/
                                      project workflow and teaching skill
archive/legacy/                       recovery copies; never import or extend
```

## Baseline

- Scope: asthma diagnosis and guideline-based management in children and adolescents
- Sources: WHO childhood-asthma guideline and NICE NG245
- Chunking: 700-token section-aware maximum with 100-token within-section overlap
- Embeddings: Ollama `nomic-embed-text`
- Vector database: Chroma with cosine distance
- Retrieval: top-5 evidence panel; generation intentionally deferred

These settings are a measured starting configuration, not asserted optima.

## Run

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Test core logic:

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src').Path
python -m unittest discover -s tests -v
```

Build the page-preserving chunk manifest:

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src').Path
python -m medical_rag.pipeline
```

For embeddings and retrieval, start Ollama, run `ollama pull nomic-embed-text`, then execute `External/Asthma_RAG_LangChain.ipynb` from the repository.
