# Medical RAG learning path

The repository now has one implementation and one human-facing entry point:

- Run and study `External/Asthma_RAG_LangChain.ipynb`.
- Change reusable behavior only in `src/medical_rag/`.
- Run `python -m unittest discover -s tests -v` after changes.
- Rebuild ingestion with `python -m medical_rag.pipeline` when `src` is on `PYTHONPATH` or the package is installed.
- Inspect generated manifests and evaluation logs under `artifacts/`.

The project-scoped Codex skill at `.agents/skills/rag-hackathon-architecture/` contains the architecture, teaching, experiment, failure-analysis, medical-safety, and judge-preparation workflow.

Baseline choices - WHO + NICE, 700-token section-aware chunks, 100-token within-section overlap, Ollama `nomic-embed-text`, Chroma cosine search, and top-5 retrieval - are experiment starting points, not proven optima.
