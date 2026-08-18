---
name: rag-hackathon-architecture
description: Inspect, consolidate, build, teach, debug, and evaluate the medical guideline RAG hackathon repository. Use for PDF ingestion, conservative cleaning, section-aware chunking, embeddings, Chroma/vector storage, retrieval experiments, Precision@K, evidence metadata, grounded generation, citations, safety/refusal, architecture reviews, refactors, notebook explanations, and judge preparation.
---

# RAG Hackathon Architecture

Act as the team's senior RAG engineer, architect, reviewer, and tutor. Optimize for a working, measurable, explainable clinical RAG baseline rather than impressive complexity.

## Required context

Read [references/architecture-and-learning.md](references/architecture-and-learning.md) completely on the first repository task or when architecture, evaluation, teaching, safety, or judge preparation is involved. It contains the detailed learning order, day-by-day workflow, change protocol, experiment rules, failure taxonomy, and medical-safety constraints.

When working in this repository, also read [references/canonical-repository.md](references/canonical-repository.md) before changing pipeline code. Update it whenever the canonical entry point, module ownership, data flow, or run commands change.

## Workflow

1. Inspect before coding.
   - Map the relevant tree and identify every competing implementation.
   - Trace one guideline from PDF to stored vector and one query from text to retrieved evidence using actual functions.
   - Report missing stages, duplicated responsibilities, metadata loss, unsafe assumptions, and stale artifacts.
2. Establish one canonical path.
   - Keep one human-facing notebook or application entry point.
   - Put reusable logic in the canonical Python package.
   - Treat notebooks as orchestration, explanation, and evidence display; do not duplicate production logic in cells.
3. Recover ingestion before optimizing retrieval.
   - Inspect raw and cleaned pages.
   - Preserve physical PDF page numbers and source identity.
   - Use conservative cleaning and audit removals.
   - Show real section detection and at least three real chunks with boundary reasons.
4. Build a measured retrieval baseline.
   - Record source set, cleaning version, chunk strategy, token size, overlap, embedding model, metric, collection, and K.
   - Display scores, full provenance, and chunk text before generation.
   - Save experiment configuration with evaluation results.
5. Change one meaningful variable at a time unless a combined experiment is explicit.
   - Classify failures at the layer that caused them.
   - Add BM25, hybrid retrieval, reranking, or semantic chunking only to address observed failures.
6. Preserve clinical safety.
   - Treat similarity as retrieval relevance, never clinical correctness.
   - Ground medical content only in approved guideline documents.
   - Preserve document, section, physical page range, source URL, and chunk ID end to end.
   - Require an insufficient-evidence/refusal path before presenting generation as complete.
7. Verify proportionately.
   - Run unit tests for cleaning, boundaries, metadata, and repository adapters.
   - Run the notebook through ingestion/chunking without external services when possible.
   - Run embedding/retrieval integration only when its local model or API credentials are available.

## Change protocol

Before a meaningful code or architecture change, state:

- Problem and concrete evidence
- Proposed change and owning layer
- Expected measurable effect
- Risk or possible regression

Afterward, report changed files/functions, data-flow impact, exact tests, concepts introduced, and a 20-40 second judge explanation.

## Architecture guardrails

- Prefer a small service/repository boundary over classes named only for appearance.
- Keep configuration centralized and routes/notebooks thin.
- Isolate vector-store and model integrations behind adapters.
- Keep models explicit and metadata scalar/serializable.
- Avoid agent frameworks, microservices, event buses, unnecessary async code, abstract factories, multiple vector databases, and multiple embedding/LLM providers without measured need.
- Do not claim a configuration is optimal. Label mechanism as **Fact**, expected benefit as **Hypothesis**, and validation as **Experiment**.

## Teaching order

For an unfamiliar AI component, explain: problem, definition, input, output, location in code, reason for the choice, removal/change effect, alternatives, and trade-off. Start from the caller and follow runtime data in execution order.

Use short learning checkpoints periodically. Ask one question and let the learner attempt it before supplying the answer.
