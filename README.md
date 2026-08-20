---
title: Pediatric Asthma CDS
emoji: 🫁
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Clinical Guideline RAG System — Pediatric Asthma Decision Support

An evidence-grounded Clinical Decision Support (CDS) Retrieval-Augmented Generation (RAG) system built for pediatric and adolescent asthma management guidelines (WHO & NICE NG245).

---

## 🏛️ Project Architecture

```text
Orange x Instant/
├── src/medical_rag/               # Modular Production Pipeline Package
│   ├── ingestion.py               # Section-aware PDF parsing & cleaning
│   ├── chunking.py                # Deterministic token chunking with section boundaries
│   ├── vector_repository.py       # Chroma vector store with Ollama embeddings
│   ├── hybrid_retrieval.py        # BM25 + Dense Hybrid Search with Reciprocal Rank Fusion & Cross-Encoder Reranking
│   ├── generation.py              # Day 3 Grounded Generation, Refusal Gate, Citation Verification
│   ├── pipeline.py                # Corpus orchestration & fingerprint validation
│   └── models.py                  # Core data contracts (Chunk, SearchResult, GeneratedAnswer, Citation)
├── tests/                         # Comprehensive Unit & Integration Test Suite
│   ├── test_chunking.py           # Section boundary & token overlap tests
│   ├── test_ingestion.py          # PDF cleaning & dose preservation tests
│   ├── test_hybrid_retrieval.py   # RRF & Reranker strategy tests
│   ├── test_generation.py         # Confidence, refusal, parsing, & safety unit tests
│   └── test_generation_live.py    # 6-scenario live end-to-end integration test
├── demo.py                        # Interactive CLI Decision Support Tool
├── Docs/                          # Clinical Guideline Sources & Day 3 Architecture Documentation
│   ├── DAY3_GROUNDED_GENERATION.md# Complete Day 3 Technical & Mentor Review Guide
│   └── Sources/                   # Approved WHO & NICE PDF guidelines
└── RAG/evaluation/                # Ground Truth Benchmark Dataset & Precision Evaluator
```

---

## ⚡ Quick Start for Team Members

### 1. Environment Setup
Install dependencies using `uv` (recommended) or `pip`:

```powershell
# Using uv (fast)
uv pip install -r requirements.txt
uv pip install -e .

# Or using standard pip
pip install -r requirements.txt
pip install -e .
```

### 2. Local LLM & Embedding Setup
Ensure [Ollama](https://ollama.com) is installed and running:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

---

## 🚀 Usage

### Interactive CLI Tool (`demo.py`)
Run the interactive decision support CLI:

```powershell
python demo.py
```

Or pass a clinical question directly:

```powershell
python demo.py "What is the recommended first-line controller treatment for children with asthma?"
```

---

## 🛡️ Grounded Generation & Safety Features

1. **Retrieved Evidence as Single Source of Truth**: The LLM (`llama3.2`) receives ONLY the top retrieved guideline chunks and is strictly forbidden from adding external medical knowledge.
2. **Pre-LLM Refusal Gate**: Patient-specific dosing, age, weight, or diagnostic requests are caught before calling the LLM and refused immediately.
3. **Objective Confidence Scoring**: Confidence (`High`, `Medium`, `Low`, `Insufficient Evidence`) is derived from retrieval geometry scores rather than LLM self-confidence.
4. **Citation Verification**: Every generated claim is mapped to its `chunk_id`. Post-generation verification validates cited chunk IDs against retrieved results and enriches provenance metadata (Document, Section, Page numbers, Scores).

---

## 🧪 Running Tests

Run the complete 48-test automated suite:

```powershell
python -m unittest discover -s tests -v
```

Run the live 6-scenario integration test:

```powershell
python tests/test_generation_live.py
```
