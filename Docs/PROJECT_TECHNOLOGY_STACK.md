# Project Technology Stack & Architecture Overview

An easy-to-understand, comprehensive breakdown of all technologies, tools, libraries, and architectural decisions used in the **Pediatric Asthma Clinical Decision Support (CDS) RAG System** from **Day 1 to Day 3**.

---

## 🧭 Executive Summary

This project builds a **zero-hallucination, evidence-grounded Clinical Decision Support (CDS) system** for pediatric and adolescent asthma management based on approved guidelines from the **World Health Organization (WHO)** and the **National Institute for Health and Care Excellence (NICE NG245)**.

The system transitions raw clinical PDFs into a fully interactive, cited decision support tool:

```text
PDF Guidelines ➔ Ingestion & Cleaning ➔ Section-Aware Chunking ➔ Hybrid Search (Dense + BM25) 
➔ Cross-Encoder Reranking ➔ Pre-LLM Refusal Gate ➔ Grounded Generation (Llama 3.2) 
➔ Citation Verification ➔ 4-Card Decision Output
```

---

## 🛠️ Complete Technology Stack

| Category | Technology / Library | Purpose in Project |
|---|---|---|
| **Runtime & Package Manager** | **Python 3.12** | Primary programming language runtime. |
| | **`uv`** | High-performance Rust-based Python environment and dependency manager. |
| **PDF Ingestion & Parsing** | **`pypdf`** | Extracting raw text and physical page numbers from official WHO and NICE PDFs. |
| | **Custom Regex Cleaning Engine** | Stripping header/footer navigation noise while preserving critical dosages (`mg`, `mcg`, `kg`). |
| **Chunking & Indexing** | **Custom Section-Aware Chunker** | Splitting text at guideline section boundaries (max 700 tokens, 100-token overlap). |
| **Local AI Models (Ollama)** | **Ollama (`0.32.14`)** | Local framework for running open-weights LLMs and embeddings privately on-device. |
| | **`nomic-embed-text`** | 768-dimensional local text embedding model fine-tuned for semantic retrieval. |
| | **`llama3.2` (3B)** | Local lightweight instruction-tuned LLM used for grounded answer generation. |
| **Vector Database** | **ChromaDB (`chromadb`)** | Open-source vector database using **Cosine Distance** indexing for dense semantic search. |
| **Search & Reranking** | **`rank-bm25`** | Sparse keyword search engine implementing Okapi BM25 for exact drug names & clinical terms. |
| | **Reciprocal Rank Fusion (RRF)** | Fusing sparse (BM25) and dense (Chroma) search ranks using $RRF(d) = \sum \frac{1}{k + r(d)}$. |
| | **Cross-Encoder Reranker** | Deep transformer model (`sentence-transformers`) for cross-attention candidate reranking. |
| **Frameworks & Interfaces** | **`langchain-ollama`** | Standardized LangChain integration for Ollama LLMs and Embeddings. |
| | **Custom CLI (`demo.py`)** | Interactive clinical CLI tool for testing queries and displaying structured outputs. |
| **Testing & Quality** | **`unittest`** | Built-in Python test runner executing **48 automated unit & integration tests**. |
| | **Pyright** | Static type checking and workspace path resolution. |

---

## 📐 Pipeline Evolution (Day 1 to Day 3)

### Day 1: Document Processing & Chunking
- **Problem**: Clinical guidelines are long, dense PDFs with complex section hierarchies, tables, and physical page boundaries. Standard naive fixed-character chunkers break medical context across section boundaries.
- **Solution**:
  - Built `src/medical_rag/ingestion.py` to extract text while retaining physical PDF page numbers.
  - Built `src/medical_rag/chunking.py` to enforce **Section-Aware Chunking**. Text is chunked up to 700 tokens, but never crosses a major guideline section heading. Within a section, a 100-token sliding window overlap preserves sentence continuity.

### Day 2: Hybrid Retrieval & Reranking
- **Problem**: Dense vector search alone misses exact medical keywords (e.g. `SABA`, `budesonide`, `25kg`), while keyword search alone misses semantic intent.
- **Solution**:
  - Implemented **Hybrid Search** (`src/medical_rag/hybrid_retrieval.py`):
    1. **Dense Retrieval**: `nomic-embed-text` embeddings stored in ChromaDB.
    2. **Sparse Retrieval**: `rank-bm25` over tokenized chunks.
    3. **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense search results into a single rank.
    4. **Cross-Encoder Reranking**: Re-scores top candidates using cross-attention.
  - Built an evaluation harness (`RAG/evaluation/evaluate_retrieval.py`) measuring **Precision@K** against ground-truth clinical query benchmarks.

### Day 3: Grounded Generation, Refusal Gate, & Citations
- **Problem**: Commercial LLMs hallucinate medical dosages, invent guidelines, or provide unsafe patient-specific treatment advice.
- **Solution**:
  - Created `src/medical_rag/generation.py`:
    1. **Pre-LLM Refusal Gate (`check_refusal`)**: Detects patient-specific requests (`my child`, `weighing 25kg`, `dose of`) and refuses immediately before calling the LLM.
    2. **Score-Based Confidence**: Confidence (`High`, `Medium`, `Low`, `Insufficient Evidence`) is calculated strictly from vector distance geometry scores, avoiding unreliable LLM self-assessments.
    3. **Grounding Prompt**: Enforces that retrieved chunks are the **only source of truth**.
    4. **Citation Verification (`verify_citations`)**: Validates every generated claim's `chunk_id` against retrieved vector results to flag hallucinations (`[VERIFIED]` vs `[UNVERIFIED]`).
    5. **4-Card Structured Answer Layout (`demo.py`)**:
       - **1. Recommendation**: Short, direct, non-patient-specific.
       - **2. Supporting Evidence**: Bullet-point short excerpts from cited chunks.
       - **3. Citations**: Document name, Section title, Page numbers, Chunk ID, and Retrieval Score.
       - **4. Confidence & Safety**: Confidence label + Clinical disclaimer.

---

## 🔑 Key Architectural Decisions

1. **Why Local Ollama Models?**  
   Medical data privacy and offline hackathon capability. Running `nomic-embed-text` and `llama3.2` locally guarantees zero data leakage and zero API costs.

2. **Why Derive Confidence from Retrieval Scores?**  
   LLMs are notoriously poorly calibrated — they express high confidence even when hallucinating. Tying confidence to vector cosine relevance scores grounds confidence in empirical retrieval quality.

3. **Why Pre-LLM Refusal?**  
   Refusing patient-specific requests (e.g. prescribing specific dosages for a specific child) *before* calling the LLM saves latency, compute, and eliminates the risk of unsafe LLM output.

---

## 🛠️ How to Test & Verify

### Run Full Test Suite (48 Tests)
```powershell
python -m unittest discover -s tests -v
```

### Run Interactive Decision Support Demo
```powershell
python demo.py
```

### Run Single Direct Query
```powershell
python demo.py "What are the common symptoms of asthma"
```
