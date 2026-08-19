# Day 4 — Local AI Systems, Deployment & Performance Master Plan

**Project:** Pediatric Asthma Clinical Decision Support (CDS) RAG System  
**Role:** Local AI Systems, Deployment & Performance Engineer  
**Status:** Master Architectural Planning Document  
**Constraint:** Planning Only — Order Independent Execution (Units A–F)

---

## 1. Repository Findings

A deep inspection of the repository (`c:\Users\Yasmine\Downloads\Orange x Instant`) reveals the following actual implementation state vs documentation:

### What Is Implemented & Working
1. **Core Data Models (`src/medical_rag/models.py`)**: Defines `SourceSpec`, `Page`, `ParsedGuideline`, `Chunk`, and `SearchResult`. Provenance tracking with physical PDF pages is fully functional.
2. **Ingestion & Cleaning (`src/medical_rag/ingestion.py`)**: PyMuPDF (`fitz`) parsing with custom regex rules for noise removal while preserving critical clinical dosages (`mg`, `mcg`, `kg`).
3. **Section-Aware Chunking (`src/medical_rag/chunking.py`)**: Token-based chunking (700 token target, 100 token overlap) constrained strictly by guideline section heading detection. Uses `tiktoken`.
4. **Pipeline Coordination (`src/medical_rag/pipeline.py`)**: `CorpusPipeline` manages document reading, chunking, deduplication, hash fingerprinting (`corpus_fingerprint`), and manifest generation (`artifacts/chunks_manifest.json`).
5. **Vector Database (`src/medical_rag/vector_repository.py`)**: `ChromaVectorRepository` wraps `langchain-chroma` using `nomic-embed-text` via `langchain-ollama` with Cosine Distance indexing (`hnsw:space: cosine`).
6. **Hybrid Retrieval & Reranking (`src/medical_rag/hybrid_retrieval.py`)**:
   - `BM25Retriever`: Custom Okapi BM25 implementation with acronym expansion (`saba`, `ics`, `feno`, etc.).
   - `reciprocal_rank_fusion`: Combines dense (Chroma) and sparse (BM25) search ranks.
   - `CrossEncoderReranker`: Term alignment & phrase matching reranker.
   - `UnifiedRetriever`: High-level entry point supporting `dense`, `bm25`, `hybrid`, `rerank`, and `hybrid_rerank`.
7. **Grounded Generation & Safety (`src/medical_rag/generation.py`)**:
   - `check_refusal`: Pre-LLM refusal gate intercepting patient-specific questions (`my child`, `dose for 25kg`) and out-of-scope queries before invoking the LLM.
   - `assess_confidence`: Empirical confidence scoring (`High`, `Medium`, `Low`, `Insufficient Evidence`) derived directly from vector similarity geometry.
   - `verify_citations`: Post-generation check cross-referencing returned `chunk_id` values against actual retrieved context IDs (`[VERIFIED]` vs `[UNVERIFIED]`).
   - `post_generation_safety_check`: Final safety guard against ungrounded dosage advice.
8. **Interactive CLI (`demo.py`)**: Functional CLI with multi-question splitting support.
9. **Test Suite (`tests/`)**: 47 active unit/integration tests running via `.venv\Scripts\python.exe -m unittest discover -s tests -v` (100% passing).

### Stale Documentation / Identified Gaps
- `PROJECT_TECHNOLOGY_STACK.md` mentions `sentence-transformers` cross-encoder model, but actual code in `hybrid_retrieval.py` (`CrossEncoderReranker`) defaults to term-matching overlap and phrase-alignment heuristic fallback for offline speed.
- `requirements.txt` / `pyproject.toml` missing explicit entries for `streamlit` and `httpx`/`requests` which are needed for Day 4 UI and HTTP diagnostics.
- There is currently no central application service layer (`rag.ask(...)`); `demo.py` orchestrates retrieval, generation, and formatting inline.

---

## 2. Target Architecture

```text
[ Browser / User ]
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Streamlit UI App Shell (Unit B)                                         │
│  - Session State (chat history, active conversation ID)                │
│  - Caching (@st.cache_resource for pipelines, @st.cache_data for UI)   │
│  - RTL Arabic toggle & 4-card clinical response renderer               │
└───────┬────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Application Service Boundary & Contracts (Unit D)                       │
│  - RagApplicationService.ask(query, conversation_id) -> RAGResponse    │
│  - Standardized RAGResponse & Evidence dataclasses                     │
└───────┬──────────────────────┬─────────────────────────┬───────────────┘
        │                      │                         │
        ▼                      ▼                         ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│ Conversation │     │ Stage Profiler   │     │ Ollama Runtime           │
│ Store        │     │ (Unit E)         │     │ Diagnostics Client       │
│ (Unit C)     │     │ - Nanosecond     │     │ (Unit A)                 │
│ - SQLite     │     │   stage timers   │     │ - HTTP health & tags     │
│   messages & │     │ - Timing metrics │     │ - Direct streaming       │
│   convs      │     │   export         │     │ - Cold/warm detection    │
└──────────────┘     └──────────────────┘     └──────────────────────────┘
                               │
                               ▼
            ┌────────────────────────────────────────┐
            │ Core RAG Pipeline (Preserved Day 1–3)  │
            │  - Ingestion & Section-Aware Chunker   │
            │  - ChromaDB + BM25 Sparse Search       │
            │  - Reciprocal Rank Fusion (RRF)        │
            │  - Cross-Encoder Reranker              │
            │  - Pre-LLM Refusal & Confidence Gate   │
            │  - Grounded Generation (Llama 3.2)     │
            │  - Citation Verification               │
            └────────────────────────────────────────┘
```

---

## 3. Decision Record

| Decision Area | Chosen Approach | Alternatives Considered | Why Chosen | Why Rejected | Key Trade-offs | Future Upgrade Path |
|---|---|---|---|---|---|---|
| **Ollama Boundary** | Dual-Layer: Keep `langchain-ollama` in generation pipeline; add direct HTTP diagnostic client (`OllamaRuntimeClient`) using stdlib `urllib.request`. | 1. Replace LangChain entirely with direct HTTP.<br>2. Pure LangChain without diagnostic client. | Exposes low-level HTTP `/api/tags` and `/api/generate` for direct learning without breaking existing working `GenerationService`. | Pure HTTP rewrite risks breaking Day 3 citation/formatting logic. Pure LangChain hides process internals. | Double interface maintenance for diagnostic calls. | Unify under a single async Ollama SDK if production demands it. |
| **App Service Boundary** | Dedicated `RagApplicationService` in `src/medical_rag/app_service.py`. | 1. FastAPI REST server.<br>2. Logic directly in Streamlit `app.py`. | Standard Python service class works seamlessly in CLI, Streamlit, unit tests, and future API wrappers. | FastAPI adds unnecessary network stack overhead for local hackathon demo. Inline Streamlit code creates tight coupling. | Additional object mapping layer. | Wrap `RagApplicationService` with FastAPI/gRPC if cloud deployment is needed. |
| **Shared Response Contract** | Canonical `RAGResponse` & `EvidenceItem` dataclasses in `src/medical_rag/contracts.py`. | 1. Heavy Pydantic v2 schemas.<br>2. Modifying `GeneratedAnswer` directly. | Clean, zero-dependency Python dataclasses. Decouples UI and API consumers from internal RAG output format. | Modifying `GeneratedAnswer` risks breaking 47 passing Day 1–3 tests. Pydantic v2 introduces version mismatch risks. | Small amount of conversion logic from `GeneratedAnswer`. | Add Pydantic validation if external REST schema serialization is required. |
| **SQLite Approach** | Standard library `sqlite3` with context managers in `src/medical_rag/persistence/sqlite_store.py`. | 1. SQLAlchemy / SQLModel.<br>2. DuckDB / Peewee. | Zero external dependencies; teaches raw SQL, schema DDL, transactions, connection lifecycle, and foreign key constraints directly. | ORMs hide database mechanics behind abstraction layers, violating learning goals. | Manual SQL string writing and mapping tuples to dicts. | Add Alembic for schema migrations if schema evolves long-term. |
| **Streamlit Caching** | `@st.cache_resource` for heavy singletons (Retriever, VectorDB, SQLite store); `@st.cache_data` for static UI/PDF renders; `st.session_state` for chat messages. | 1. Caching query responses globally.<br>2. Storing Retriever in `st.session_state`. | Prevents reloading vector DB and embedding models on every UI rerun. Session state maintains user context. | Global query caching produces stale answers for dynamic chat. Storing non-picklable objects in `session_state` causes serialization bugs. | Must carefully clear cache if underlying index changes. | Add TTL-based caching for frequent clinical queries. |
| **Streaming Architecture** | Generator-based token yield (`Iterable[str]`) from LLM callback/stream to Streamlit `st.write_stream`. | 1. Tight Streamlit callback inside generator.<br>2. Full response buffering. | Decouples token emission from UI framework; works in CLI typewriter output and Streamlit streaming. | Direct Streamlit callbacks tightly couple RAG service to UI thread. Buffering causes high perceived latency. | Handling structured JSON parsing on streamed tokens requires buffer accumulation. | Implement Server-Sent Events (SSE) for web clients. |
| **Profiling Strategy** | `StageProfiler` context manager (`src/medical_rag/profiling.py`) using `time.perf_counter_ns()`. | 1. `cProfile` / `py-spy`. | Provides nanosecond-precision microsecond-level timing per pipeline stage without performance overhead. | `cProfile` introduces heavy function-call overhead and noisy output for LLM IO calls. | Measures wall-clock time rather than CPU instruction counters. | Export timing spans to OpenTelemetry / Jaeger. |
| **Configuration** | Extend `RagConfig` in `src/medical_rag/config.py`. | 1. Scattered `.env` files.<br>2. Hardcoded parameters in `app.py`. | Extends established, validated configuration pattern already used throughout the codebase. | Hardcoding parameters causes demo failures when ports or paths differ. | Requires passing config down through service constructors. | Add YAML/TOML config file loading support. |
| **Error Handling** | Graceful fallback & degradation with user-actionable error messages. | 1. Crashing process on error.<br>2. Silent try/except swallowing. | Ensures app remains interactive during live demo even if Ollama or DB fails, displaying clear remediation steps. | Crashing looks unpolished in live demo. Swallowing errors makes debugging impossible. | Requires error branch testing for every external system call. | Add automatic server recovery scripts. |

---

## 4. Shared Contracts

Canonical Location: `src/medical_rag/contracts.py`

### Data Structures
```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class CitationContract:
    claim: str
    chunk_id: str
    document: str
    section: str
    page: str
    score: float
    verified: bool

@dataclass(frozen=True)
class EvidenceContract:
    chunk_id: str
    text: str
    document_name: str
    section_title: str
    page_number: str
    retrieval_score: float
    verification_status: bool

@dataclass(frozen=True)
class RAGResponse:
    status: str                       # ANSWER | REFUSAL | NEEDS_CLARIFICATION | ERROR
    language: str                     # "en" | "ar"
    query: str
    resolved_query: str
    recommendation: str
    evidence: tuple[EvidenceContract, ...]
    citations: tuple[CitationContract, ...]
    confidence: str                   # High | Medium | Low | Insufficient Evidence
    safety_message: str
    clarification_question: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
```

---

## 5. File Ownership Map

| File Path | Action | Unit Owner | Purpose | Other Units Allowed to Touch? |
|---|---|---|---|---|
| `src/medical_rag/contracts.py` | CREATE | Unit D (or Bootstrap) | Shared response contracts and data interfaces | YES (Units B, C, E consume or bootstrap minimal version) |
| `src/medical_rag/ollama_runtime.py` | CREATE | Unit A | Ollama HTTP runtime health, tags, warm-up, and direct stream client | NO |
| `tests/test_ollama_runtime.py` | CREATE | Unit A | Smoke and unit tests for Ollama runtime boundary | NO |
| `app.py` | CREATE | Unit B | Streamlit application shell & interactive UI | NO |
| `tests/test_ui_lifecycle.py` | CREATE | Unit B | Headless verification of Streamlit state and caching logic | NO |
| `src/medical_rag/persistence/sqlite_store.py` | CREATE | Unit C | SQLite conversation, message, and preference storage | NO |
| `tests/test_sqlite_persistence.py` | CREATE | Unit C | Unit tests for SQLite DDL, CRUD, transactions, and foreign keys | NO |
| `src/medical_rag/app_service.py` | CREATE | Unit D | High-level `RagApplicationService` facade coordinating RAG pipeline | NO |
| `tests/test_app_service.py` | CREATE | Unit D | Unit tests for application service boundary | NO |
| `src/medical_rag/profiling.py` | CREATE | Unit E | Stage timer, latency breakdown recorder, and dashboard exporter | NO |
| `tests/test_profiling.py` | CREATE | Unit E | Unit tests for timing metrics and stage profiling | NO |
| `src/medical_rag/learning_lab/embeddings_lab.py` | CREATE | Unit F | Interactive vector dimension & cosine similarity educational lab | NO |
| `tests/test_embeddings_lab.py` | CREATE | Unit F | Unit tests for embeddings learning lab calculations | NO |
| `src/medical_rag/config.py` | MODIFY | Unit D / A | Add persistence DB path and Ollama base URL options | YES (Non-breaking additive additions only) |
| `src/medical_rag/__init__.py` | MODIFY | Unit D | Export canonical service & contract types | YES (Non-breaking export additions only) |

---

## 6. Detailed Implementation Unit Plans (A–F)

---

### Unit A — Ollama Runtime & Local Model Boundary

- **Goal:** Build an explicit, low-level HTTP client (`OllamaRuntimeClient`) to inspect and manage the local Ollama server process (`http://localhost:11434`), verify model availability (`nomic-embed-text`, `llama3.2`), measure cold vs warm inference overhead, and test direct streaming without hiding mechanisms behind LangChain.
- **Why High-Yield:** De-mystifies Ollama by demonstrating it is a standard HTTP server process accepting JSON REST payloads over local TCP sockets.

#### Files
- `[NEW]` `src/medical_rag/ollama_runtime.py`
- `[NEW]` `tests/test_ollama_runtime.py`

#### Implementation Plan
1. Implement `OllamaRuntimeClient` in `src/medical_rag/ollama_runtime.py` using standard library `urllib.request` and `json`.
2. Methods to implement:
   - `is_server_reachable() -> bool`: GET `http://localhost:11434/`
   - `list_local_models() -> list[str]`: GET `http://localhost:11434/api/tags`
   - `check_model_availability(model_name: str) -> bool`
   - `warmup_model(model_name: str) -> dict[str, Any]`: Sends minimal request to force model loading into GPU/RAM.
   - `generate_direct_stream(prompt: str, model: str) -> Iterable[str]`: Reads chunked HTTP response stream from POST `/api/generate`.
3. Create `tests/test_ollama_runtime.py` testing connection fallback handling when server is offline or model is missing.

#### Independence & Dependencies
- **Hard Dependencies:** None.
- **Can Execute First?** YES.
- **Fallback when other units do not exist:** Self-contained. Uses standard library `urllib.request`.

#### Manual Exploration & Verification
- Manually run `curl http://localhost:11434/api/tags` in terminal.
- Run `python -m tests.test_ollama_runtime` to observe cold vs warm response latency.
- Run `ollama list` in PowerShell and compare with output of `OllamaRuntimeClient.list_local_models()`.

---

### Unit B — Streamlit Application Lifecycle

- **Goal:** Construct a reliable, high-performance Streamlit UI shell (`app.py`) presenting the 4-card clinical decision response layout, evidence drawer, RTL Arabic rendering support, and proper resource management using `@st.cache_resource` and `st.session_state`.
- **Why High-Yield:** Direct hands-on learning of Streamlit's script rerun execution model, state persistence across interactions, and memory management.

#### Files
- `[NEW]` `app.py`
- `[NEW]` `tests/test_ui_lifecycle.py`

#### Implementation Plan
1. Create `app.py` root Streamlit script.
2. Define cached resource initializers:
   ```python
   @st.cache_resource
   def get_rag_service():
       # Initialises RAG pipeline or uses fallback bridge if Unit D not executed yet
       ...
   ```
3. Initialize `st.session_state` for:
   - `active_conversation_id`
   - `messages` (list of rendered Q&A pairs)
   - `arabic_mode` (boolean)
4. Implement UI layout:
   - Sidebar: Server status indicator, model selector, clear chat button, RTL toggle.
   - Main Chat Area: `st.chat_input`, message container loop.
   - Response Cards: 1. Recommendation, 2. Supporting Evidence, 3. Citations, 4. Confidence & Safety.
5. Create `tests/test_ui_lifecycle.py` to test state initialization and contract adapter rendering headlessly.

#### Independence & Dependencies
- **Hard Dependencies:** None.
- **Can Execute First?** YES.
- **Fallback if Unit D is absent:** Uses a simple bridge calling `setup_pipeline()` and `ask_question()` from `demo.py` or `UnifiedRetriever` directly.

#### Manual Exploration & Verification
- Run `streamlit run app.py`.
- Add a temporary `st.sidebar.write(f"Rerun count: {st.session_state.get('reruns', 0)}")` to observe how button clicks trigger top-to-bottom script re-execution.
- Toggle Arabic RTL mode and check CSS text alignment.

---

### Unit C — SQLite Conversation Persistence

- **Goal:** Implement clean relational conversation and message persistence in `src/medical_rag/persistence/sqlite_store.py` using Python's standard `sqlite3` library.
- **Why High-Yield:** Teaches explicit SQL schema design (`CREATE TABLE`), foreign keys, transaction handling (`commit`/`rollback`), index optimization, and connection management without magic ORMs.

#### Files
- `[NEW]` `src/medical_rag/persistence/sqlite_store.py`
- `[NEW]` `tests/test_sqlite_persistence.py`

#### Implementation Plan
1. Implement `SQLiteStore` in `src/medical_rag/persistence/sqlite_store.py`.
2. DDL Schema:
   ```sql
   CREATE TABLE IF NOT EXISTS conversations (
       id TEXT PRIMARY KEY,
       title TEXT NOT NULL,
       language TEXT DEFAULT 'en',
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   CREATE TABLE IF NOT EXISTS messages (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       conversation_id TEXT NOT NULL,
       role TEXT NOT NULL,
       content TEXT NOT NULL,
       structured_response TEXT, -- Stored as JSON string
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
   );

   CREATE TABLE IF NOT EXISTS preferences (
       key TEXT PRIMARY KEY,
       value TEXT NOT NULL
   );
   ```
3. Methods:
   - `create_conversation(title: str, language: str = 'en') -> str`
   - `add_message(conversation_id: str, role: str, content: str, response_obj: dict | None = None)`
   - `get_conversation_messages(conversation_id: str) -> list[dict]`
   - `list_conversations() -> list[dict]`
4. Implement context manager for safe connection opening/closing and transaction commits.
5. Create `tests/test_sqlite_persistence.py` testing database creation, foreign key cascade, JSON serialization, and queries.

#### Independence & Dependencies
- **Hard Dependencies:** None.
- **Can Execute First?** YES.
- **Fallback when other units do not exist:** Completely independent standard library module.

#### Manual Exploration & Verification
- Open SQLite DB via CLI: `sqlite3 artifacts/conversations.db ".schema"`.
- Run manual query: `SELECT id, title, created_at FROM conversations;`.
- Insert test message and verify persistence across program restarts.

---

### Unit D — RAG Application Adapter / Service Boundary

- **Goal:** Create `RagApplicationService` in `src/medical_rag/app_service.py` to serve as the unified API boundary encapsulating retrieval, generation, safety gating, profiling, and optional persistence.
- **Why High-Yield:** Teaches clean architecture principles, application facade patterns, and dependency injection.

#### Files
- `[NEW]` `src/medical_rag/contracts.py` (if not created during bootstrap)
- `[NEW]` `src/medical_rag/app_service.py`
- `[NEW]` `tests/test_app_service.py`
- `[MODIFY]` `src/medical_rag/__init__.py`

#### Implementation Plan
1. Define `RagApplicationService` accepting:
   - `retriever: UnifiedRetriever`
   - `generation_service: GenerationService`
   - `persistence_store: SQLiteStore | None = None`
   - `profiler: StageProfiler | None = None`
2. Implement `ask(query: str, conversation_id: str | None = None, strategy: str = "hybrid_rerank") -> RAGResponse`.
3. Orchestrate:
   - Preprocessing & multi-question check.
   - Retrieval via `retriever.search`.
   - Generation & safety via `generation_service.generate`.
   - Conversion of `GeneratedAnswer` to canonical `RAGResponse`.
   - Async/optional persistence saving if `persistence_store` is attached.
4. Ensure `demo.py` can be refactored to use `RagApplicationService` without changing CLI behavior.

#### Independence & Dependencies
- **Hard Dependencies:** None (reuses existing Day 1-3 pipeline components).
- **Can Execute First?** YES.

#### Manual Exploration & Verification
- Run `python -m tests.test_app_service`.
- Verify `demo.py` still functions cleanly when backed by `RagApplicationService`.

---

### Unit E — Streaming, Profiling & Performance

- **Goal:** Instrument the RAG pipeline with nanosecond-precision stage timers (`StageProfiler`) to measure query preprocessing, embedding generation, dense retrieval, sparse BM25, RRF, reranking, and LLM generation latency.
- **Why High-Yield:** Emphasizes evidence-based performance analysis ("Measure → identify bottleneck → understand cause → change one thing → measure again").

#### Files
- `[NEW]` `src/medical_rag/profiling.py`
- `[NEW]` `tests/test_profiling.py`

#### Implementation Plan
1. Implement `StageProfiler` in `src/medical_rag/profiling.py`:
   - `profile_stage(stage_name: str)` context manager using `time.perf_counter()`.
   - Records latency per stage in milliseconds.
   - `get_summary() -> dict[str, float]`
   - Exports JSON timing payload compatible with Day 4 Evaluation Dashboard.
2. Measure cold-start latency (first embedding model call + Chroma DB disk load) vs warm latency (subsequent queries).
3. Test batching optimization for vector repository search.

#### Independence & Dependencies
- **Hard Dependencies:** None.
- **Can Execute First?** YES.

#### Manual Exploration & Verification
- Execute profiling benchmark test to display timing breakdown table.
- Compare cold request total latency (~3-5s) vs warm request total latency (~1-2s).
- Verify LLM generation represents >80% of total latency before attempting vector search micro-optimizations.

---

### Unit F — Embeddings Learning Lab

- **Goal:** Create a standalone, interactive educational lab module (`src/medical_rag/learning_lab/embeddings_lab.py`) to inspect physical text embedding vectors (768 dimensions), calculate cosine similarity manually using NumPy/Python math, and compare semantic distances between medical vs non-medical queries.
- **Why High-Yield:** Provides physical intuition into how text is mapped into vector space and why semantic retrieval finds relevant clinical passages.

#### Files
- `[NEW]` `src/medical_rag/learning_lab/embeddings_lab.py`
- `[NEW]` `tests/test_embeddings_lab.py`

#### Implementation Plan
1. Implement vector comparison helper in `src/medical_rag/learning_lab/embeddings_lab.py`:
   - `get_embedding(text: str) -> list[float]` using `OllamaEmbeddings`.
   - `cosine_similarity(vec1: list[float], vec2: list[float]) -> float` manual calculation:
     $$\text{similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
2. Comparative test sets:
   - Query 1: `"childhood asthma symptoms"`
   - Query 2: `"signs of asthma in children"` (high similarity expected)
   - Query 3: `"banana bread recipe"` (low similarity expected)
   - Query 4 (if available): English vs Arabic medical terms.
3. Print vector dimension size, norm, min/max values, and similarity matrix.

#### Independence & Dependencies
- **Hard Dependencies:** None.
- **Can Execute First?** YES.

#### Manual Exploration & Verification
- Run `python -m src.medical_rag.learning_lab.embeddings_lab` and inspect vector dimension outputs and dot-product geometry.

---

## 7. Execution Independence & Bootstrap Rules

### Independence Matrix

| Start Executing With | Supported? | Bootstrap Actions Taken | Actions Prohibited |
|---|---|---|---|
| **Unit A First** | YES | Creates `ollama_runtime.py`. | Does NOT create UI, database, or modify existing RAG code. |
| **Unit B First** | YES | Creates minimal `contracts.py` if missing. Builds UI shell using `demo.py` / `UnifiedRetriever` bridge. | Does NOT duplicate RAG logic inside Streamlit script. |
| **Unit C First** | YES | Creates `sqlite_store.py` and standalone schema. | Does NOT force Streamlit or Ollama dependencies. |
| **Unit D First** | YES | Creates `contracts.py` and `app_service.py` wrapping existing RAG. | Does NOT implement Streamlit UI or SQLite store. |
| **Unit E First** | YES | Creates `profiling.py` wrapper measuring execution blocks. | Does NOT modify core retrieval algorithms. |
| **Unit F First** | YES | Creates standalone `embeddings_lab.py`. | Does NOT touch production Chroma DB collection. |

---

## 8. Recommended Learning Sequence

While execution order is strictly independent, the recommended sequence for maximum learning yield is:

```text
Unit A (Ollama Runtime)
       ↓
Unit D (App Service & Contracts)
       ↓
Unit B (Streamlit UI Shell)
       ↓
Unit C (SQLite Persistence)
       ↓
Unit E (Profiling & Latency)
       ↓
Unit F (Embeddings Learning Lab)
```

### Rationale
1. **Unit A** first demystifies the local LLM server process.
2. **Unit D** creates clean contracts and API boundaries.
3. **Unit B** builds the visual Streamlit shell around Unit D.
4. **Unit C** adds stateful conversation persistence to Unit B & D.
5. **Unit E** measures the end-to-end timing across all integrated layers.
6. **Unit F** dives deep into vector math to round out understanding.

---

## 9. Demo Preflight Sequence

Before presenting the live hackathon demo, execute the following preflight checks:

1. **Verify Ollama Process:**
   ```powershell
   curl http://localhost:11434/
   # Expect: "Ollama is running"
   ```
2. **Verify Required Models:**
   ```powershell
   ollama list
   # Expect: nomic-embed-text, llama3.2
   ```
3. **Run Automated Unit Test Suite:**
   ```powershell
   .venv\Scripts\python.exe -m unittest discover -s tests -v
   # Expect: 47+ tests OK
   ```
4. **Run CLI Fallback Verification:**
   ```powershell
   .venv\Scripts\python.exe demo.py "What are the common symptoms of asthma"
   # Expect: 4-card clinical decision response with verified citations
   ```
5. **Launch Streamlit App:**
   ```powershell
   .venv\Scripts\python.exe -m streamlit run app.py
   # Expect: App loads at http://localhost:8501
   ```
