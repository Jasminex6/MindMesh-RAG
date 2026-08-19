# Conversational Query Rewriter & Context-Aware RAG Integration Guide

This guide documents the design, architecture, implementation decisions, and integration interfaces for the **Conversational Query Rewriter** and **Context-Aware RAG Pipeline**. It is written to enable seamless integration into web apps, APIs, or downstream services without merge conflicts or architectural regressions.

---

## 1. Executive Summary & Objective

### Problem Statement
In multi-turn conversational RAG, users frequently ask implicit, pronoun-heavy, or contextual follow-up questions such as:
- *"What about the second one?"*
- *"Can you elaborate on that?"*
- *"What are its side effects?"*

Without preprocessing, passing these raw follow-up strings directly to vector retrieval results in:
1. **Low Retrieval Relevance Scores** (e.g. cosine relevance < 0.40), causing the refusal gate to trigger false "out of scope" or "insufficient evidence" refusals.
2. **Loss of Clinical Context** (e.g. losing patient age band, drug names, or diagnostic context established in turn 1).
3. **Grounding Guardrail False Triggers** in generation prompts that lack explicit instructions for follow-up explanations.

### Solution Overview
We implemented a **Conversational Query Rewriter** module and updated the RAG pipeline flow:

$$\text{User Input} + \text{Chat History} \xrightarrow{\text{Query Rewriter}} \text{Standalone Query} \xrightarrow{\text{Retriever}} \text{Evidence Chunks} \xrightarrow{\text{Grounded LLM Generation}} \text{Answer}$$

---

## 2. Codebase Additions & Modifications

### A. [NEW] `src/medical_rag/query_rewriter.py`
* **Purpose**: Encapsulates conversational query reformulation into a dedicated service.
* **Key Class**: `ConversationalQueryRewriter(model="llama3.2", temperature=0.0, max_turns=4)`
* **Key Method**: `rewrite(query: str, chat_history: list[dict[str, str]] | None, skip_llm: bool = False) -> str`
* **Functionality**:
  1. Accepts user input + recent chat history.
  2. Applies a **sliding window of the last 3-4 turns** (up to 8 messages) to maintain prompt size efficiency.
  3. Uses a tailored LLM system prompt (`_REWRITE_SYSTEM_PROMPT`) to rephrase ambiguous follow-ups into standalone clinical queries.
  4. Includes a deterministic **heuristic fallback** (`_heuristic_fallback`) if LLM invocation is skipped (`skip_llm=True`) or encounters an exception.
* **Convenience Wrapper**: `rewrite_conversational_query(query, chat_history, model, skip_llm)`

---

### B. [MODIFIED] `src/medical_rag/generation.py`
* **Changes**:
  1. **Grounding System Prompt Rule 8**:
     ```text
     8. CONVERSATIONAL FOLLOW-UPS & GROUNDING: Answer follow-up questions using the provided context and conversation history. Do not trigger a refusal if the user asks for explanations, examples, or clarifications of previous points, provided they map back to the retrieved chunks and history.
     ```
  2. **`build_user_prompt()`**:
     - Added parameter `chat_history: list[dict[str, str]] | None = None`.
     - When provided, prepends a formatted `RECENT CONVERSATION HISTORY` block to the user prompt so the LLM generation step is aware of prior turns.
  3. **`GenerationService.generate()`**:
     - Added parameter `chat_history: list[dict[str, str]] | None = None` and passed it down to `build_user_prompt()`.

---

### C. [MODIFIED] `src/medical_rag/__init__.py`
* **Exported Symbols**:
  - `ConversationalQueryRewriter`
  - `rewrite_conversational_query`

---

### D. [MODIFIED] `demo.py`
* **Changes**:
  1. **Stateful Chat History in `ask_question()`**:
     - Signature updated:
       ```python
       def ask_question(
           query: str,
           retriever: UnifiedRetriever,
           gen_service: GenerationService,
           slots: dict | None = None,
           chat_history: list[dict[str, str]] | list[tuple[str, str]] | None = None,
           skip_llm_rewriter: bool = False,
           interactive: bool = True,
       ) -> object:
       ```
     - Rewrites `query` $\rightarrow$ `processed_query` prior to classification and vector retrieval.
     - Appends user and assistant turns to `chat_history`.
  2. **Interactive Clarification Prompts**:
     - Uses `sys.stdin.isatty()` guard so interactive CLI prompts the user for choices (e.g. *Acute treatment* vs *Long-term maintenance* when typing `"treatment"`), while non-interactive automated test suites continue seamlessly without blocking.
  3. **CLI Loop (`main()`)**:
     - Instantiates `chat_history = []` across iterations.

---

### E. [NEW / MODIFIED] Test Suite
* **[NEW] `tests/test_query_rewriter.py`**: Unit tests covering sliding window history formatting, LLM query rephrasing, and fallback logic.
* **[MODIFIED] `tests/test_integration_flow.py`**: Added end-to-end multi-turn tests (`test_multiturn_followup_rewriting_and_grounding` and `test_multiturn_elaboration_fallback`).

---

## 3. Rationale & Key Decisions

1. **Preprocessing Before Vector Retrieval**:
   - Rewriting the query *before* retrieval guarantees that vector embeddings are generated from clinically complete sentences rather than ambiguous phrases like *"What about the second one?"*.

2. **Sliding Window of 3-4 Turns (6-8 Messages)**:
   - Preserves relevant context (drug names, age band, symptoms) while avoiding context window bloating or latency degradation.

3. **Fallback Resiliency**:
   - If Ollama is offline or `skip_llm=True`, the system falls back gracefully to a heuristic combiner (`_heuristic_fallback`) so the pipeline never crashes.

4. **Non-Breaking API Contracts**:
   - All new parameters (`chat_history`, `skip_llm_rewriter`, `interactive`) are **optional** with default values (`None` / `False` / `True`). Existing code calls like `ask_question(query, retriever, gen_service)` will continue to work without modification.

---

## 4. Teammate Integration Guide

If you are integrating this module into a Web UI, API framework (FastAPI / Flask / Django), or background worker, follow these guidelines:

### Data Structure for `chat_history`
Maintain chat history as a Python list of dictionaries:

```python
chat_history = [
    {"role": "user", "content": "What are the first-line controllers for asthma in children?"},
    {"role": "assistant", "content": "First-line controllers include low-dose ICS and LTRA."},
]
```

### Usage Pattern in an API / Service

```python
from medical_rag import (
    GenerationService,
    UnifiedRetriever,
    ConversationalQueryRewriter,
)
from demo import ask_question

# 1. Initialize services once during startup
retriever = ...  # your UnifiedRetriever instance
gen_service = GenerationService(model="llama3.2")

# 2. In your API endpoint request handler:
@app.post("/chat")
def chat_endpoint(user_message: str, session_history: list[dict[str, str]]):
    # Call ask_question with non-interactive mode for API workers
    result = ask_question(
        query=user_message,
        retriever=retriever,
        gen_service=gen_service,
        chat_history=session_history,
        interactive=False,  # Disables CLI stdin prompts in web/API context
    )
    
    # result is a GeneratedAnswer instance
    return {
        "recommendation": result.recommendation,
        "supporting_evidence": result.supporting_evidence,
        "citations": [asdict(c) for c in result.citations],
        "confidence": result.confidence,
        "refused": result.refused,
        "chat_history": session_history, # Updated automatically with new turn
    }
```

### Direct Module Import (Stand-alone Query Rewriting)

If you only need to rephrase a follow-up query before running custom logic:

```python
from medical_rag import ConversationalQueryRewriter

rewriter = ConversationalQueryRewriter(model="llama3.2")
standalone_query = rewriter.rewrite(
    query="What are its side effects?",
    chat_history=session_history,
)
```

---

## 5. Verification Commands

To verify that your integration causes no regressions:

```bash
# Run full test suite (155 tests)
uv run pytest

# Run specific query rewriter & integration tests
uv run pytest tests/test_query_rewriter.py tests/test_integration_flow.py
```
