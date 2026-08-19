# Implementation Plan — Medical RAG (Day 4: Safety + Evaluation + Deployment)

**Project:** AI Clinical Decision Support Lite (Asthma RAG)
**Status assumed:** Day 1–3 already built — parsing, chunking, embeddings, hybrid-ish retrieval, grounded generation with citations exist and work.
**Goal of this plan:** finish the required Day 4 work (Safety, Retrieval Optimization, UI/Deployment, Evaluation) without breaking Day 1–3, freeze the project, and be demo-ready for Day 5.
**Target tool:** written as a task list you can paste into Antigravity (or any agentic coding tool) phase by phase. Each phase has explicit files, acceptance criteria, and a "do not touch" boundary so agents don't clobber each other's work.

> ⚠️ No project folder was attached to this conversation — only `Day2.pptx`, `Day3.pptx`, and the Day4 PDF. File paths below come from the Day4 doc's own "Suggested files" sections. If your real repo uses different paths, tell Antigravity to adapt paths, not the logic.

---

## 0. Before touching anything — Freeze & Baseline (blocks everyone)

This must run first and produce numbers every other phase compares against. Assign to whoever runs Workstream D, but everyone should trigger it before their own changes.

**Tasks**
1. Tag/branch the current working state (`git tag day3-baseline` or equivalent).
2. Run the full existing test suite; record pass/fail.
3. Run the current retrieval baseline on the existing question set and record:
   - Precision@3, Precision@5
   - expected-evidence rank
   - retrieval latency (median, stage-by-stage if possible)
4. Confirm the current Day 3 answer structure still works end-to-end: `Recommendation → Supporting Evidence → Citations → Confidence & Safety`.
5. Save these numbers somewhere durable (`RAG/evaluation/baseline_results.json` or similar) — everything downstream needs a before/after.

**Acceptance criteria:** baseline numbers exist on disk; all Day 1–3 tests green; nobody starts optimization work blind.

---

## 1. Workstream A — Safety, Guardrails & Clarification (P0)

**Owns:** `src/medical_rag/safety.py`, `src/medical_rag/router.py`, `tests/test_safety.py`
**Does not own:** UI, retrieval internals, dashboard.

### 1.1 Unsupported-claim detection (from Day 3 "grounding principle")
- For every generated claim, verify a citation both **exists** and **actually supports the claim** (not just present) — this is the Day 3 "citation mismatch" failure mode made measurable.
- If a claim can't be traced to retrieved text, the pipeline must remove, soften, or refuse it rather than ship it (per Day 3 Step 5 refusal rules and the "fluent unsupported answer is worse than a careful refusal" principle).

### 1.2 Request-type detection (router)
Detect and route each of these before generation runs:
- [ ] patient-specific diagnosis requests
- [ ] dosage / prescribing requests
- [ ] emergency / unsafe requests
- [ ] out-of-scope questions
- [ ] ambiguous questions
- [ ] prompt-injection attempts

### 1.3 Ambiguity flow
Implement:
```
AMBIGUOUS → ask clarification → resolve query → retrieve → answer
```
Reuse the Day 3 structured-answer contract once resolved — don't build a second answer format for the clarified path.

### 1.4 Prompt-injection test set
- Build injection tests in **English + Arabic** (Arabic matters because Workstream B is adding Arabic retrieval this same day — coordinate on shared test phrases).

### 1.5 Metrics this workstream must produce (feeds the Day 4 dashboard, section 4)
- Correct Refusal Rate
- False Refusal Rate
- Unsupported Claim Rate
- Injection Attack Success Rate

### 1.6 P1 bonus (only after P0 passes)
- One automatic regeneration attempt when a claim is unsupported.
- Stronger numeric / dosage / threshold consistency checks.

**Acceptance criteria:** all six detection categories have at least one passing + one adversarial test case; refusal message matches the Day 3 "good refusal" pattern (states what's missing, doesn't just say no); metrics land in a format Workstream D can ingest.

---

## 2. Workstream B — Multilingual Retrieval & Performance Optimization (P0)

**Owns:** `src/medical_rag/hybrid_retrieval.py`, `src/medical_rag/multilingual.py`, `src/medical_rag/query_decomposition.py`, `src/medical_rag/query_normalization.py`, `RAG/evaluation/`, `tests/test_multilingual.py`
**Does not own:** Streamlit UI, dashboard UI, safety router, database layer. Its job is the retrieval engine, full stop.

This is the workstream that both **satisfies Day 2's original retrieval requirements** (Precision@K discipline, Top-K tuning, chunk tuning, hybrid/reranking) **and layers in the new Day 4 asks** (multilingual, query decomposition, latency). Treat Day 2's method as the discipline; Day 4 as new scope on top of it.

### 2.1 Freeze current retrieval baseline first
Preserve and benchmark the existing stack before changing knobs:
```
nomic-embed-text + BM25 + RRF + current cross-encoder reranker + current Top-K/chunk settings
```
Record Precision@3, Precision@5, expected evidence rank, retrieval latency — this is what step 0's baseline should already contain; use it here.

### 2.2 Arabic + multilingual retrieval
- Arabic questions must retrieve the **same English NICE/WHO evidence** directly — no generic-translator main path.
- Test Arabic, English, and mixed Arabic-English queries.
- Preserve clinical terms/units exactly: `ICS`, `SABA`, `LABA`, `MART`, `FeNO`, drug names, `mg`, `mcg`, `%`, age ranges, thresholds.
- Add safe Arabic normalization that doesn't mangle English acronyms or numbers.
- Example mixed query to test against: "كثيراً، متى يجب تصعيد العلاج SABA طفل يستخدم؟"

### 2.3 Retrieval pipeline tuning (this is Day 2's Top-K / chunk-size work, continued)
Tune against the frozen baseline, one variable at a time (per Day 2's own rule — don't change many knobs at once):
- dense candidate count, BM25 candidate count, RRF constant, reranker candidate count, final Top-K, retrieval threshold, deduplication threshold, query normalization, medical term expansion, chunk size/overlap, batch size, caching.

Candidate flow to benchmark:
```
Dense Top-20 + BM25 Top-20 → RRF → Deduplicate → Rerank Top-10 → Final Top-5
```
Compare against Day 2's own recommended chunk sizes (400–600 tokens / 10–15% overlap for recommendation-style sections; 700–900 tokens for long-paragraph/table sections) rather than assuming new numbers are better.

### 2.4 Query preprocessing
Small medical query normalizer handling: Arabic punctuation/spacing, mixed Arabic-English, medical acronyms, spelling variants, exact drug names, numeric values/units, asthma terminology — without rewriting so aggressively the medical meaning changes.

### 2.5 Multi-question retrieval
For compound queries ("what symptoms suggest asthma, how is it diagnosed, and what first-line controller is recommended?"): split into independent retrieval intents (Q1/Q2/Q3), retrieve + rerank each independently, preserve each evidence group, pass all groups forward for one structured answer.

### 2.6 Latency profiling
Profile stage-by-stage: query preprocessing → embedding → dense retrieval → BM25 → RRF/fusion → deduplication → reranking → generation. Look for repeated model loads, oversized candidate counts, over-reranking, repeated embeddings, avoidable disk reads, missing caching, sequential work that could batch.

### 2.7 Benchmark candidates to actually run
```
BASELINE: nomic-embed-text + current reranker + llama3.2
MULTILINGUAL CANDIDATE: BGE-M3 + multilingual reranker + existing generator
OPTIONAL FULL MULTILINGUAL: BGE-M3 + multilingual reranker + multilingual-capable generator
```
Rule: don't replace the current stack because something is newer — only replace it if it wins on the team's own asthma benchmark on quality/latency trade-off.

### 2.8 Required output to hand off to Workstream D
baseline P@3/P@5, optimized P@3/P@5, Arabic P@3/P@5, expected chunk rank before/after, median latency before/after, chosen candidate counts/Top-K/RRF/reranker settings, documented failure cases, final retrieval config.

**Acceptance criteria:** every optimization has a measured before/after against the frozen baseline (this satisfies both Day 2's "measure, don't guess" principle and Day 4's requirement); Arabic retrieval numbers exist and are compared to English equivalents, not assumed.

---

## 3. Workstream C — Streamlit UI, Memory & Deployment (P0)

**Owns:** `app.py`, `src/medical_rag/memory.py`, `src/medical_rag/database.py`

### 3.1 Interface
- Stable Streamlit app, keeping the **Day 3 output structure** exactly: `1. Recommendation → 2. Supporting Evidence → 3. Citations → 4. Confidence & Safety`. Don't invent a new answer shape.
- Chat history for the current session.
- Expandable **Evidence Panel** (this is Day 2's "transparency before generation" principle, carried into Day 4's UI): chunk text, document, section, page, retrieval score, verification status.
- Arabic RTL display support.
- Visible safety disclaimer.
- App must work locally against the existing Ollama setup.
- Keep `demo.py` alive as a non-UI fallback.

### 3.2 Memory
- Today: short-term chat memory via Streamlit `session_state`.
- Optional persistence: SQLite with `conversations`, `messages`, `preferences` tables — store preferred language and conversation history only.
- **Do not** build a permanent patient medical profile — explicitly out of scope.

### 3.3 P1 bonus (after P0)
- "View Source Page" button rendering the real WHO/NICE PDF page for cited evidence.
- Integrate Workstream D's 📊 Evaluation dashboard tab into this app.

**Acceptance criteria:** UI never hides retrieved evidence; disclaimer always visible; app boots with no live dependency installation; `demo.py` still runs as backup.

---

## 4. Workstream D — Evaluation, Architecture & Day 5 Demo (P0)

**Owns:** `RAG/evaluation/`, `PROJECT_TECHNOLOGY_STACK.md`, `demo.py`, `tests/`

This workstream is the aggregation point — it depends on outputs from A, B, and C.

### 4.1 Required Day 4 Evaluation Dashboard (this is a required deliverable, not a bonus)
Build a Streamlit tab (📊 Evaluation) showing:

| Metric | Formula | Required |
|---|---|---|
| Precision@3 | relevant chunks in Top-3 ÷ 3 | ✅ |
| Precision@5 | relevant chunks in Top-5 ÷ 5 | ✅ |
| Citation Accuracy | correct supporting citations ÷ total checked | ✅ |
| Faithfulness | supported claims ÷ total claims | ✅ |
| Unsupported Claim Rate | unsupported claims ÷ total claims | ✅ |
| Correct Refusal Rate | correctly refused ÷ should-refuse cases | ✅ |
| False Refusal Rate | wrongly refused ÷ valid questions | ✅ |
| Clarification Accuracy | correctly routed ÷ ambiguous questions | ✅ |
| Multi-question Completion Rate | sub-questions answered+cited ÷ total | ✅ |
| Arabic Retrieval Performance | Arabic P@K vs English equivalents | ✅ if Arabic kept |
| Prompt Injection Success Rate | successful attacks ÷ attempts | ✅ if injection tested |
| Median Latency | end-to-end median | recommended |

Dashboard sections: metric cards → retrieval results table by test case → safety/guardrail results table → Arabic benchmark table → prompt-injection results table → **failure analysis** (at least one documented failure with question/expected/actual/failure type/probable cause/fix/result-after-fix).

### 4.2 Benchmark set
20–30 structured test cases: 6 direct, 4 paraphrased/abbreviation, 4 ambiguous, 4 multi-question, 4 out-of-scope/patient-specific, 4 Arabic equivalents, plus a separate prompt-injection set. If time-constrained, prioritize retrieval + citation accuracy + faithfulness + refusal behavior first.

Each benchmark row: `test_id, question, language, category, expected_document, expected_section, expected_chunk_id, should_answer, should_clarify, should_refuse, retrieved_top_3, retrieved_top_5, precision_at_3, precision_at_5, citation_correct, claims_supported, claims_total, unsupported_claims, latency_ms, notes`.

### 4.3 Architecture decisions to document in `PROJECT_TECHNOLOGY_STACK.md`
For each: chosen tech, alternatives considered, why chosen, why rejected, benchmark evidence, limitations. Required comparisons:
- UI: Streamlit vs Chainlit vs FastAPI
- Embeddings: nomic-embed-text vs multilingual alternative
- Generator: llama3.2 vs multilingual-capable alternative
- Memory: session_state+SQLite vs LangGraph memory
- Guardrails: custom deterministic safety layer vs external guardrail frameworks

### 4.4 Day 5 demo prep
Prepare four demo cases:
- **Case A (Success):** clear question → correct retrieval → structured answer → exact citation.
- **Case B (Complex):** multi-question or follow-up → multiple evidence sections → combined structured answer.
- **Case C (Safe Refusal):** diagnosis/dosage/out-of-scope/unsafe request → correct refusal.
- **Bonus:** same grounded question in Arabic.

Save before freezing: final P@3/P@5, citation accuracy, faithfulness/unsupported-claim rate, refusal performance, one documented retrieval failure + fix, Arabic benchmark result, injection result, final config (embedding model, chunk size/overlap, Top-K, reranker, generator, confidence thresholds).

**Acceptance criteria:** dashboard renders all required metrics from real Phase A/B/C outputs (not placeholders); `PROJECT_TECHNOLOGY_STACK.md` has evidence-backed comparisons, not opinions; demo script rehearsed against all four cases.

---

## 5. Integration Order (run in this sequence once workstreams are individually done)

```
1. Safety / Router
      ↓
2. Arabic + Query Decomposition
      ↓
3. Existing RAG Pipeline (Day 1–3 core)
      ↓
4. UI / Memory
      ↓
5. Full Evaluation
      ↓
6. Freeze
```

## 6. Shared Interface Agreement (agree on this before Phase 1 starts, so workstreams can build in parallel)

```python
RAGResponse(
    status,
    language,
    query,
    resolved_query,
    recommendation,
    evidence,          # list[Evidence]
    citations,
    confidence,
    safety_message,
    clarification_question=None,
)

Evidence(
    chunk_id,
    text,
    document_name,
    section_title,
    page_number,
    retrieval_score,
    verification_status,
)
```
Every workstream reads/writes only this contract — it's what lets A, B, and C work independently without breaking each other.

## 7. Shared Definition of Done

Not done until **all** of these hold:
- [ ] Day 1–3 functionality still works, all previous tests pass
- [ ] Retrieval remains traceable; every recommendation has a citation that actually supports the claim
- [ ] Weak evidence triggers low confidence or refusal
- [ ] Ambiguous questions ask for clarification instead of guessing
- [ ] Multi-question prompts don't lose sub-questions
- [ ] Arabic retrieval is measured, not assumed
- [ ] Prompt injections are tested
- [ ] UI never hides evidence
- [ ] `demo.py` remains a working backup
- [ ] Models downloaded and vector DB built *before* the demo — no live installs during presentation

## 8. Explicitly out of scope today (do not let Antigravity wander into these)
Replacing Chroma for its own sake · rebuilding on LangGraph · React frontend · fine-tuning an LLM · new broad medical datasets · medical image diagnosis · fake diagnosis probabilities · voice assistant · full auth/accounts · Kubernetes/cloud infra.

**Language guardrail for docs/demo copy:** never claim "zero hallucination" or "100% safe." Use: *"Evidence-grounded and hallucination-resistant, with measured unsupported-claim and refusal performance."*

## 9. Final target pipeline (what Day 5 judges should see)
```
User Question
   → Safety / Ambiguity Router
   → Hybrid Retrieval + Reranking
   → Evidence Threshold
   → Grounded Generation
   → Claim Verification
   → Structured Answer + Citations
   → Transparent Evidence Panel
```
The goal isn't just "the system works" — it's showing *how you know* it works, via the dashboard numbers in section 4.

---

## How to run this with Antigravity

Suggested prompt pattern per phase (adjust paths to your real repo once uploaded):

> "Working in `src/medical_rag/`, implement Workstream [A/B/C/D] from `RAG_Implementation_Plan_Day4.md`, section [N]. Do not modify files outside its 'Owns' list. Run the existing test suite before and after your change and report pass/fail. Stop and report before starting P1 bonus items."

Run Workstream A, B, C in parallel agent sessions (they own disjoint files), then run Workstream D last since it depends on their outputs, then run the Section 5 integration order as a final pass.
