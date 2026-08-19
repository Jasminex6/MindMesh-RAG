You are the senior software architect and implementation planner for an existing **Pediatric Asthma Clinical Decision Support RAG hackathon project**.

Your task in this message is **PLANNING ONLY**.

**Do not implement production code yet.**

I want you to inspect the repository deeply, understand the existing architecture, make the remaining technical decisions, explain why you made them, and produce an implementation plan that I can later execute piece-by-piece by saying:

`execute A`

or:

`execute C`

etc.

The critical requirement is:

> **A, B, C, D, E, and F must be independently executable implementation units.**
>
> If I start with C before A or B exists, C must still be implementable correctly.
>
> Later, when A and B are implemented, everything must integrate without rewrites or competing architectures.

---

# 1. Project Context

This is a 5-day hackathon project.

Days 1–3 are already substantially implemented.

The current documented architecture is approximately:

```text
Official WHO / NICE PDFs
        ↓
PDF Parsing + Cleaning
        ↓
Section-Aware Chunking
        ↓
Embeddings
        ↓
ChromaDB Dense Search
        +
BM25 Sparse Search
        ↓
Reciprocal Rank Fusion
        ↓
Cross-Encoder Reranking
        ↓
Pre-LLM Refusal Gate
        ↓
Grounded Generation
        ↓
Citation Verification
        ↓
Structured Clinical Response
```

The documented stack currently includes approximately:

```text
Python 3.12
uv

pypdf
custom regex cleaning
custom section-aware chunking

Ollama
nomic-embed-text
llama3.2:3b

ChromaDB
rank-bm25
Reciprocal Rank Fusion
sentence-transformers cross-encoder

langchain-ollama

demo.py CLI

unittest
Pyright
```

Do not assume the documentation is perfectly synchronized with the code.

**Inspect the actual repository and treat working code as evidence.**

---

# 2. Files / Sources You Must Inspect First

Before designing anything, inspect whatever of these are available:

```text
PROJECT_TECHNOLOGY_STACK.md

Day4_Team_Workstreams.md
or the current Day 4 workstream note if named differently

Day1.pptx
Day2.pptx
Day3.pptx

Overview.pdf.pdf
AI AGENDA.pdf

pyproject.toml
uv.lock
requirements files if any

demo.py

src/
tests/
RAG/
Docs/
```

Then inspect the actual project tree.

Pay special attention to:

```text
src/medical_rag/
```

and locate the real implementations for:

```text
configuration
ingestion
chunking
embedding
vector repository
BM25
hybrid retrieval
RRF
reranking
generation
safety/refusal
citation verification
evaluation
```

Do not design duplicate systems if equivalent functionality already exists.

---

# 3. My Personal Role Has Changed

The team workstream note may describe different team assignments.

For **my personal work today**, treat my role as:

# Local AI Systems, Deployment & Performance

I am a beginner in AI engineering.

I do **not** want my entire day to consist of:

- asking an AI agent to generate benchmarks,
- randomly changing model parameters,
- blindly tuning RRF constants,
- generating tests I do not understand,
- making pretty UI CSS,
- or letting abstractions hide the underlying technology.

I want **high-yield technical discovery**.

The implementation should deliberately expose me to the technologies underneath an AI application.

I want to physically understand things such as:

```text
browser
↓
Streamlit
↓
Python process
↓
application service
↓
RAG pipeline
↓
Ollama HTTP server
↓
local model
```

and:

```text
conversation
↓
session state
↓
persistent storage
↓
context reconstruction
```

and:

```text
request
↓
embedding
↓
retrieval
↓
reranking
↓
generation
↓
streaming response
```

---

# 4. Fixed Decisions

These decisions were already agreed on.

Do not re-litigate them unless repository inspection reveals a serious technical reason they are impossible.

## 4.1 Preserve the existing RAG

Do not rewrite the Day 1–3 RAG pipeline.

The existing:

```text
ingestion
chunking
Chroma
BM25
RRF
reranking
grounded generation
citations
safety
```

must remain usable.

Any application layer must **consume** the RAG system instead of duplicating it.

---

## 4.2 Streamlit is the hackathon UI

Use **Streamlit** for today's application unless an actual repository constraint makes it infeasible.

I specifically want to learn:

```text
Streamlit rerun model
st.session_state
st.cache_resource
st.cache_data
chat components
streaming
resource lifetime
```

Do not spend excessive time on visual styling.

---

## 4.3 Keep the CLI

`demo.py` must remain functional as a backup for Day 5.

The Streamlit app cannot become the only way to run the system.

---

## 4.4 Short-term chat memory

For the active UI session:

```text
Streamlit session state
```

should hold short-term conversation state.

---

## 4.5 Persistent storage

Use **SQLite** for lightweight persistence unless repository inspection provides a strong reason not to.

Persistent data should be intentionally limited.

Reasonable persistence:

```text
conversation metadata
messages
preferred language
basic UI preferences
timestamps
```

Do not build a permanent medical patient profile.

Do not persist unnecessary sensitive clinical information.

---

## 4.6 Local-first

The project currently uses Ollama and local models.

Keep the demo local-first.

Do not introduce:

```text
Kubernetes
complex cloud infrastructure
microservices
React frontend
production authentication
enterprise databases
```

unless absolutely required.

They are outside today's goal.

---

# 5. Decisions YOU Must Make

After inspecting the repository, make concrete decisions on the following.

Do not leave these as vague alternatives.

For each decision provide:

```text
Chosen approach
Alternatives considered
Why chosen
Why alternatives were rejected for THIS project
Trade-offs
Future upgrade path
```

Decide:

### Ollama boundary

Should the application:

- continue using `langchain-ollama` for generation,
- use direct Ollama HTTP for runtime diagnostics,
- use direct Ollama streaming,
- or combine those approaches?

Do not replace working LangChain integration without a reason.

I want to understand that Ollama is a **server process**, not magic.

---

### Application service boundary

Determine the cleanest existing or new boundary for something approximately like:

```python
rag.ask(...)
```

The UI must not know how:

```text
BM25
Chroma
RRF
reranking
generation
```

work internally.

---

### Shared response contract

Inspect existing response models.

Reuse them if possible.

Otherwise design one canonical contract approximately like:

```python
RAGResponse(
    status,
    language,
    query,
    resolved_query,
    recommendation,
    evidence,
    citations,
    confidence,
    safety_message,
    clarification_question=None,
)
```

with evidence approximately:

```python
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

Do not introduce unnecessary Pydantic/dataclass layers if the existing project already solves this cleanly.

---

### SQLite approach

Choose between:

```text
Python sqlite3
SQLAlchemy
another already-installed lightweight abstraction
```

Prefer the approach that teaches the actual database model without introducing pointless complexity.

Define:

```text
schema
connection lifecycle
transaction behavior
repository/service boundary
```

---

### Streamlit caching

Explicitly decide what belongs in:

```python
st.cache_resource
```

versus:

```python
st.cache_data
```

versus:

```python
st.session_state
```

I want the reasoning.

Examples you must inspect:

```text
embedding model
reranker
Chroma connection
BM25 corpus/index
Ollama client
configuration
conversation
query results
PDF page rendering
```

Do not cache something simply because caching sounds faster.

---

### Streaming architecture

Decide how response streaming should work.

Explain whether streaming comes from:

```text
Ollama directly
LangChain callback / stream
generator abstraction
other existing mechanism
```

The UI should not become tightly coupled to one backend mechanism.

---

### Profiling

Choose a practical profiling strategy.

At minimum I want stage timing for:

```text
query preprocessing
embedding
dense retrieval
BM25
RRF
reranking
generation
total request
```

Also decide when to use:

```python
time.perf_counter()
```

and whether tools such as:

```text
cProfile
py-spy
memory profiling
```

are actually worth using today.

---

### Configuration

Decide where configuration belongs.

Avoid:

```text
hard-coded ports
hard-coded model names scattered across files
duplicated constants
```

Reuse the existing config design if possible.

---

### Error handling

Define graceful behavior for:

```text
Ollama not running
model not installed
Chroma unavailable/corrupt
database unavailable
retrieval failure
generation timeout
Streamlit refresh
missing guideline index
```

The system should fail informatively rather than explode during the demo.

---

# 6. Plan the Work as Six Independent Units

Within THIS personal plan:

`A`, `B`, `C`, `D`, `E`, `F`

refer to the implementation units below.

They are **NOT** the A/B/C/D team workstreams from the team note.

---

# A — Ollama Runtime & Local Model Boundary

Goal:

Understand and implement the boundary:

```text
Python
↓ HTTP / client
Ollama
↓
local model
```

Plan for:

```text
Ollama health check
server availability
model availability
model warm-up if useful
request lifecycle
streaming capability
timeouts
useful runtime diagnostics
cold vs warm behavior
```

I want at least one way to manually observe or call the Ollama HTTP interface so I understand what is happening underneath LangChain.

Do not replace the project's working generation stack just for educational purity.

---

# B — Streamlit Application Lifecycle

Goal:

Create a simple, reliable Streamlit application shell around the existing RAG system.

Focus on understanding:

```text
reruns
session state
resource caching
data caching
chat history
component lifecycle
errors
```

The final clinical response must preserve the Day 3 structure:

```text
1. Recommendation
2. Supporting Evidence
3. Citations
4. Confidence & Safety
```

Also support:

```text
evidence panel
Arabic RTL display when needed
safe disclaimer
```

The interface should have an integration point for the team's required Evaluation Dashboard, but **my work does not need to own the benchmark logic itself**.

Do not waste time on elaborate styling.

---

# C — SQLite Conversation Persistence

Goal:

Learn what "memory" actually means rather than hiding it behind LangChain.

Plan a minimal relational model such as:

```text
conversations
messages
preferences
```

I want to understand:

```text
CREATE TABLE
INSERT
SELECT
UPDATE
primary keys
foreign keys where appropriate
transactions
connection lifecycle
```

The persistence module must be independently usable without Streamlit.

The Streamlit layer should later consume it.

Do not create a permanent medical patient profile.

---

# D — RAG Application Adapter / Service Boundary

Goal:

Expose the existing Day 1–3 RAG pipeline through one clean application-facing API.

Approximately:

```python
result = rag_service.ask(
    query=query,
    conversation_context=context,
)
```

but inspect the code and choose the shape that actually fits.

The application layer must not know implementation details such as:

```text
Chroma
BM25
RRF
cross-encoder
prompt internals
```

This unit must work from:

```text
CLI
Streamlit
future API
```

without duplicating logic.

Preserve `demo.py`.

---

# E — Streaming, Profiling & Performance

Goal:

Find where time is actually spent and optimize the **important bottlenecks**.

Measure something comparable to:

```text
application startup
model warm-up
query preprocessing
embedding
dense retrieval
BM25
RRF
reranking
generation
database
total latency
```

Distinguish:

```text
cold start
warm request
```

Explore practical optimizations such as:

```text
correct resource caching
avoiding repeated model initialization
avoiding repeated DB/index loading
batching reranker calls
reducing unnecessary serialization
streaming perceived latency
connection reuse
```

Do not optimize a 20 ms component while generation takes 6 seconds.

Prioritize according to measured bottlenecks.

Produce metrics that can later be consumed by the team's Evaluation Dashboard.

---

# F — Embeddings Learning Lab

This is intentionally a small educational unit.

It must **not alter production retrieval by default**.

I want a simple executable experiment that helps me understand:

```text
text
↓
embedding vector
↓
cosine similarity
↓
semantic retrieval
```

Use examples such as:

```text
"childhood asthma symptoms"

"signs of asthma in children"

"banana bread recipe"
```

and, if the current embedding model supports meaningful comparison:

```text
English asthma query
Arabic equivalent
```

I want to inspect:

```text
vector dimension
similarity scores
nearest relationships
```

The point is to build intuition for what Chroma is actually searching.

Do not turn F into a giant embedding benchmark.

---

# 7. ORDER INDEPENDENCE IS NON-NEGOTIABLE

This is the most important architectural constraint.

I may execute tasks in any order:

```text
C → A → E → B → D → F
```

or:

```text
A → D → B → C → E → F
```

or any other order.

You must design the plan so this is safe.

---

# 8. Shared Contract / Bootstrap Rule

Create one clearly documented **canonical shared contract location**.

First inspect whether the repository already has an appropriate location.

Reuse it if possible.

If not, choose the minimal correct location.

When I later say:

`execute C`

and the canonical contract file does not yet exist:

C may create **only the minimal shared declarations required by C**, exactly according to the master plan.

It must NOT secretly implement A, B, D, or E.

Later tasks must reuse those declarations rather than creating competing types.

Likewise:

If B is executed before D, B must use:

```text
a narrow adapter
existing RAG entry point
or contract-compatible temporary bridge
```

It must **not copy the RAG implementation into Streamlit**.

When D later executes, the bridge should be replaceable with the canonical service without redesigning B.

---

# 9. File Ownership Is Required

Your plan must specify exact files.

For every implementation unit, produce a file-by-file plan.

Example format:

```text
UNIT C

CREATE:
src/medical_rag/persistence/sqlite_store.py

Purpose:
...

Public API:
...

Reads from:
...

Called by:
...

Must NOT import:
...

Reason for location:
...

If C is implemented first:
...

If B already exists:
...

If B does not exist:
...

Manual verification:
...

Integration verification:
...
```

Do this for **every file you propose creating or modifying**.

Do not say vague things such as:

> "modify the backend"

Give the exact path.

---

# 10. Avoid File Collisions

Minimize situations where A, B, C, D and E all need to heavily edit the same file.

Prefer:

```text
small focused modules
stable public interfaces
dependency injection where useful
composition
```

over one giant:

```text
app.py
```

containing everything.

If two implementation units genuinely must touch the same file:

1. explain why,
2. identify the exact integration point,
3. explain how either execution order remains safe.

---

# 11. Recommended Order vs Required Order

You may recommend an ideal learning / implementation order.

For example, you may decide something approximately like:

```text
A
↓
B
↓
C
↓
D
↓
E
↓
F
```

But this is only a **recommended order**.

It must NOT become a hard dependency chain.

For every unit explicitly provide:

```text
Hard dependencies:
Soft dependencies:
Can execute first? YES/NO
Fallback if related unit is absent:
```

My expectation is that almost every unit should be:

```text
Can execute first? YES
```

through contracts/adapters.

If you believe one unit truly cannot be independent, justify it strongly and redesign first before accepting that limitation.

---

# 12. Manual Learning Requirements

I do not want AI to hide the interesting parts from me.

Every unit must include a section:

# What I Should Manually Explore

Give me concrete things to inspect myself.

Examples:

## A

```text
check Ollama process
inspect port 11434
send a manual HTTP request
observe cold vs warm response
inspect installed models
```

## B

```text
add a visible rerun counter temporarily
observe Streamlit reruns
watch when cached resources initialize
refresh the page
observe session state behavior
```

## C

```text
open SQLite
inspect schema
SELECT rows manually
insert a message
close the program
restart
verify persistence
```

## E

```text
time each stage
compare cold and warm requests
disable one cache
observe the difference
```

## F

&#x20;compare medically related vs unrelated text

The point is **learning by observing the system**, not merely accepting generated code.

---

# 13. Acceptance Checks, Not Test Theater

This is still a medical hackathon, so correctness matters.

However, do not turn my personal workstream into a giant unit-test-writing exercise.

For each implementation unit define:

```text
small automated smoke checks where valuable
manual verification steps
integration check
failure behavior
```

The rest of the team owns broader Day 4 evaluation.

Existing tests must continue passing.

Do not remove or weaken existing tests.

---

# 14. Integration With Other Team Work

Other teammates may independently implement:

```text
safety / guardrails
Arabic retrieval
multi-question routing
evaluation dashboard
benchmarks
```

My modules must not block them.

Design application interfaces so future fields such as:

```text
clarification_question
verification_status
language
safety_message
timing data
```

can pass cleanly through the UI and service layer.

Do not hard-code assumptions that every query always produces a normal answer.

The application must be able to represent states such as:

```text
ANSWER
REFUSAL
NEEDS_CLARIFICATION
ERROR
```

---

# 15. Day 4 Dashboard Integration

The project requires a Day 4 evaluation/dashboard effort elsewhere in the team.

My performance work should make it easy for that dashboard to consume:

```text
request latency
stage latency
cold/warm status
model/runtime status
possibly cache-hit information
```

Choose a simple representation.

Do not couple the entire RAG engine to Streamlit just to expose metrics.

Prefer something reusable such as:

```text
structured timing object
dictionary
dataclass
JSON-serializable record
```

depending on existing conventions.

---

# 16. Deployment Goal

The final local demo should have a boring, reliable startup procedure.

Plan for something approximately like:

```text
1. verify Ollama
2. verify required models
3. verify vector DB/index
4. start Streamlit
5. open app
```

I want the startup sequence documented.

Do not make the judges watch:

```text
package installation
model downloads
index building
```

during the live demo.

Plan a simple preflight / health-check mechanism if worthwhile.

Also preserve:

```text
python demo.py
```

or its actual existing equivalent as the emergency fallback.

---

# 17. Performance Philosophy

I care a lot about optimization.

Do not let that turn into random micro-optimization.

Use this rule:

> **Measure → identify bottleneck → understand cause → change one thing → measure again.**

I specifically want the plan to teach me the difference between:

```text
throughput
latency
cold-start latency
warm latency
actual latency
perceived latency
resource initialization
caching
```

If generation dominates total response time, say so.

If retrieval is already negligible, do not pretend optimizing it by 30% matters.

---

# 18. Architecture Must Remain Hackathon-Sized

Reject unnecessary architecture astronautics.

Do not create 40 files because a production SaaS might someday need them.

I want:

```text
clean
modular
understandable
easy to debug
easy to demo
```

architecture.

Prefer the smallest design that establishes good boundaries.

---

# 19. REQUIRED OUTPUT FROM YOU NOW

Again:

**DO NOT CODE YET.**

Create a planning document in the repository.

Choose an appropriate location based on the existing documentation structure.

A reasonable default if nothing better exists is:

```text
DAY4_LOCAL_AI_SYSTEMS_PLAN.md
```

Your response and plan file must contain the following sections.

---

## 19.1 Repository Findings

Explain:

```text
what exists
what is actually implemented
what documentation is stale
existing useful abstractions
existing technical debt relevant to my role
```

Do not invent findings.

---

## 19.2 Final Architecture

Show the architecture after these units are complete.

Use ASCII / Mermaid if appropriate.

Clearly show boundaries between:

```text
Streamlit
application service
conversation store
RAG
Ollama
profiling
```

---

## 19.3 Decision Record

Table:

| Decision | Chosen | Alternatives | Why chosen | Why rejected | Trade-off |
| -------- | ------ | ------------ | ---------- | ------------ | --------- |

Include every important decision from Section 5.

---

## 19.4 Shared Contracts

Define the exact contracts that allow independent execution.

Include:

```text
canonical path
public API
who owns it
how absent modules are handled
```

---

## 19.5 File Ownership Map

Table:

| File | Create / Modify | Unit | Purpose | Other units allowed to touch? |
| ---- | --------------- | ---- | ------- | ----------------------------- |

This must cover every planned file.

---

## 19.6 Unit A Plan

Include:

```text
Goal
Why this is high-yield
Files
File-by-file implementation plan
Public interfaces
Hard dependencies
Soft dependencies
Can execute first?
Fallback when other units do not exist
Manual exploration
Smoke checks
Integration checks
Expected failure modes
What I should understand after completing it
```

---

## 19.7 Unit B Plan

Same structure.

---

## 19.8 Unit C Plan

Same structure.

---

## 19.9 Unit D Plan

Same structure.

---

## 19.10 Unit E Plan

Same structure.

---

## 19.11 Unit F Plan

Same structure.

---

## 19.12 Execution Independence Matrix

Create a matrix such as:

| Start With | Works? | What bootstrap happens | What must NOT happen |
| ---------- | ------ | ---------------------- | -------------------- |
| A          | YES    | ...                    | ...                  |
| B          | YES    | ...                    | ...                  |
| C          | YES    | ...                    | ...                  |
| D          | YES    | ...                    | ...                  |
| E          | YES    | ...                    | ...                  |
| F          | YES    | ...                    | ...                  |

Then explicitly reason through at least:

```text
C implemented first
B implemented first
E implemented before A
D implemented before B
```

Show why they remain integratable.

---

## 19.13 Recommended Learning Order

Give your recommended order and explain why.

Again:

recommended ≠ required.

---

## 19.14 Final Integration Sequence

Explain how all completed modules connect regardless of implementation order.

---

## 19.15 Demo Preflight

Define the final pre-demo checks.

---

# 20. FUTURE EXECUTION PROTOCOL

The plan must be written so future Antigravity sessions can use it.

When I later say:

`execute A`

you must:

1. Re-read the master plan.
2. Inspect the repository as it exists at that moment.
3. Detect which other units have already been implemented.
4. Implement **only Unit A**.
5. Create only minimal shared-contract/bootstrap pieces if they are absent.
6. Do not secretly implement B/C/D/E/F.
7. Integrate with already-existing units instead of overwriting them.
8. Preserve existing behavior.
9. Run the unit's smoke checks.
10. Run relevant existing regression tests.
11. Show me what changed.
12. Explain the important technology I should inspect manually.
13. Tell me exactly how to manually verify the behavior.
14. Report any deviation from the original plan and explain why it became necessary.

Apply the same rule to:

```text
execute B
execute C
execute D
execute E
execute F
```

---

# 21. DO NOT DO THESE THINGS

Do not:

```text
rewrite the RAG from scratch
replace Chroma for no reason
replace Ollama for no reason
rebuild with LangGraph
add React
add Kubernetes
add authentication
fine-tune a model
add a broad new medical corpus
build medical-image diagnosis
create fake diagnostic confidence percentages
remove demo.py
duplicate RAG logic in Streamlit
hide database logic behind unnecessary frameworks
optimize without measuring
implement all units when I ask for one
```

---

# 22. Key Philosophy

I want to finish today understanding:

```text
how a local model is served
how Python talks to it
how Streamlit actually executes
how resources live across reruns
how session memory differs from persistent memory
how SQLite stores conversations
how the RAG becomes an application service
how streaming works
where latency actually comes from
what caching really changes
what an embedding physically is
why semantic retrieval works
```

The goal is not simply:

> "AI generated me an app."

The goal is:

> **I can draw the system from memory, inspect every important boundary, break it intentionally, diagnose why it broke, and explain why the architecture was chosen.**

Now inspect the repository and create the master plan.

**Do not implement A–F yet.**
