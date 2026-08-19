# Day 3: Grounded Generation & Citations

## 1. What We Built

We extended the existing retrieval-only pipeline into a full **grounded answer generation system** that:

- Takes retrieved guideline chunks as its **only source of truth**
- Produces structured answers formatted strictly to the official hackathon specifications
- Refuses to answer when evidence is missing, weak, or the question is patient-specific
- Validates every citation against the actual retrieved chunks post-generation

**Pipeline Data Flow:**

```text
Query
  → UnifiedRetriever.search()              # Day 2 Hybrid Retrieval
  → assess_confidence(results)              # Retrieval-score-based confidence
  → check_refusal(query, results)           # Pre-LLM Safety Gate (Patient-specific & threshold check)
  → build_evidence_block(results)           # Format top-k chunks with provenance & scores
  → call_llm(system_prompt, user_prompt)    # Ollama llama3.2 execution
  → parse_llm_response(raw_output)          # Structured JSON extraction
  → verify_citations(citations, results)    # Validate chunk_ids & enrich metadata
  → post_generation_safety_check()        # Final safety audit & fallback sweep
  → GeneratedAnswer                         # Formatted 4-component structured output
```

---

## 2. Technical Implementation Details

### A. Calibrated Retrieval Geometry & Score Thresholds
Because vector distance spaces vary by embedding model, we calibrated our score thresholds specifically for `nomic-embed-text` cosine relevance scores:
- **`High` Confidence**: Top score $\ge 0.42$ AND $\ge 2$ results $\ge 0.35$.
- **`Medium` Confidence**: Top score $\ge 0.35$ AND $\ge 1$ result $\ge 0.25$.
- **`Low` Confidence**: Top score $\ge 0.25$.
- **`Insufficient Evidence`**: Top score $< 0.25$ or empty results.

### B. Answer Formatting (Matching Official Slide Deck)
The system outputs a 4-component structured answer:

1. **Recommendation**: Short, direct, clinical statement based ONLY on retrieved chunks. No patient-specific treatment.
2. **Supporting Evidence**: Bullet points containing short excerpts from the exact cited guideline chunks.
3. **Citations**: Complete provenance breakdown for each cited claim:
   - Document Name
   - Section Heading
   - Physical Page Number(s)
   - Chunk ID
   - Retrieval Relevance Score
   - Verification Status (`[VERIFIED]` / `[UNVERIFIED]`)
4. **Confidence & Safety**: Retrieval-derived confidence label + clinical disclaimer note.

### C. Pre-LLM & Post-LLM Refusal Guards
- **Patient-Specific Regex Gate**: Pre-LLM check using pattern matching (`my child`, `how much should I take`, `weighing 25kg`, `patient presenting with`) that returns an instant refusal before invoking the LLM.
- **Out-of-Scope / Insufficient Evidence Gate**: When retrieval quality is below threshold or when the LLM outputs `"Insufficient Evidence"`, the system returns a grounded refusal.

---

## 3. End-of-Day Review & Mentor Discussion Points

Use these points during mentor reviews to demonstrate team readiness for **Day 4: Safety & Evaluation**:

### 🎯 Grounding Quality
- **Does the answer use only retrieved context?**  
  *Yes.* The system prompt explicitly forbids the LLM from adding external medical knowledge from its training weights.
- **Are unsupported claims removed?**  
  *Yes.* Post-generation verification checks all cited `chunk_id`s against retrieved results. Unverified claims trigger safety warnings and confidence downgrades.
- **Does it refuse weak evidence?**  
  *Yes.* Queries with top retrieval score $< 0.25$ or off-topic queries (e.g. "diabetes", "appendicitis") trigger refusal.

### 📜 Citation Quality
- **Does each recommendation have a citation?**  
  *Yes.* The LLM is required to map every claim to its exact `chunk_id`.
- **Are document, section, and page shown?**  
  *Yes.* `verify_citations()` enriches every valid citation with document title, section heading, physical PDF page numbers, and retrieval score.
- **Do citations actually support the claims?**  
  *Yes.* `verify_citations()` cross-references chunk IDs to ensure non-existent or hallucinated IDs are flagged as `[UNVERIFIED]`.

### 🛡️ Safety Quality
- **Is there a disclaimer?**  
  *Yes.* Every output includes an explicit disclaimer: *"Grounded in official guideline evidence. Clinical judgment required."*
- **Is confidence shown carefully?**  
  *Yes.* Confidence is derived from vector retrieval similarity geometry, NOT from LLM self-confidence.
- **Are patient-specific requests handled safely?**  
  *Yes.* Patient-specific requests (e.g., specific age/weight dosage prompts) are caught by the Pre-LLM Safety Gate and refused immediately.

---

## 4. Key Files & Structure

| File | Purpose |
|---|---|
| [`src/medical_rag/generation.py`](file:///c:/Users/Yasmine/Downloads/Orange%20x%20Instant/src/medical_rag/generation.py) | Core generation service, confidence assessment, refusal gate, prompt assembly, citation verification, safety checks |
| [`demo.py`](file:///c:/Users/Yasmine/Downloads/Orange%20x%20Instant/demo.py) | Interactive CLI demo tool for live testing and mentor presentation |
| [`tests/test_generation.py`](file:///c:/Users/Yasmine/Downloads/Orange%20x%20Instant/tests/test_generation.py) | 22 unit tests for generation components |
| [`tests/run_generation_tests.py`](file:///c:/Users/Yasmine/Downloads/Orange%20x%20Instant/tests/run_generation_tests.py) | Standalone test runner (34 tests) |
| [`tests/test_generation_live.py`](file:///c:/Users/Yasmine/Downloads/Orange%20x%20Instant/tests/test_generation_live.py) | End-to-end integration test runner (6 live clinical scenarios) |

---

## 5. How to Run & Demonstrate

### Run Automated Unit Tests (48/48 Passing)
```powershell
python -m unittest discover -s tests -v
```

### Run Interactive Demo
```powershell
python demo.py
```

### Run Single Query Demo
```powershell
python demo.py "What are the common symptoms of asthma"
```
