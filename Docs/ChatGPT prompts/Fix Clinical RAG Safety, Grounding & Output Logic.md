# Fix Clinical RAG Safety, Grounding & Output Logic

Inspect the current Pediatric Asthma CDS implementation and **implement the fixes below without rebuilding the working ingestion/retrieval architecture**. Preserve existing hybrid retrieval, reranking, citations, and UI unless a change is required for correctness.

### Required fixes

1. **Input intent/risk classification**
   - Classify queries by meaning, not simple keywords:
     - `GENERAL_EDUCATION`
     - `PATIENT_SCENARIO`
     - `DOSAGE_REQUEST`
     - `EMERGENCY`
     - `OUT_OF_SCOPE`
     - `PROCEDURAL_GUIDANCE`
   - Patient-specific diagnosis/treatment requests must not pass through normal answer generation.
   - Do not over-refuse general educational questions.

2. **Evidence sufficiency / answerability gate**
   - After reranking and before generation, determine whether retrieved evidence actually answers the user's intent.
   - A high similarity score alone is **not sufficient**.
   - If evidence is merely related but does not contain the requested information, return `Insufficient Evidence`.
   - Example: a spacer recommendation does **not** answer “how to use an inhaler.”

3. **Real citation verification**
   - Current verification must not merely confirm that a cited `chunk_id` exists.
   - Verify that each generated clinical claim is actually supported by the cited chunk text.
   - Only mark a citation `VERIFIED` when semantic claim → evidence support is established.
   - Unsupported claims must be removed, regenerated, or cause an insufficient-evidence response.

4. **Confidence calculation**
   - Stop deriving `High/Medium/Low` from top retrieval score alone.
   - Confidence must reflect:
     - retrieval relevance,
     - answerability/evidence match,
     - claim-to-citation support,
     - citation coverage,
     - safety/guardrail result.
   - Never show `High` when the retrieved evidence does not directly answer the question.

5. **Clean refusal/output path**
   - Once a query is refused or evidence is insufficient, do not populate normal recommendation/evidence cards with irrelevant chunks.
   - Do not leave empty headings such as `1. RECOMMENDATION`.
   - Retrieved debugging passages may remain available separately for developer inspection, but must not appear as supporting evidence to the user.

6. **Metadata quality**
   - Inspect section-title extraction.
   - Prevent ordinary body sentences such as “Recommendations were based on a balance of benefits and harms…” from being stored/displayed as section titles.
   - Preserve real document, section, page, chunk ID, and score provenance.

### Mandatory regression cases

Use these exact queries and verify behavior:

- **“asthma symptoms”**
  - Answer symptoms from directly relevant asthma evidence.
  - Do not present bronchiolitis, occupational asthma, or acute-treatment chunks as supporting evidence.
  - Confidence should reflect only evidence actually used.

- **“what are asthma symptoms”**
  - Same expected behavior despite paraphrasing.

- **“can i eat ice cream”**
  - Clean insufficient/out-of-scope response.
  - No irrelevant asthma evidence presented as support.

- **“breast cancer screening info”**
  - Preserve the existing correct out-of-scope refusal with zero unnecessary retrieval.

- **“i have been coughing all night, might it be asthma?”**
  - Detect as a patient-specific diagnostic scenario before normal generation.
  - Do not cite unrelated asthma-monitoring evidence as an answer.

- **“how to use an inhaler”**
  - Only provide technique if retrieved guideline evidence actually contains technique instructions.
  - Otherwise state that the loaded evidence is insufficient; do not substitute spacer recommendations.

### Definition of done

For every generated clinical statement, the system must be able to prove:

**user intent → relevant retrieved evidence → generated claim → exact supporting citation**

A retrieved chunk existing is not proof of grounding.

Implement the fixes, add/update automated tests for all regression cases, run the full test suite, and report:
1. root cause found,
2. files changed,
3. logic introduced,
4. test results,
5. any remaining limitations.

Do not hide failures by hardcoding these six questions; fix the underlying general behavior.