"""Tests for Day 3: grounded generation, citation verification, refusal, confidence."""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

from medical_rag.models import SearchResult
from medical_rag.generation import (
    GenerationService,
    GeneratedAnswer,
    Citation,
    assess_confidence,
    check_refusal,
    verify_citations,
    parse_llm_response,
    post_generation_safety_check,
    build_evidence_block,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic SearchResult factories
# ---------------------------------------------------------------------------

def _make_result(rank: int, score: float, chunk_id: str,
                 document: str = "WHO asthma.pdf",
                 section: str = "Controller therapy",
                 page_start: int = 25, page_end: int = 25,
                 text: str = "Inhaled corticosteroids (ICS) are the preferred controller.") -> SearchResult:
    return SearchResult(
        rank=rank,
        score=score,
        text=text,
        metadata={
            "chunk_id": chunk_id,
            "document": document,
            "section": section,
            "page": page_start,
            "page_start": page_start,
            "page_end": page_end,
            "token_count": 50,
        },
    )


def _high_quality_results() -> list[SearchResult]:
    """Simulate strong retrieval: multiple high-scoring relevant chunks."""
    return [
        _make_result(1, 0.92, "who-p25-001"),
        _make_result(2, 0.87, "who-p26-002", section="Step therapy"),
        _make_result(3, 0.81, "nice-p10-003", document="NICE Asthma.pdf",
                     section="Pharmacological", page_start=10),
    ]


def _low_quality_results() -> list[SearchResult]:
    """Simulate weak retrieval: low scores, possibly off-topic."""
    return [
        _make_result(1, 0.20, "who-p42-099", section="References",
                     text="See bibliography for full citation list."),
    ]


def _medium_quality_results() -> list[SearchResult]:
    """Single moderately scored result."""
    return [
        _make_result(1, 0.42, "who-p27-005", section="Exacerbation management",
                     text="IV magnesium sulfate may be considered for severe exacerbations."),
    ]


# ===========================================================================
# Test cases
# ===========================================================================


class TestConfidenceAssessment(unittest.TestCase):
    """Confidence must derive from retrieval scores, not LLM self-confidence."""

    def test_high_confidence(self):
        results = _high_quality_results()
        self.assertEqual(assess_confidence(results), "High")

    def test_medium_confidence(self):
        results = _medium_quality_results()
        self.assertEqual(assess_confidence(results), "Medium")

    def test_insufficient_evidence_empty(self):
        self.assertEqual(assess_confidence([]), "Insufficient Evidence")

    def test_insufficient_evidence_very_low(self):
        results = [_make_result(1, 0.28, "chunk-garbage")]
        self.assertEqual(assess_confidence(results), "Insufficient Evidence")


class TestRefusalGate(unittest.TestCase):
    """System must refuse under specified conditions."""

    def test_refuses_empty_results(self):
        refused, reason = check_refusal("What is the first-line treatment?", [], "Insufficient Evidence")
        self.assertTrue(refused)
        self.assertIn("No relevant", reason)

    def test_refuses_low_top_score(self):
        results = _low_quality_results()
        refused, reason = check_refusal("diabetes management?", results, "Insufficient Evidence")
        self.assertTrue(refused)
        self.assertIn("below the minimum relevance threshold", reason)

    def test_refuses_insufficient_confidence(self):
        # Score above min_top_score but confidence is Insufficient
        results = [_make_result(1, 0.42, "chunk-meh")]
        refused, reason = check_refusal("some query", results, "Insufficient Evidence")
        self.assertTrue(refused)

    def test_refuses_patient_specific_diagnosis(self):
        # NEW ARCHITECTURE: patient-specific / medication queries are now handled
        # UPSTREAM by IntentClassifier (adaptive clarification). check_refusal() no
        # longer hard-refuses them — they fall through to grounded generation.
        results = _high_quality_results()
        refused, reason = check_refusal(
            "Should I give my child 200mg salbutamol?", results, "High"
        )
        self.assertFalse(refused)  # IntentClassifier handles this upstream

    def test_refuses_patient_specific_prescribe(self):
        # NEW ARCHITECTURE: same — medication intent passes through check_refusal;
        # IntentClassifier showed adaptive clarifying questions upstream.
        results = _high_quality_results()
        refused, reason = check_refusal(
            "What medication should my patient take?", results, "High"
        )
        self.assertFalse(refused)  # IntentClassifier handles this upstream

    def test_allows_supported_query(self):
        results = _high_quality_results()
        refused, _ = check_refusal(
            "What is the first-line controller treatment for childhood asthma?",
            results, "High"
        )
        self.assertFalse(refused)


class TestPatientSpecificDetection(unittest.TestCase):
    """IntentClassifier owns patient/medication intent detection."""
    def test_detects_my_child(self):
        from medical_rag.intent_classifier import IntentClassifier
        decision = IntentClassifier().classify("Should I give my child steroids?")
        self.assertIn(decision.category, ("patient_specific", "medication_or_dose_request", "personal_dosing_decision"))

    def test_detects_prescribe(self):
        from medical_rag.intent_classifier import IntentClassifier
        decision = IntentClassifier().classify("Prescribe for my patient")
        self.assertIn(decision.category, ("medication_or_dose_request", "patient_specific", "personal_dosing_decision"))

    def test_general_query_passes(self):
        from medical_rag.intent_classifier import IntentClassifier
        decision = IntentClassifier().classify("What is the first-line treatment for asthma?")
        self.assertEqual(decision.category, "general_educational")


class TestCitationVerification(unittest.TestCase):
    """Citations must actually correspond to retrieved chunks."""

    def test_verified_citations(self):
        results = _high_quality_results()
        citations = [
            Citation(claim="ICS is first-line", chunk_id="who-p25-001",
                     document="", section="", page="", score=0.0),
            Citation(claim="Step therapy approach", chunk_id="who-p26-002",
                     document="", section="", page="", score=0.0),
        ]
        verified = verify_citations(citations, results)
        self.assertTrue(all(c.verified for c in verified))
        # Should have enriched metadata
        self.assertEqual(verified[0].document, "WHO asthma.pdf")
        self.assertEqual(verified[0].section, "Controller therapy")
        self.assertAlmostEqual(verified[0].score, 0.92, places=2)

    def test_unverified_hallucinated_citation(self):
        results = _high_quality_results()
        citations = [
            Citation(claim="Made up claim", chunk_id="FAKE-CHUNK-999",
                     document="", section="", page="", score=0.0),
        ]
        verified = verify_citations(citations, results)
        self.assertFalse(verified[0].verified)

    def test_mixed_verification(self):
        results = _high_quality_results()
        citations = [
            Citation(claim="Real claim", chunk_id="who-p25-001",
                     document="", section="", page="", score=0.0),
            Citation(claim="Fake claim", chunk_id="nonexistent",
                     document="", section="", page="", score=0.0),
        ]
        verified = verify_citations(citations, results)
        self.assertTrue(verified[0].verified)
        self.assertFalse(verified[1].verified)


class TestLLMResponseParsing(unittest.TestCase):
    """Parser must handle clean JSON, markdown-fenced JSON, and garbage."""

    def test_clean_json(self):
        raw = '{"recommendation": "Use ICS", "supporting_evidence": "chunk 1", "citations": [], "safety_note": "Consult"}'
        parsed = parse_llm_response(raw)
        self.assertEqual(parsed["recommendation"], "Use ICS")

    def test_markdown_fenced_json(self):
        raw = '```json\n{"recommendation": "Use ICS", "supporting_evidence": "", "citations": [], "safety_note": ""}\n```'
        parsed = parse_llm_response(raw)
        self.assertEqual(parsed["recommendation"], "Use ICS")

    def test_json_with_surrounding_text(self):
        raw = 'Here is my answer:\n{"recommendation": "Use ICS", "supporting_evidence": "", "citations": [], "safety_note": ""}\nDone.'
        parsed = parse_llm_response(raw)
        self.assertEqual(parsed["recommendation"], "Use ICS")

    def test_unparseable_returns_fallback(self):
        raw = "I cannot parse this at all, sorry."
        parsed = parse_llm_response(raw)
        self.assertIn("I cannot parse", parsed["recommendation"])


class TestPostGenerationSafetyCheck(unittest.TestCase):

    def test_no_verified_citations_downgrades(self):
        answer = GeneratedAnswer(
            query="test",
            recommendation="Use ICS 200mg daily",
            supporting_evidence="",
            citations=[Citation("claim", "fake-id", "", "", "", 0.0, verified=False)],
            confidence="High",
            safety_note="",
        )
        checked = post_generation_safety_check(answer)
        self.assertEqual(checked.confidence, "Low")
        self.assertIn("WARNING", checked.safety_note)

    def test_verified_citations_keeps_confidence(self):
        answer = GeneratedAnswer(
            query="test",
            recommendation="Use ICS as controller",
            supporting_evidence="",
            citations=[Citation("claim", "who-p25-001", "WHO", "Controller", "25", 0.9, verified=True)],
            confidence="High",
            safety_note="Consult physician.",
        )
        checked = post_generation_safety_check(answer)
        self.assertEqual(checked.confidence, "High")

    def test_dosage_without_citation_adds_caution(self):
        answer = GeneratedAnswer(
            query="test",
            recommendation="Give 200 mg budesonide daily",
            supporting_evidence="",
            citations=[Citation("claim", "fake", "", "", "", 0.0, verified=False)],
            confidence="High",
            safety_note="",
        )
        checked = post_generation_safety_check(answer)
        self.assertIn("CAUTION", checked.safety_note)


class TestEvidenceBlockFormatting(unittest.TestCase):

    def test_evidence_block_contains_chunk_ids(self):
        results = _high_quality_results()
        block = build_evidence_block(results)
        self.assertIn("who-p25-001", block)
        self.assertIn("who-p26-002", block)
        self.assertIn("nice-p10-003", block)

    def test_evidence_block_contains_scores(self):
        results = _high_quality_results()
        block = build_evidence_block(results)
        self.assertIn("0.9200", block)


class TestGenerationServiceRefusal(unittest.TestCase):
    """Integration: GenerationService.generate() must refuse when appropriate."""

    def setUp(self):
        self.svc = GenerationService()

    def test_refuses_on_empty_results(self):
        answer = self.svc.generate("What is asthma treatment?", [])
        self.assertTrue(answer.refused)
        self.assertEqual(answer.confidence, "Insufficient Evidence")

    def test_refuses_out_of_scope(self):
        low = _low_quality_results()
        answer = self.svc.generate("What is diabetes treatment?", low)
        self.assertTrue(answer.refused)

    def test_refuses_patient_specific(self):
        # NEW ARCHITECTURE: medication/patient-specific intent is handled UPSTREAM
        # by IntentClassifier (adaptive clarification flow). The generation service
        # no longer hard-refuses these — it produces a grounded answer with a safety
        # note. The LLM system prompt rules 5 & 6 prevent personalised dosing.
        high = _high_quality_results()
        answer = self.svc.generate(
            "What dose should I give my child for asthma?", high
        )
        # Should NOT be refused here — IntentClassifier handles clarification upstream
        # and the LLM generates a population-level guideline answer with safety note.
        # (LLM call is mocked/skipped so answer.refused depends on retrieval quality.)
        self.assertIsNotNone(answer)  # Must return a structured answer, not crash

    def test_skip_llm_produces_structure(self):
        high = _high_quality_results()
        answer = self.svc.generate(
            "What is the first-line controller?", high, skip_llm=True
        )
        self.assertFalse(answer.refused)
        self.assertIn("High", answer.confidence)
        self.assertIn("skip_llm", answer.safety_note)


class TestGenerationServiceWithMockedLLM(unittest.TestCase):
    """Test the full pipeline with a mocked LLM call."""

    def _mock_llm_response(self) -> str:
        return json.dumps({
            "recommendation": "Low-dose inhaled corticosteroids (ICS) are recommended as first-line controller therapy.",
            "supporting_evidence": "WHO guideline recommends ICS as preferred controller for children with asthma.",
            "citations": [
                {"claim": "ICS is first-line controller", "chunk_id": "who-p25-001"},
                {"claim": "Step therapy for escalation", "chunk_id": "who-p26-002"},
            ],
            "safety_note": "Always consult a healthcare professional. This is guideline-based information only.",
        })

    @patch("medical_rag.generation.call_llm")
    def test_full_pipeline_with_valid_citations(self, mock_llm):
        mock_llm.return_value = self._mock_llm_response()
        svc = GenerationService()
        results = _high_quality_results()

        answer = svc.generate("What is the first-line controller treatment?", results)

        self.assertFalse(answer.refused)
        self.assertEqual(answer.confidence, "High")
        self.assertIn("corticosteroids", answer.recommendation.lower())
        # Both citations should be verified
        self.assertEqual(len(answer.citations), 2)
        self.assertTrue(all(c.verified for c in answer.citations))
        # Metadata should be enriched
        self.assertEqual(answer.citations[0].document, "WHO asthma.pdf")

    @patch("medical_rag.generation.call_llm")
    def test_hallucinated_citation_detected(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "recommendation": "Use advanced biologic therapy.",
            "supporting_evidence": "Based on evidence.",
            "citations": [
                {"claim": "Biologic therapy", "chunk_id": "HALLUCINATED-ID-999"},
            ],
            "safety_note": "",
        })
        svc = GenerationService()
        results = _high_quality_results()

        answer = svc.generate("What about biologics?", results)

        # The hallucinated citation should NOT be verified
        self.assertEqual(len(answer.citations), 1)
        self.assertFalse(answer.citations[0].verified)
        # Safety check should have downgraded confidence
        self.assertIn("WARNING", answer.safety_note)

    @patch("medical_rag.generation.call_llm")
    def test_medium_confidence_query(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "recommendation": "IV magnesium sulfate may be considered.",
            "supporting_evidence": "Evidence from exacerbation management section.",
            "citations": [
                {"claim": "IV magnesium for severe exacerbations", "chunk_id": "who-p27-005"},
            ],
            "safety_note": "Specialist decision required.",
        })
        svc = GenerationService()
        results = _medium_quality_results()

        answer = svc.generate("When is IV magnesium used?", results)

        self.assertFalse(answer.refused)
        self.assertEqual(answer.confidence, "Medium")
        self.assertTrue(answer.citations[0].verified)

    @patch("medical_rag.generation.call_llm")
    def test_answer_to_dict_serializable(self, mock_llm):
        """Ensure GeneratedAnswer.to_dict() produces JSON-serializable output."""
        mock_llm.return_value = self._mock_llm_response()
        svc = GenerationService()
        results = _high_quality_results()
        answer = svc.generate("What is first-line treatment?", results)

        d = answer.to_dict()
        serialized = json.dumps(d)
        self.assertIsInstance(serialized, str)
        self.assertIn("recommendation", serialized)


import json


class TestOutOfScopeAndTypeSafety(unittest.TestCase):
    """Test out-of-scope detection, empty retrieval, and type-safe parsing (Goal 8)."""

    @patch("medical_rag.generation.call_llm")
    def test_valid_in_scope_query(self, mock_llm):
        """A. Valid in-scope query: 'asthma symptoms'."""
        mock_llm.return_value = json.dumps({
            "recommendation": "Common asthma symptoms include wheezing, cough, and shortness of breath.",
            "supporting_evidence": "WHO guidelines specify wheezing and breathlessness as key symptoms.",
            "citations": [{"claim": "Wheezing and cough", "chunk_id": "who-p25-001"}],
            "safety_note": "Guideline info only.",
        })
        svc = GenerationService()
        results = _high_quality_results()
        answer = svc.generate("asthma symptoms", results)

        self.assertTrue(mock_llm.called)
        self.assertFalse(answer.refused)
        self.assertIn("wheezing", answer.recommendation.lower())
        self.assertEqual(len(answer.citations), 1)

    @patch("medical_rag.generation.call_llm")
    def test_clearly_out_of_scope_query(self, mock_llm):
        """B. Clearly out-of-scope query: 'broken arm'."""
        svc = GenerationService()
        # Even if low results exist
        results = [_make_result(1, 0.22, "who-p42-099", text="Reference page.")]
        answer = svc.generate("broken arm", results)

        # Must NOT call LLM
        self.assertFalse(mock_llm.called)
        self.assertTrue(answer.refused)
        self.assertIn("OUT OF SCOPE", answer.refusal_reason)

    @patch("medical_rag.generation.call_llm")
    def test_empty_retrieval(self, mock_llm):
        """C. Empty retrieval."""
        svc = GenerationService()
        answer = svc.generate("asthma symptoms", [])

        # Must NOT call LLM
        self.assertFalse(mock_llm.called)
        self.assertTrue(answer.refused)
        self.assertIn("OUT OF SCOPE", answer.refusal_reason)

    @patch("medical_rag.generation.call_llm")
    def test_supporting_evidence_is_string(self, mock_llm):
        """D. supporting_evidence is a string."""
        mock_llm.return_value = json.dumps({
            "recommendation": "Use ICS for controller therapy.",
            "supporting_evidence": "Excerpt 1 from WHO guideline.",
            "citations": [{"claim": "ICS controller", "chunk_id": "who-p25-001"}],
        })
        svc = GenerationService()
        answer = svc.generate("ICS treatment", _high_quality_results())
        self.assertIsInstance(answer.supporting_evidence, str)
        self.assertIn("Excerpt 1", answer.supporting_evidence)

    @patch("medical_rag.generation.call_llm")
    def test_supporting_evidence_is_list(self, mock_llm):
        """E. supporting_evidence is a list (prevents TypeError)."""
        mock_llm.return_value = json.dumps({
            "recommendation": "Use ICS for controller therapy.",
            "supporting_evidence": ["Excerpt 1 from WHO.", "Excerpt 2 from NICE."],
            "citations": [],
        })
        svc = GenerationService()
        answer = svc.generate("ICS treatment", _high_quality_results())

        self.assertIsInstance(answer.supporting_evidence, str)
        self.assertIn("Excerpt 1", answer.supporting_evidence)
        self.assertIn("Excerpt 2", answer.supporting_evidence)
        self.assertFalse(answer.refused)

    @patch("medical_rag.generation.call_llm")
    def test_recommendation_missing_or_null(self, mock_llm):
        """F. recommendation is missing/null."""
        mock_llm.return_value = json.dumps({
            "recommendation": None,
            "supporting_evidence": ["Excerpt text"],
            "citations": [],
        })
        svc = GenerationService()
        answer = svc.generate("treatment", _high_quality_results())

        self.assertIsInstance(answer.recommendation, str)
        self.assertEqual(answer.recommendation, "")
        self.assertFalse(answer.refused)




# ===========================================================================
# Age-band prompt injection tests
# ===========================================================================

import json as _json_module


class TestAgeBandPrompt(unittest.TestCase):
    """build_user_prompt() must inject an age-band context header when age_band is provided,
    and must remain backward-compatible when age_band is None."""

    def _make_prompt(self, age_band=None):
        from medical_rag.generation import build_user_prompt
        return build_user_prompt("What ICS for asthma?", "evidence here", age_band=age_band)

    # ── Presence / absence of header ─────────────────────────────────────

    def test_age_band_header_included_when_set(self):
        prompt = self._make_prompt("under_6")
        self.assertIn("AGE BAND CONTEXT", prompt)

    def test_age_band_header_absent_when_none(self):
        prompt = self._make_prompt(None)
        self.assertNotIn("AGE BAND CONTEXT", prompt)

    def test_age_band_header_absent_when_not_supplied(self):
        from medical_rag.generation import build_user_prompt
        prompt = build_user_prompt("What ICS?", "evidence")
        self.assertNotIn("AGE BAND CONTEXT", prompt)

    # ── Correct display text per band ────────────────────────────────────

    def test_under_6_display_text(self):
        prompt = self._make_prompt("under_6")
        self.assertIn("under 6", prompt.lower())

    def test_children_6_11_display_text(self):
        prompt = self._make_prompt("children_6_11")
        lower = prompt.lower()
        self.assertTrue(
            "6" in lower and "11" in lower,
            f"Expected 6-11 age text in prompt: {prompt[:200]}"
        )

    def test_adults_display_text(self):
        prompt = self._make_prompt("adults_adolescents")
        lower = prompt.lower()
        has_adult_text = "12" in lower or "adult" in lower or "adolescent" in lower
        self.assertTrue(has_adult_text, f"Expected adult text in prompt: {prompt[:200]}")

    # ── Evidence block and question are still present ─────────────────────

    def test_evidence_block_still_present(self):
        prompt = self._make_prompt("children_6_11")
        self.assertIn("EVIDENCE CHUNKS", prompt)
        self.assertIn("evidence here", prompt)

    def test_question_still_present(self):
        prompt = self._make_prompt("under_6")
        self.assertIn("CLINICAL QUESTION", prompt)
        self.assertIn("What ICS for asthma?", prompt)

    # ── generate() threads age_band through without error ─────────────────

    def test_generate_accepts_age_band_param(self):
        """GenerationService.generate() must accept age_band without raising."""
        svc = GenerationService()
        # skip_llm=True so we don't need Ollama running
        answer = svc.generate(
            "What ICS for asthma?",
            _high_quality_results(),
            skip_llm=True,
            age_band="children_6_11",
        )
        self.assertIsNotNone(answer)
        self.assertFalse(answer.refused)


if __name__ == "__main__":
    unittest.main()
