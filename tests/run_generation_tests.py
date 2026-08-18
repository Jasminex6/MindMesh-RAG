"""Standalone test runner for Day 3 generation module.

Runs without tiktoken/pymupdf by mocking the heavy __init__ imports.
Only tests generation.py logic which has no heavy native deps.
"""
import sys
import types
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Mock heavy modules that generation.py does NOT use
# but that __init__.py imports transitively via chunking/ingestion
for mod_name in ("tiktoken", "pymupdf", "fitz", "langchain_ollama", 
                 "langchain_core", "langchain_chroma", "chromadb"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Now we can import generation directly (it only needs models.py)
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
    _is_patient_specific,
)

import json
import unittest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(rank, score, chunk_id, document="WHO asthma.pdf",
                 section="Controller therapy", page_start=25, page_end=25,
                 text="Inhaled corticosteroids (ICS) are the preferred controller."):
    return SearchResult(
        rank=rank, score=score, text=text,
        metadata={
            "chunk_id": chunk_id, "document": document, "section": section,
            "page": page_start, "page_start": page_start, "page_end": page_end,
            "token_count": 50,
        },
    )


def _high_quality():
    return [
        _make_result(1, 0.92, "who-p25-001"),
        _make_result(2, 0.87, "who-p26-002", section="Step therapy"),
        _make_result(3, 0.81, "nice-p10-003", document="NICE Asthma.pdf",
                     section="Pharmacological", page_start=10),
    ]


def _low_quality():
    return [_make_result(1, 0.20, "who-p42-099", section="References",
                         text="See bibliography.")]


def _medium_quality():
    return [_make_result(1, 0.38, "who-p27-005", section="Exacerbation management",
                         text="IV magnesium sulfate may be considered for severe exacerbations.")]


# ===========================================================================
class TestConfidence(unittest.TestCase):
    def test_high(self):
        self.assertEqual(assess_confidence(_high_quality()), "High")

    def test_medium(self):
        self.assertEqual(assess_confidence(_medium_quality()), "Medium")

    def test_low(self):
        self.assertEqual(assess_confidence([_make_result(1, 0.28, "low")]), "Low")

    def test_insufficient_empty(self):
        self.assertEqual(assess_confidence([]), "Insufficient Evidence")

    def test_insufficient_very_low(self):
        self.assertEqual(assess_confidence([_make_result(1, 0.20, "g")]), "Insufficient Evidence")


class TestRefusal(unittest.TestCase):
    def test_empty_results(self):
        r, _ = check_refusal("query?", [], "Insufficient Evidence")
        self.assertTrue(r)

    def test_low_score(self):
        r, msg = check_refusal("diabetes?", _low_quality(), "Insufficient Evidence")
        self.assertTrue(r)
        self.assertIn("below the minimum threshold", msg)

    def test_patient_specific(self):
        r, msg = check_refusal("Should I give my child 200mg?", _high_quality(), "High")
        self.assertTrue(r)
        self.assertIn("healthcare professional", msg)

    def test_patient_prescribe(self):
        r, _ = check_refusal("What medication should my patient take?", _high_quality(), "High")
        self.assertTrue(r)

    def test_allows_supported(self):
        r, _ = check_refusal("What is first-line controller?", _high_quality(), "High")
        self.assertFalse(r)

    def test_out_of_scope_low_score(self):
        r, _ = check_refusal("Type 2 diabetes treatment?", _low_quality(), "Insufficient Evidence")
        self.assertTrue(r)


class TestPatientSpecific(unittest.TestCase):
    def test_my_child(self):
        self.assertTrue(_is_patient_specific("Should I give my child steroids?"))

    def test_prescribe(self):
        self.assertTrue(_is_patient_specific("Prescribe for my patient"))

    def test_general_ok(self):
        self.assertFalse(_is_patient_specific("What is the first-line treatment?"))


class TestCitationVerification(unittest.TestCase):
    def test_verified(self):
        cits = [
            Citation("ICS first-line", "who-p25-001", "", "", "", 0.0),
            Citation("Step therapy", "who-p26-002", "", "", "", 0.0),
        ]
        v = verify_citations(cits, _high_quality())
        self.assertTrue(all(c.verified for c in v))
        self.assertEqual(v[0].document, "WHO asthma.pdf")
        self.assertAlmostEqual(v[0].score, 0.92, places=2)

    def test_hallucinated(self):
        cits = [Citation("Fake", "FAKE-999", "", "", "", 0.0)]
        v = verify_citations(cits, _high_quality())
        self.assertFalse(v[0].verified)

    def test_mixed(self):
        cits = [
            Citation("Real", "who-p25-001", "", "", "", 0.0),
            Citation("Fake", "nonexistent", "", "", "", 0.0),
        ]
        v = verify_citations(cits, _high_quality())
        self.assertTrue(v[0].verified)
        self.assertFalse(v[1].verified)


class TestParsing(unittest.TestCase):
    def test_clean_json(self):
        p = parse_llm_response('{"recommendation":"Use ICS","supporting_evidence":"","citations":[],"safety_note":""}')
        self.assertEqual(p["recommendation"], "Use ICS")

    def test_fenced(self):
        p = parse_llm_response('```json\n{"recommendation":"Use ICS","supporting_evidence":"","citations":[],"safety_note":""}\n```')
        self.assertEqual(p["recommendation"], "Use ICS")

    def test_surrounded(self):
        p = parse_llm_response('Here:\n{"recommendation":"Use ICS","supporting_evidence":"","citations":[],"safety_note":""}\nDone.')
        self.assertEqual(p["recommendation"], "Use ICS")

    def test_garbage(self):
        p = parse_llm_response("Cannot parse this.")
        self.assertIn("Cannot parse", p["recommendation"])


class TestSafetyCheck(unittest.TestCase):
    def test_no_verified_downgrades(self):
        a = GeneratedAnswer("q", "Use ICS 200mg", "", [Citation("c", "fake", "", "", "", 0.0, False)],
                           "High", "")
        c = post_generation_safety_check(a)
        self.assertEqual(c.confidence, "Low")
        self.assertIn("WARNING", c.safety_note)

    def test_verified_keeps_confidence(self):
        a = GeneratedAnswer("q", "Use ICS", "", [Citation("c", "id", "WHO", "Sec", "25", 0.9, True)],
                           "High", "Consult.")
        c = post_generation_safety_check(a)
        self.assertEqual(c.confidence, "High")

    def test_dosage_no_citation_warns(self):
        a = GeneratedAnswer("q", "Give 200 mg daily", "",
                           [Citation("c", "fake", "", "", "", 0.0, False)], "High", "")
        c = post_generation_safety_check(a)
        self.assertIn("CAUTION", c.safety_note)


class TestEvidenceBlock(unittest.TestCase):
    def test_contains_ids(self):
        b = build_evidence_block(_high_quality())
        self.assertIn("who-p25-001", b)
        self.assertIn("nice-p10-003", b)

    def test_contains_scores(self):
        b = build_evidence_block(_high_quality())
        self.assertIn("0.9200", b)


class TestServiceRefusal(unittest.TestCase):
    def setUp(self):
        self.svc = GenerationService()

    def test_empty_results(self):
        a = self.svc.generate("What is asthma treatment?", [])
        self.assertTrue(a.refused)

    def test_out_of_scope(self):
        a = self.svc.generate("Diabetes?", _low_quality())
        self.assertTrue(a.refused)

    def test_patient_specific(self):
        a = self.svc.generate("What dose should I give my child?", _high_quality())
        self.assertTrue(a.refused)
        self.assertIn("healthcare professional", a.refusal_reason)

    def test_skip_llm(self):
        a = self.svc.generate("First-line controller?", _high_quality(), skip_llm=True)
        self.assertFalse(a.refused)
        self.assertEqual(a.confidence, "High")


class TestServiceWithMock(unittest.TestCase):
    def _mock_response(self):
        return json.dumps({
            "recommendation": "Low-dose inhaled corticosteroids (ICS) are recommended.",
            "supporting_evidence": "WHO guideline recommends ICS.",
            "citations": [
                {"claim": "ICS is first-line", "chunk_id": "who-p25-001"},
                {"claim": "Step therapy", "chunk_id": "who-p26-002"},
            ],
            "safety_note": "Consult a healthcare professional.",
        })

    @patch("medical_rag.generation.call_llm")
    def test_full_pipeline(self, mock):
        mock.return_value = self._mock_response()
        a = GenerationService().generate("First-line controller?", _high_quality())
        self.assertFalse(a.refused)
        self.assertEqual(a.confidence, "High")
        self.assertEqual(len(a.citations), 2)
        self.assertTrue(all(c.verified for c in a.citations))

    @patch("medical_rag.generation.call_llm")
    def test_hallucinated_citation(self, mock):
        mock.return_value = json.dumps({
            "recommendation": "Use biologic.", "supporting_evidence": "Based on.",
            "citations": [{"claim": "Biologic", "chunk_id": "HALLUCINATED-999"}],
            "safety_note": "",
        })
        a = GenerationService().generate("Biologics?", _high_quality())
        self.assertFalse(a.citations[0].verified)
        self.assertIn("WARNING", a.safety_note)

    @patch("medical_rag.generation.call_llm")
    def test_medium_confidence(self, mock):
        mock.return_value = json.dumps({
            "recommendation": "IV magnesium may be considered.",
            "supporting_evidence": "Exacerbation section.",
            "citations": [{"claim": "IV mag", "chunk_id": "who-p27-005"}],
            "safety_note": "Specialist decision.",
        })
        a = GenerationService().generate("When IV magnesium?", _medium_quality())
        self.assertFalse(a.refused)
        self.assertEqual(a.confidence, "Medium")
        self.assertTrue(a.citations[0].verified)

    @patch("medical_rag.generation.call_llm")
    def test_serializable(self, mock):
        mock.return_value = self._mock_response()
        a = GenerationService().generate("First-line?", _high_quality())
        s = json.dumps(a.to_dict())
        self.assertIsInstance(s, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
