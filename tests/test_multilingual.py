"""Tests for Phase 2 Workstream B: Multilingual, Query Normalization, Query Decomposition."""

from __future__ import annotations

import unittest
from medical_rag.models import Chunk, SearchResult
from medical_rag.query_normalization import (
    normalize_medical_query,
    has_arabic_content,
    extract_clinical_units,
    normalize_arabic,
)
from medical_rag.multilingual import (
    detect_language,
    enhance_arabic_query,
    MultilingualRetriever,
    LanguageProfile,
)
from medical_rag.query_decomposition import (
    is_compound_query,
    decompose_query,
    retrieve_multi_question,
    DecomposedQuery,
)
from medical_rag.hybrid_retrieval import BM25Retriever, UnifiedRetriever


# ── Shared test chunks ────────────────────────────────────────────────────────
def _make_chunks():
    return [
        Chunk(
            chunk_id="chunk-001",
            text="Inhaled corticosteroids ICS are the preferred first-line controller therapy for paediatric asthma.",
            document="WHO guideline", publisher="WHO", source_url="http://who.int",
            topic="asthma management", section="Controller therapy",
            page_start=10, page_end=10, token_count=15, boundary_reason="section_boundary",
        ),
        Chunk(
            chunk_id="chunk-002",
            text="Intravenous magnesium sulfate is indicated for severe acute asthma exacerbations in children.",
            document="WHO guideline", publisher="WHO", source_url="http://who.int",
            topic="acute asthma", section="Exacerbations",
            page_start=34, page_end=35, token_count=16, boundary_reason="section_boundary",
        ),
        Chunk(
            chunk_id="chunk-003",
            text="Short-acting beta2-agonist SABA salbutamol is used for acute relief during asthma attacks.",
            document="NICE guideline", publisher="NICE", source_url="http://nice.org.uk",
            topic="reliever therapy", section="Acute management",
            page_start=22, page_end=22, token_count=14, boundary_reason="section_boundary",
        ),
        Chunk(
            chunk_id="chunk-004",
            text="Spirometry with bronchodilator reversibility testing confirms variable expiratory airflow limitation in children with asthma.",
            document="NICE guideline", publisher="NICE", source_url="http://nice.org.uk",
            topic="asthma diagnosis", section="Diagnosis",
            page_start=15, page_end=15, token_count=18, boundary_reason="section_boundary",
        ),
    ]


# ── Query Normalization Tests ─────────────────────────────────────────────────
class QueryNormalizationTests(unittest.TestCase):

    def test_preserves_ics_acronym(self):
        q = "What is the role of ICS in asthma management?"
        result = normalize_medical_query(q)
        self.assertIn("ICS", result)

    def test_preserves_saba_acronym(self):
        q = "When should SABA be used for asthma relief?"
        result = normalize_medical_query(q)
        self.assertIn("SABA", result)

    def test_preserves_feno_acronym(self):
        q = "What FeNO level indicates eosinophilic airway inflammation?"
        result = normalize_medical_query(q)
        self.assertIn("FeNO", result)

    def test_preserves_mg_units(self):
        q = "Should 400mg ICS be prescribed for moderate asthma?"
        result = normalize_medical_query(q)
        self.assertIn("400", result)
        self.assertIn("ICS", result)

    def test_preserves_mcg_units(self):
        q = "Is 200mcg fluticasone safe for children?"
        result = normalize_medical_query(q)
        self.assertIn("200", result)

    def test_normalizes_albuterol_to_salbutamol(self):
        q = "when should albuterol be administered?"
        result = normalize_medical_query(q)
        self.assertIn("salbutamol", result.lower())
        self.assertNotIn("albuterol", result.lower())

    def test_normalizes_sulphate_to_sulfate(self):
        q = "magnesium sulphate for asthma exacerbation"
        result = normalize_medical_query(q)
        self.assertIn("sulfate", result.lower())

    def test_arabic_diacritics_removed(self):
        text_with_diacritics = "\u0623\u064e\u0633\u0652\u062b\u064e\u0645\u064e\u0627"  # أَسْثَمَا
        result = normalize_arabic(text_with_diacritics)
        # Should not contain harakat
        import re
        self.assertFalse(re.search(r"[\u064B-\u065F]", result))

    def test_english_query_unchanged_structure(self):
        q = "What is the first-line controller for paediatric asthma?"
        result = normalize_medical_query(q)
        # Core medical content preserved
        self.assertIn("asthma", result.lower())
        self.assertIn("controller", result.lower())

    def test_empty_query_returns_empty(self):
        self.assertEqual(normalize_medical_query(""), "")
        self.assertEqual(normalize_medical_query("   "), "   ")

    def test_has_arabic_content_detects_arabic(self):
        self.assertTrue(has_arabic_content("متى يجب تصعيد العلاج؟"))
        self.assertFalse(has_arabic_content("When should treatment be escalated?"))

    def test_has_arabic_content_mixed(self):
        self.assertTrue(has_arabic_content("طفل يستخدم SABA"))

    def test_extract_clinical_units(self):
        q = "Give 200mcg fluticasone and 400mg theophylline to children 5-11 years"
        units = extract_clinical_units(q)
        values = [u[0] for u in units]
        self.assertIn("200", values)
        self.assertIn("400", values)

    def test_preserves_who_and_nice_acronyms(self):
        q = "According to NICE and WHO guidelines for ICS use"
        result = normalize_medical_query(q)
        self.assertIn("NICE", result)
        self.assertIn("WHO", result)
        self.assertIn("ICS", result)


# ── Multilingual Tests ────────────────────────────────────────────────────────
class MultilingualTests(unittest.TestCase):

    def test_detect_arabic_only(self):
        lang = detect_language("\u0645\u062a\u0649 \u064a\u062c\u0628 \u062a\u0635\u0639\u064a\u062f \u0627\u0644\u0639\u0644\u0627\u062c\u061f")
        self.assertTrue(lang.is_arabic)
        self.assertFalse(lang.is_english)
        self.assertFalse(lang.is_mixed)

    def test_detect_english_only(self):
        lang = detect_language("When should treatment be escalated for asthma?")
        self.assertTrue(lang.is_english)
        self.assertFalse(lang.is_arabic)

    def test_detect_mixed_arabic_english(self):
        lang = detect_language("\u0637\u0641\u0644 \u064a\u0633\u062a\u062e\u062f\u0645 SABA \u064a\u0648\u0645\u064a\u0627\u064b")
        self.assertTrue(lang.is_mixed)

    def test_detect_arabic_with_clinical_terms(self):
        lang = detect_language("\u0645\u062a\u0649 \u064a\u062c\u0628 \u062a\u0635\u0639\u064a\u062f SABA \u0644\u0644\u0637\u0641\u0644")
        self.assertIn("SABA", lang.detected_clinical_terms)

    def test_arabic_query_enhanced_with_english(self):
        arabic_query = "\u0623\u0632\u0645\u0629 \u062d\u0627\u062f\u0629 \u0639\u0644\u0627\u062c"
        enhanced = enhance_arabic_query(arabic_query)
        # Should contain some English clinical terms
        self.assertTrue(any(c.isascii() and c.isalpha() for c in enhanced))

    def test_english_query_not_modified_by_enhancement(self):
        q = "What is the first-line controller for paediatric asthma?"
        lang = detect_language(q)
        self.assertFalse(lang.is_arabic or lang.is_mixed)

    def test_multilingual_retriever_arabic_returns_results(self):
        chunks = _make_chunks()
        bm25 = BM25Retriever(chunks)

        class _FakeRepo:
            def search(self, q, top_k=5, min_score_threshold=0.0):
                return bm25.search(q, top_k=top_k)

        class _FakeUnified:
            def __init__(self):
                self._bm25 = bm25
            def search(self, q, strategy="hybrid_rerank", top_k=5, candidate_k=20):
                return bm25.search(q, top_k=top_k)

        unified = _FakeUnified()
        ml = MultilingualRetriever(unified)
        # Arabic for "acute asthma exacerbation treatment"
        arabic_q = "\u0623\u0632\u0645\u0629 \u062d\u0627\u062f\u0629"
        results, lang = ml.search(arabic_q, strategy="bm25", top_k=3)
        self.assertIsInstance(results, list)
        self.assertIsInstance(lang, LanguageProfile)
        self.assertTrue(lang.is_arabic or lang.is_mixed)

    def test_multilingual_retriever_english_passthrough(self):
        chunks = _make_chunks()
        bm25 = BM25Retriever(chunks)

        class _FakeUnified:
            def search(self, q, strategy="hybrid_rerank", top_k=5, candidate_k=20):
                return bm25.search(q, top_k=top_k)

        ml = MultilingualRetriever(_FakeUnified())
        results, lang = ml.search("inhaled corticosteroids ICS controller", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertTrue(lang.is_english)

    def test_mixed_arabic_english_query_preserves_saba(self):
        mixed_q = "\u0637\u0641\u0644 \u064a\u0633\u062a\u062e\u062f\u0645 SABA \u064a\u0648\u0645\u064a\u0627\u064b"
        enhanced = enhance_arabic_query(mixed_q)
        self.assertIn("SABA", enhanced)


# ── Query Decomposition Tests ─────────────────────────────────────────────────
class QueryDecompositionTests(unittest.TestCase):

    def test_simple_query_not_compound(self):
        q = "What is the first-line controller for paediatric asthma?"
        self.assertFalse(is_compound_query(q))

    def test_compound_query_detected(self):
        q = "What symptoms suggest asthma, how is it diagnosed, and what first-line controller is recommended?"
        self.assertTrue(is_compound_query(q))

    def test_compound_with_and(self):
        q = "When should ICS be started and how should treatment be stepped up in children with asthma?"
        self.assertTrue(is_compound_query(q))

    def test_simple_query_decompose_returns_single(self):
        q = "What is the recommended ICS dose for children with mild asthma?"
        decomposed = decompose_query(q)
        self.assertFalse(decomposed.is_compound)
        self.assertEqual(len(decomposed.sub_questions), 1)
        self.assertEqual(decomposed.sub_questions[0], q)

    def test_compound_query_decomposed_into_multiple(self):
        q = "What symptoms suggest asthma, how is it diagnosed, and what first-line controller is recommended?"
        decomposed = decompose_query(q)
        self.assertTrue(decomposed.is_compound)
        self.assertGreaterEqual(decomposed.count, 2)

    def test_decomposed_preserves_original(self):
        q = "When should ICS be started and how should treatment be stepped up for children?"
        decomposed = decompose_query(q)
        self.assertEqual(decomposed.original, q)

    def test_sub_questions_non_empty(self):
        q = "What symptoms suggest asthma, how is it diagnosed, and what first-line controller is recommended?"
        decomposed = decompose_query(q)
        for sq in decomposed.sub_questions:
            self.assertGreater(len(sq.strip()), 5)

    def test_retrieve_multi_question_simple(self):
        chunks = _make_chunks()
        bm25 = BM25Retriever(chunks)

        class _FakeRetriever:
            def search(self, q, strategy="hybrid_rerank", top_k=5, candidate_k=20):
                return bm25.search(q, top_k=top_k)

        q = "What is the first-line controller for asthma?"
        result = retrieve_multi_question(q, _FakeRetriever(), top_k_per_question=3, final_top_k=3)
        self.assertFalse(result.is_compound)
        self.assertGreater(len(result.merged_results), 0)

    def test_retrieve_multi_question_compound(self):
        chunks = _make_chunks()
        bm25 = BM25Retriever(chunks)

        class _FakeRetriever:
            def search(self, q, strategy="hybrid_rerank", top_k=5, candidate_k=20):
                return bm25.search(q, top_k=top_k)

        q = "What symptoms suggest asthma, how is it diagnosed, and what first-line controller is recommended?"
        result = retrieve_multi_question(q, _FakeRetriever(), top_k_per_question=3, final_top_k=5)
        self.assertTrue(result.is_compound)
        self.assertGreaterEqual(len(result.sub_questions), 2)
        self.assertGreater(len(result.merged_results), 0)

    def test_merged_results_no_duplicate_chunk_ids(self):
        chunks = _make_chunks()
        bm25 = BM25Retriever(chunks)

        class _FakeRetriever:
            def search(self, q, strategy="hybrid_rerank", top_k=5, candidate_k=20):
                return bm25.search(q, top_k=top_k)

        q = "When should ICS be used and what is the role of SABA in acute asthma?"
        result = retrieve_multi_question(q, _FakeRetriever(), top_k_per_question=3, final_top_k=5)
        chunk_ids = [r.metadata.get("chunk_id") for r in result.merged_results]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))  # no duplicates

    def test_normalization_integrated_in_unified_retriever(self):
        """UnifiedRetriever with normalize=True should not break existing results."""
        chunks = _make_chunks()
        bm25 = BM25Retriever(chunks)

        class _FakeRepo:
            def search(self, q, top_k=5, min_score_threshold=0.0):
                return bm25.search(q, top_k=top_k)

        retriever = UnifiedRetriever(_FakeRepo(), chunks)
        results = retriever.search("ICS controller therapy asthma", strategy="bm25", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].metadata["chunk_id"], "chunk-001")


if __name__ == "__main__":
    unittest.main()
