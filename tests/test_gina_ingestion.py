"""Unit tests for GINA 2026 PDF ingestion, chunking, age metadata, and age-aware retrieval."""

from __future__ import annotations

import unittest
from medical_rag.models import Chunk, SourceSpec
from medical_rag.chunking import _derive_age_and_population, is_section_heading
from medical_rag.hybrid_retrieval import parse_query_age, BM25Retriever, UnifiedRetriever
from medical_rag.ingestion import COPYRIGHT_BOILERPLATE_RE


class TestGinaIngestionAndChunking(unittest.TestCase):

    def test_copyright_boilerplate_regex(self):
        sample = "COPYRIGHTED MATERIAL - DO NOT COPY OR DISTRIBUTE"
        self.assertTrue(bool(COPYRIGHT_BOILERPLATE_RE.search(sample)))

        sample_gina = "GINA 2026 SUMMARY GUIDE FOR ASTHMA MANAGEMENT"
        self.assertTrue(bool(COPYRIGHT_BOILERPLATE_RE.search(sample_gina)))

    def test_gina_table_and_step_headings(self):
        self.assertTrue(is_section_heading("Table 3: Suggested initial treatment for adults and adolescents"))
        self.assertTrue(is_section_heading("Table 4: Suggested initial treatment for children"))
        self.assertTrue(is_section_heading("Step 1: Low dose controller"))
        self.assertTrue(is_section_heading("Step 2: Maintenance ICS"))
        self.assertTrue(is_section_heading("Managing asthma in specific populations"))
        self.assertTrue(is_section_heading("Pregnancy"))
        self.assertTrue(is_section_heading("Elderly"))

    def test_derive_age_and_population_gina_tables(self):
        # Table 3 -> adults_adolescents
        age, covers_u6, pop = _derive_age_and_population(
            "GINA-Summary-Guide-2026-WEB-WMS",
            "Table 3: Suggested initial treatment for adults and adolescents",
            "Initial controller options for adults 12+ years"
        )
        self.assertEqual(age, "adults_adolescents")
        self.assertFalse(covers_u6)
        self.assertEqual(pop, "")

        # Table 4 -> children_6_11
        age4, covers_u6_4, pop4 = _derive_age_and_population(
            "GINA-Summary-Guide-2026-WEB-WMS",
            "Table 4: Suggested initial treatment for children",
            "Track options for children 6-11 years"
        )
        self.assertEqual(age4, "children_6_11")
        self.assertFalse(covers_u6_4)
        self.assertEqual(pop4, "")

        # Specific population: Pregnancy
        age_p, covers_u6_p, pop_p = _derive_age_and_population(
            "GINA-Summary-Guide-2026-WEB-WMS",
            "Managing asthma in specific populations > Pregnancy",
            "Asthma management during pregnancy and lactation"
        )
        self.assertEqual(age_p, "specific_population")
        self.assertFalse(covers_u6_p)
        self.assertEqual(pop_p, "pregnancy")

    def test_parse_query_age_detection(self):
        # Under 6
        age_label, is_u6 = parse_query_age("What treatment for a 4-year-old child?")
        self.assertEqual(age_label, "children_under_6")
        self.assertTrue(is_u6)

        age_label2, is_u6_2 = parse_query_age("Asthma management in toddlers under 6")
        self.assertEqual(age_label2, "children_under_6")
        self.assertTrue(is_u6_2)

        # 6-11
        age_label3, is_u6_3 = parse_query_age("Treatment for 8-year-old child with asthma")
        self.assertEqual(age_label3, "children_6_11")
        self.assertFalse(is_u6_3)

        # 12+ (adults/adolescents)
        age_label4, is_u6_4 = parse_query_age("First line therapy for adults and teenagers")
        self.assertEqual(age_label4, "adults_adolescents")
        self.assertFalse(is_u6_4)

    def test_under_6_filters_out_gina_sources(self):
        chunks = [
            Chunk(
                chunk_id="who-001",
                text="WHO guidelines cover children under 5 with suspected asthma.",
                document="WHO childhood asthma guideline 2026",
                publisher="WHO", source_url="http://who.int",
                topic="treatment", section="Under 5s",
                page_start=5, page_end=5, token_count=10, boundary_reason="section",
                age_group="children_under_6", covers_under_6=True,
            ),
            Chunk(
                chunk_id="gina-001",
                text="GINA Table 4 treatment options for children 6-11 years.",
                document="GINA-Summary-Guide-2026-WEB-WMS",
                publisher="GINA", source_url="https://ginasthma.org",
                topic="treatment", section="Table 4",
                page_start=12, page_end=12, token_count=10, boundary_reason="section",
                age_group="children_6_11", covers_under_6=False,
            ),
        ]
        bm25 = BM25Retriever(chunks)
        class _FakeRepo:
            def search(self, q, top_k=5, min_score_threshold=0.0):
                return bm25.search(q, top_k=top_k)

        retriever = UnifiedRetriever(_FakeRepo(), chunks)
        # Search for a 4-year-old query
        results = retriever.search("asthma treatment for a 4-year-old child", strategy="bm25", top_k=5)
        # Only WHO chunk (covers_under_6=True) should be present
        for r in results:
            self.assertTrue(r.metadata.get("covers_under_6", True))


if __name__ == "__main__":
    unittest.main()
