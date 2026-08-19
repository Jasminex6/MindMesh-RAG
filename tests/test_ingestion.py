import unittest

from medical_rag.ingestion import IngestionService


class IngestionTests(unittest.TestCase):
    def test_cleaning_preserves_dose_and_date(self):
        service = IngestionService()
        text = "Header\nRecommendation 1: Give 2 mg daily in 2026.\n12"
        cleaned, audit = service.clean_page(text, 12, {"header"})
        self.assertIn("2 mg", cleaned)
        self.assertIn("2026", cleaned)
        self.assertNotIn("Header", cleaned)
        self.assertNotIn("\n12", cleaned)
        self.assertEqual({item.reason for item in audit}, {"page_number", "recurring_header_or_footer"})

    def test_hyphenated_word_is_repaired(self):
        service = IngestionService()
        cleaned, _ = service.clean_page("diag-\nnosis", 1, set())
        self.assertEqual(cleaned, "diagnosis")

    def test_navigation_artifact_is_removed(self):
        service = IngestionService()
        cleaned, audit = service.clean_page(
            "Return to recommendations\nClinical evidence remains.", 1, set()
        )
        self.assertEqual(cleaned, "Clinical evidence remains.")
        self.assertEqual(audit[0].reason, "navigation")


if __name__ == "__main__":
    unittest.main()
