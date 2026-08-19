import unittest

from medical_rag.chunking import is_section_heading


class HeadingDetectionTests(unittest.TestCase):
    def test_body_lines_with_keywords_are_not_headings(self):
        self.assertFalse(is_section_heading(
            "• Immediately started with first-line treatment"
        ))
        self.assertFalse(is_section_heading(
            "strong recommendation would encourage procurement"
        ))
        self.assertFalse(is_section_heading(
            "Full details of the evidence are in evidence review R:"
        ))

    def test_guideline_headings_are_detected(self):
        self.assertTrue(is_section_heading("4.2 Second-line therapy options"))
        self.assertTrue(is_section_heading("Recommendation 2:"))
        self.assertTrue(is_section_heading("Monitoring asthma control"))

    def test_deep_numbered_recommendation_is_not_a_section(self):
        self.assertFalse(is_section_heading(
            "1.4.2 Refer people with suspected occupational asthma"
        ))


if __name__ == "__main__":
    unittest.main()
