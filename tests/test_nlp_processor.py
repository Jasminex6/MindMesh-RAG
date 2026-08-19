"""Unit tests for NLP Query Processor module."""

import unittest
from medical_rag.nlp_processor import (
    NlpQueryProcessor,
    detect_language,
    extract_clinical_terms,
    detect_topic_shift,
    decompose_bilingual_multi_query,
)


class TestNlpQueryProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = NlpQueryProcessor()

    def test_detect_language(self):
        self.assertEqual(detect_language("What are asthma symptoms?"), "en")
        self.assertEqual(detect_language("ما هي أعراض الربو عند الأطفال؟"), "ar")
        self.assertEqual(detect_language("بخاخ فنتولين للأطفال"), "ar")

    def test_extract_arabic_clinical_terms(self):
        terms, age_band = extract_clinical_terms("ما هي أعراض حساسية الصدر للأطفال أقل من 6 سنوات؟")
        self.assertTrue(terms.get("asthma"))
        self.assertTrue(terms.get("symptoms"))
        self.assertEqual(age_band, "under_6")

    def test_extract_english_age_terms(self):
        terms, age_band = extract_clinical_terms("What treatment is recommended for an infant with asthma?")
        self.assertEqual(age_band, "under_6")

    def test_detect_topic_shift_new_topic(self):
        history = [
            {"role": "user", "content": "What are pediatric asthma symptoms?"},
            {"role": "assistant", "content": "Wheezing and cough."},
        ]
        # Query about diabetes is a topic shift
        self.assertTrue(detect_topic_shift("What is the treatment for diabetes?", history))

    def test_detect_topic_shift_self_contained_query(self):
        history = [
            {"role": "user", "content": "What are pediatric asthma symptoms?"},
            {"role": "assistant", "content": "Wheezing and cough."},
        ]
        # Self-contained question with explicit subject is a topic shift (does not force history)
        self.assertTrue(detect_topic_shift("What are the diagnostic criteria for severe asthma in adults?", history))

    def test_detect_topic_shift_implicit_followup(self):
        history = [
            {"role": "user", "content": "What are pediatric asthma symptoms?"},
            {"role": "assistant", "content": "Wheezing and cough."},
        ]
        # Implicit pronoun query is NOT a topic shift
        self.assertFalse(detect_topic_shift("How is it treated?", history))
        self.assertFalse(detect_topic_shift("وما هي آثاره الجانبية؟", history))

    def test_bilingual_multi_query_decomposition_english(self):
        is_multi, subs = decompose_bilingual_multi_query("1. What are asthma symptoms? 2. How is it treated?")
        self.assertTrue(is_multi)
        self.assertEqual(len(subs), 2)
        self.assertIn("What are asthma symptoms", subs[0])
        self.assertIn("How is it treated", subs[1])

    def test_bilingual_multi_query_decomposition_arabic(self):
        is_multi, subs = decompose_bilingual_multi_query("ما هي أعراض الربو؟ وكيف يتم علاجه؟")
        self.assertTrue(is_multi)
        self.assertEqual(len(subs), 2)

    def test_processor_full_pipeline_ar(self):
        res = self.processor.process("أعراض الربو للأطفال تحت 6")
        self.assertEqual(res.language, "ar")
        self.assertEqual(res.implied_age_band, "under_6")
        self.assertIn("ما هي أعراض الربو عند الأطفال؟", res.reconstructed_query)


if __name__ == "__main__":
    unittest.main()
