"""Unit tests for Safety & Intent Classification Layer."""

from __future__ import annotations

import unittest
from medical_rag.intent_classifier import IntentClassifier, IntentDecision


class TestIntentClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = IntentClassifier()

    def test_general_educational_query(self):
        query = "What is asthma and how does it affect the lungs?"
        intent = self.classifier.classify(query)
        self.assertEqual(intent.category, "general_educational")
        self.assertFalse(intent.requires_clarification)
        self.assertFalse(intent.requires_emergency)

    def test_medication_or_dose_request_query(self):
        query = "What dose of ICS should I use for asthma?"
        intent = self.classifier.classify(query)
        self.assertEqual(intent.category, "medication_or_dose_request")
        self.assertTrue(intent.requires_clarification)
        self.assertFalse(intent.requires_emergency)
        # Should contain adaptive questions
        q_ids = [q.id for q in intent.adaptive_questions]
        self.assertIn("age_group", q_ids)
        self.assertIn("current_symptoms", q_ids)

    def test_medication_request_with_implied_age_skips_age_question(self):
        query = "What dose of ICS should I use for an 8-year-old child?"
        intent = self.classifier.classify(query)
        self.assertEqual(intent.category, "medication_or_dose_request")
        self.assertEqual(intent.implied_age_group, "children_6_11")
        q_ids = [q.id for q in intent.adaptive_questions]
        self.assertNotIn("age_group", q_ids)  # Skipped because age was implied

    def test_diagnosis_or_symptoms_query(self):
        query = "Do I have asthma? Why do I keep coughing at night?"
        intent = self.classifier.classify(query)
        self.assertEqual(intent.category, "diagnosis_or_symptoms")
        self.assertTrue(intent.requires_clarification)
        self.assertFalse(intent.requires_emergency)

    def test_patient_specific_query(self):
        query = "Should I be worried about my child's symptoms?"
        intent = self.classifier.classify(query)
        self.assertEqual(intent.category, "patient_specific")
        self.assertTrue(intent.requires_clarification)
        self.assertFalse(intent.requires_emergency)

    def test_emergency_query_bypasses_retrieval(self):
        emergency_queries = [
            "My child is gasping for air and turning blue!",
            "Patient is suffocating, can't breathe and can't speak in full sentences",
            "Call 911 my baby has severe respiratory distress",
        ]
        for eq in emergency_queries:
            intent = self.classifier.classify(eq)
            self.assertEqual(intent.category, "emergency")
            self.assertFalse(intent.requires_clarification)
            self.assertTrue(intent.requires_emergency)
            self.assertIn("EMERGENCY", intent.emergency_response)


if __name__ == "__main__":
    unittest.main()
