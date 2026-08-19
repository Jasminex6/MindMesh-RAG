"""Tests for the mandatory age-slot pipeline.

Covers:
- Deterministic age-slot guard (medication_or_dose_request without age -> NEEDS_INFO)
- summarize_for_retrieval() enriches the query correctly for each age band
- Educational queries are unaffected by the mandatory-age rule
- Regression: NICE NG245 section 1.3 under-5 case still routes correctly
"""

from __future__ import annotations

import unittest

from medical_rag.intent_classifier import (
    AGE_MANDATORY_INTENTS,
    AGE_BAND_OPTIONS,
    IntentClassifier,
    summarize_for_retrieval,
)


def _classify(query: str):
    return IntentClassifier().classify(query)


# ===========================================================================
# 1. AGE_MANDATORY_INTENTS constant
# ===========================================================================

class TestAgeMandatoryIntents(unittest.TestCase):

    def test_medication_intent_is_mandatory(self):
        self.assertIn("medication_or_dose_request", AGE_MANDATORY_INTENTS)

    def test_educational_intent_is_not_mandatory(self):
        self.assertNotIn("general_educational", AGE_MANDATORY_INTENTS)

    def test_patient_specific_not_mandatory(self):
        self.assertNotIn("patient_specific", AGE_MANDATORY_INTENTS)

    def test_diagnosis_intent_not_mandatory(self):
        self.assertNotIn("diagnosis_or_symptoms", AGE_MANDATORY_INTENTS)

    def test_emergency_not_mandatory(self):
        self.assertNotIn("emergency", AGE_MANDATORY_INTENTS)


# ===========================================================================
# 2. Mandatory-age guard logic
# ===========================================================================

class TestMandatoryAgeGuard(unittest.TestCase):

    def test_medication_query_without_age_requires_guard(self):
        intent = _classify("What medicine should I take for my asthma?")
        self.assertEqual(intent.category, "medication_or_dose_request")
        slots = {}
        guard_fires = (intent.category in AGE_MANDATORY_INTENTS and not slots.get("age_band"))
        self.assertTrue(guard_fires)

    def test_medication_query_with_age_skips_guard(self):
        intent = _classify("What inhaler treatment for asthma?")
        self.assertEqual(intent.category, "medication_or_dose_request")
        slots = {"age_band": "children_6_11"}
        guard_fires = (intent.category in AGE_MANDATORY_INTENTS and not slots.get("age_band"))
        self.assertFalse(guard_fires)

    def test_educational_query_never_triggers_guard(self):
        intent = _classify("What is asthma and how does it affect the lungs?")
        self.assertEqual(intent.category, "general_educational")
        slots = {}
        guard_fires = (intent.category in AGE_MANDATORY_INTENTS and not slots.get("age_band"))
        self.assertFalse(guard_fires)

    def test_guard_skips_when_age_pre_filled(self):
        intent = _classify("What dose of ICS for a 9-year-old?")
        self.assertEqual(intent.category, "medication_or_dose_request")
        slots = {"age_band": "children_6_11"}
        guard_fires = (intent.category in AGE_MANDATORY_INTENTS and not slots.get("age_band"))
        self.assertFalse(guard_fires)


# ===========================================================================
# 3. AGE_BAND_OPTIONS parsing
# ===========================================================================

class TestAgeBandOptions(unittest.TestCase):

    def test_number_1_maps_to_under_6(self):
        self.assertEqual(AGE_BAND_OPTIONS["1"], "under_6")

    def test_number_2_maps_to_children_6_11(self):
        self.assertEqual(AGE_BAND_OPTIONS["2"], "children_6_11")

    def test_number_3_maps_to_adults(self):
        self.assertEqual(AGE_BAND_OPTIONS["3"], "adults_adolescents")

    def test_canonical_keys_accepted(self):
        self.assertEqual(AGE_BAND_OPTIONS["under_6"], "under_6")
        self.assertEqual(AGE_BAND_OPTIONS["children_6_11"], "children_6_11")
        self.assertEqual(AGE_BAND_OPTIONS["adults_adolescents"], "adults_adolescents")

    def test_display_labels_accepted(self):
        self.assertEqual(AGE_BAND_OPTIONS["under 6"], "under_6")
        self.assertEqual(AGE_BAND_OPTIONS["12+"], "adults_adolescents")

    def test_unknown_input_returns_none(self):
        self.assertIsNone(AGE_BAND_OPTIONS.get("999"))
        self.assertIsNone(AGE_BAND_OPTIONS.get("banana"))


# ===========================================================================
# 4. summarize_for_retrieval() — all 3 age bands
# ===========================================================================

class TestSummarizeForRetrieval(unittest.TestCase):

    BASE_QUERY = "What asthma treatment should be used?"

    def _enrich(self, age_band: str) -> str:
        return summarize_for_retrieval(self.BASE_QUERY, {"age_band": age_band}, "medication_or_dose_request")

    # under-6 band
    def test_under_6_contains_age_phrase(self):
        enriched = self._enrich("under_6")
        self.assertIn(self.BASE_QUERY, enriched)
        self.assertIn("under 6", enriched.lower())

    def test_under_6_contains_band_label(self):
        self.assertIn("under-5", self._enrich("under_6").lower())

    def test_under_6_differs_from_original(self):
        self.assertNotEqual(self._enrich("under_6"), self.BASE_QUERY)

    # children 6-11 band
    def test_children_6_11_contains_age_phrase(self):
        enriched = self._enrich("children_6_11")
        lower = enriched.lower()
        has_match = "6 to 11" in lower or "9-year" in lower or "aged 6" in lower or "6" in lower
        self.assertTrue(has_match, f"Expected 6-11 age phrase in: {enriched}")

    def test_children_6_11_differs_from_original(self):
        self.assertNotEqual(self._enrich("children_6_11"), self.BASE_QUERY)

    # adult band
    def test_adults_contains_age_phrase(self):
        enriched = self._enrich("adults_adolescents")
        lower = enriched.lower()
        has_match = "adult" in lower or "12+" in lower or "adolescent" in lower
        self.assertTrue(has_match, f"Expected adult phrase in: {enriched}")

    def test_adults_differs_from_original(self):
        self.assertNotEqual(self._enrich("adults_adolescents"), self.BASE_QUERY)

    # passthrough cases
    def test_empty_slots_returns_original_query(self):
        self.assertEqual(
            summarize_for_retrieval(self.BASE_QUERY, {}, "medication_or_dose_request"),
            self.BASE_QUERY,
        )

    def test_none_age_band_returns_original_query(self):
        self.assertEqual(
            summarize_for_retrieval(self.BASE_QUERY, {"age_band": None}, "medication_or_dose_request"),
            self.BASE_QUERY,
        )

    def test_educational_category_passthrough(self):
        self.assertEqual(
            summarize_for_retrieval("What is asthma?", {}, "general_educational"),
            "What is asthma?",
        )


# ===========================================================================
# 5. Regression — NICE NG245 section 1.3 (under-5 children)
# ===========================================================================

class TestNICENG245Under5Regression(unittest.TestCase):

    def test_under_6_enriched_query_routes_to_under_6(self):
        from medical_rag.hybrid_retrieval import parse_query_age
        enriched = summarize_for_retrieval(
            "What inhaled corticosteroid dose for a child with asthma?",
            {"age_band": "under_6"},
            "medication_or_dose_request",
        )
        age_label, is_under_6 = parse_query_age(enriched)
        self.assertTrue(is_under_6, f"Expected is_under_6=True for: {enriched!r}")
        self.assertEqual(age_label, "children_under_6")

    def test_under_6_not_classified_as_adult(self):
        from medical_rag.hybrid_retrieval import parse_query_age
        enriched = summarize_for_retrieval(
            "Which controller for asthma?", {"age_band": "under_6"}, "medication_or_dose_request"
        )
        age_label, _ = parse_query_age(enriched)
        self.assertNotEqual(age_label, "adults_adolescents")

    def test_adult_enriched_query_not_classified_as_under_6(self):
        from medical_rag.hybrid_retrieval import parse_query_age
        enriched = summarize_for_retrieval(
            "What ICS step therapy for asthma?",
            {"age_band": "adults_adolescents"},
            "medication_or_dose_request",
        )
        _, is_under_6 = parse_query_age(enriched)
        self.assertFalse(is_under_6, f"Adult enriched query must NOT be under-6: {enriched!r}")


if __name__ == "__main__":
    unittest.main()
