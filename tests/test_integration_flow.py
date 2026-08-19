"""End-to-End Integration Tests for ask_question() Pipeline.

Tests the complete flow from entry point through single-classifier routing,
slot enforcement, retrieval, and generation to prevent classifier conflicts.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo import ask_question
from medical_rag.models import SearchResult
from medical_rag.generation import GenerationService, GeneratedAnswer


def _mock_results() -> list[SearchResult]:
    return [
        SearchResult(
            rank=1,
            score=0.88,
            text="Inhaled corticosteroids (ICS) are recommended for regular controller therapy in asthma.",
            metadata={
                "chunk_id": "chunk-001",
                "document": "GINA-Summary-Guide-2026-WEB-WMS",
                "section": "Initial Controller Treatment",
                "page": "15",
                "page_start": 15,
                "page_end": 15,
            },
        )
    ]


class TestAskQuestionEndToEnd(unittest.TestCase):
    """Integration tests calling the actual ask_question() entry point."""

    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_retriever.search.return_value = _mock_results()

        self.mock_gen_service = MagicMock(spec=GenerationService)
        self.mock_gen_service.generate.return_value = GeneratedAnswer(
            query="test",
            recommendation="Guideline recommendations suggest low-dose ICS for controller therapy.",
            supporting_evidence="Inhaled corticosteroids (ICS) are recommended.",
            citations=[],
            confidence="High",
            safety_note="Grounded in official guideline evidence.",
            refused=False,
        )

    def test_coughing_night_diagnosis_triggers_clarification_not_refusal(self):
        """Case 1: 'I keep coughing at night, could it be asthma?'
        Must trigger clarification questions, MUST NOT be hard-refused.
        """
        query = "I keep coughing at night, could it be asthma?"
        result = ask_question(query, self.mock_retriever, self.mock_gen_service)

        # Must NOT be refused
        self.assertIsNotNone(result)
        if isinstance(result, GeneratedAnswer):
            self.assertFalse(result.refused, "Diagnosis query must NOT be hard refused")

        # Must have called retrieval/generation
        self.assertTrue(self.mock_retriever.search.called)

    def test_treatment_recommendation_requires_age_slot_not_refusal(self):
        """Case 2: 'what treatment do you recommend for my son's asthma?'
        Must require age slot / proceed to retrieval once age provided, MUST NOT be hard-refused.
        """
        query = "what treatment do you recommend for my son's asthma?"
        # Pre-fill age_band to simulate user completing age slot
        slots = {"age_band": "under_6"}

        result = ask_question(query, self.mock_retriever, self.mock_gen_service, slots=slots)

        # Must NOT be refused
        self.assertIsNotNone(result)
        if isinstance(result, GeneratedAnswer):
            self.assertFalse(result.refused, "Treatment recommendation query must NOT be hard refused")

        # Must have called retrieval/generation
        self.assertTrue(self.mock_retriever.search.called)

    def test_personal_dosing_decision_hard_refused(self):
        """Case 3: 'should I give my son a second puff right now?'
        Must be HARD REFUSED with personal_dosing_decision reason, bypassing retrieval.
        """
        query = "should I give my son a second puff right now?"
        result = ask_question(query, self.mock_retriever, self.mock_gen_service)

        # MUST be hard refused
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, GeneratedAnswer))
        self.assertTrue(result.refused, "Personal dosing decision query MUST be hard refused")
        self.assertIn("Personalized medication dosage decisions", result.refusal_reason)

        # Retrieval/generation MUST NOT be called
        self.assertFalse(self.mock_retriever.search.called)

    @patch("medical_rag.query_rewriter.ConversationalQueryRewriter.rewrite")
    def test_multiturn_followup_rewriting_and_grounding(self, mock_rewrite):
        """Case 4: Multi-turn conversation with pronoun-heavy follow-up ('What about the second one?').
        Must rewrite the follow-up into a standalone query, run retrieval, and pass grounding without refusal.
        """
        mock_rewrite.return_value = "What are the indications and side effects of Leukotriene Receptor Antagonists (LTRA) in children?"

        chat_history = []
        # Turn 1
        q1 = "What are the recommended controller treatments for asthma in children?"
        slots = {"age_band": "children_6_11"}
        res1 = ask_question(q1, self.mock_retriever, self.mock_gen_service, slots=slots, chat_history=chat_history)
        self.assertIsNotNone(res1)
        self.assertFalse(res1.refused)

        # Verify chat_history recorded Turn 1
        self.assertEqual(len(chat_history), 2)

        # Turn 2: Pronoun-heavy follow-up
        q2 = "What about the second one?"
        res2 = ask_question(
            q2,
            self.mock_retriever,
            self.mock_gen_service,
            slots=slots,
            chat_history=chat_history,
        )

        self.assertIsNotNone(res2)
        self.assertFalse(res2.refused, "Multi-turn follow-up must NOT be refused")
        # Verify query rewriter was called with chat history
        mock_rewrite.assert_called()

        # Verify chat_history recorded Turn 2
        self.assertEqual(len(chat_history), 4)

    def test_multiturn_elaboration_fallback(self):
        """Case 5: Multi-turn elaboration follow-up ('Can you elaborate on that?') using fallback rewriter.
        Must rewrite query using context and execute retrieval.
        """
        chat_history = [
            {"role": "user", "content": "How should asthma control be monitored?"},
            {"role": "assistant", "content": "Asthma control is monitored using symptom score and spirometry."},
        ]
        q2 = "Can you elaborate on that?"
        res2 = ask_question(
            q2,
            self.mock_retriever,
            self.mock_gen_service,
            chat_history=chat_history,
            skip_llm_rewriter=True,
        )
        self.assertIsNotNone(res2)
        if isinstance(res2, list):
            self.assertTrue(all(not a.refused for a in res2), "Elaboration follow-up must NOT be refused")
        else:
            self.assertFalse(res2.refused, "Elaboration follow-up must NOT be refused")
        self.assertTrue(self.mock_retriever.search.called)


if __name__ == "__main__":
    unittest.main()

