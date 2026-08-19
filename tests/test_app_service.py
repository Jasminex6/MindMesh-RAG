"""Unit tests for RagApplicationService."""

import unittest
from unittest.mock import MagicMock

from medical_rag.app_service import RagApplicationService
from medical_rag.contracts import RAGResponse
from medical_rag.generation import GeneratedAnswer
from medical_rag.models import SearchResult


class TestRagApplicationService(unittest.TestCase):

    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_gen_service = MagicMock()
        self.service = RagApplicationService(
            retriever=self.mock_retriever,
            generation_service=self.mock_gen_service,
        )

    def test_out_of_scope_query_refused(self):
        """Conversational joke query returns REFUSAL RAGResponse with 0 retrieval calls."""
        response = self.service.ask("tell me a joke")
        self.assertIsInstance(response, RAGResponse)
        self.assertEqual(response.status, "REFUSAL")
        self.mock_retriever.search.assert_not_called()

    def test_clinical_query_returns_answer(self):
        """Valid clinical query invokes retrieval and generation."""
        self.mock_retriever.search.return_value = [
            SearchResult(rank=1, score=0.85, text="Sample clinical chunk", metadata={"chunk_id": "chunk-001"})
        ]
        self.mock_gen_service.generate.return_value = GeneratedAnswer(
            query="asthma symptoms",
            recommendation="Wheeze and cough",
            supporting_evidence="- Wheeze and cough",
            citations=[],
            confidence="High",
            safety_note="Grounded evidence.",
        )

        response = self.service.ask("asthma symptoms")
        self.assertIsInstance(response, RAGResponse)
        self.assertIn(response.status, ("SUCCESS", "ANSWER"))
        self.assertGreater(len(response.evidence), 0)

    def test_multi_turn_follow_up_contextualization(self):
        """Follow-up question 'how is it treated in children under 5?' resolves pronoun 'it' from history."""
        self.mock_retriever.search.return_value = [
            SearchResult(rank=1, score=0.85, text="ICS treatment chunk", metadata={"chunk_id": "chunk-002"})
        ]
        self.mock_gen_service.generate.return_value = GeneratedAnswer(
            query="How is paediatric asthma treated?",
            recommendation="Low dose ICS",
            supporting_evidence="- Low dose ICS",
            citations=[],
            confidence="High",
            safety_note="Grounded.",
        )

        chat_history = [
            {"role": "user", "content": "paediatric asthma symptoms"},
            {"role": "assistant", "content": "Cough and wheeze"},
        ]

        response = self.service.ask("how is it treated?", chat_history=chat_history)
        self.assertTrue("treat" in response.resolved_query.lower() and ("child" in response.resolved_query.lower() or "pediatric" in response.resolved_query.lower() or "paediatric" in response.resolved_query.lower()))


if __name__ == "__main__":
    unittest.main()
