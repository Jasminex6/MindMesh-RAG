"""End-to-end integration flow tests for multi-turn conversational RAG."""

import unittest
from unittest.mock import patch, MagicMock

from medical_rag.models import SearchResult
from medical_rag.generation import GenerationService, GeneratedAnswer, Citation
from medical_rag.query_rewriter import ConversationalQueryRewriter


def _make_result(rank: int, score: float, chunk_id: str,
                 document: str = "WHO_Asthma_Guideline_2026.pdf",
                 section: str = "Pediatric Management",
                 text: str = "Inhaled corticosteroids (ICS) are recommended as first-line controller therapy for pediatric asthma.") -> SearchResult:
    return SearchResult(
        rank=rank,
        score=score,
        text=text,
        metadata={
            "chunk_id": chunk_id,
            "document": document,
            "section": section,
            "page": 12,
            "page_start": 12,
            "page_end": 12,
            "token_count": 45,
        },
    )


class TestIntegrationFlow(unittest.TestCase):

    def setUp(self):
        self.gen_service = GenerationService(model="llama3.2", temperature=0.1)

    def test_multiturn_followup_rewriting_and_grounding(self):
        """Verify that multi-turn query follow-ups pass chat_history through prompt building."""
        history = [
            {"role": "user", "content": "What are paediatric asthma symptoms?"},
            {"role": "assistant", "content": "Common symptoms include wheezing, cough, and shortness of breath."},
        ]
        results = [_make_result(1, 0.88, "chunk-ics-001")]

        # Skip LLM call to verify prompt assembly and data structure
        answer = self.gen_service.generate(
            "How is it treated?",
            results=results,
            chat_history=history,
            skip_llm=True,
        )

        self.assertFalse(answer.refused)
        self.assertIn(answer.confidence, ["High", "Medium"])
        self.assertEqual(answer.query, "How is it treated?")

    @patch("medical_rag.generation.call_llm")
    def test_multiturn_elaboration_with_llm(self, mock_llm):
        """Verify generation with chat_history and LLM response parsing."""
        mock_llm.return_value = '{"recommendation": "Inhaled corticosteroids preferred controller", "supporting_evidence": ["Inhaled corticosteroids (ICS) recommended"], "citations": [{"claim": "Inhaled corticosteroids preferred controller", "chunk_id": "chunk-ics-001"}], "safety_note": "Grounded."}'
        
        history = [
            {"role": "user", "content": "What are paediatric asthma symptoms?"},
            {"role": "assistant", "content": "Common symptoms include wheezing, cough, and shortness of breath."},
        ]
        results = [_make_result(1, 0.88, "chunk-ics-001")]

        answer = self.gen_service.generate(
            "What is the first-line treatment for it?",
            results=results,
            chat_history=history,
            skip_llm=False,
        )

        self.assertFalse(answer.refused)
        self.assertIn(answer.confidence, ["High", "Medium"])
        self.assertTrue(len(answer.citations) > 0)
        self.assertTrue(answer.citations[0].verified)


if __name__ == "__main__":
    unittest.main()
