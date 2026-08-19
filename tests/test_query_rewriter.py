"""Unit tests for ConversationalQueryRewriter."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from medical_rag.query_rewriter import ConversationalQueryRewriter, rewrite_conversational_query


class TestConversationalQueryRewriter(unittest.TestCase):
    """Test suite for ConversationalQueryRewriter module."""

    def setUp(self):
        self.rewriter = ConversationalQueryRewriter()

    def test_empty_query_or_history_returns_unchanged(self):
        """If history is empty or query is empty, return original query."""
        self.assertEqual(self.rewriter.rewrite(""), "")
        self.assertEqual(self.rewriter.rewrite("What is asthma?"), "What is asthma?")
        self.assertEqual(self.rewriter.rewrite("What is asthma?", chat_history=[]), "What is asthma?")

    def test_sliding_window_formatting(self):
        """Ensure sliding window limits history to max_turns."""
        rewriter = ConversationalQueryRewriter(max_turns=2)
        history = [
            {"role": "user", "content": "Turn 1 Q"},
            {"role": "assistant", "content": "Turn 1 A"},
            {"role": "user", "content": "Turn 2 Q"},
            {"role": "assistant", "content": "Turn 2 A"},
            {"role": "user", "content": "Turn 3 Q"},
            {"role": "assistant", "content": "Turn 3 A"},
        ]
        formatted = rewriter._format_history(history)
        self.assertNotIn("Turn 1 Q", formatted)
        self.assertIn("Turn 2 Q", formatted)
        self.assertIn("Turn 3 Q", formatted)

    def test_heuristic_fallback_with_skip_llm(self):
        """Test fallback query reformulation when LLM is skipped."""
        history = [
            {"role": "user", "content": "What are the first-line controllers for asthma in children?"},
            {"role": "assistant", "content": "First-line controllers include low-dose ICS and LTRA."},
        ]
        query = "What about the side effects of the second one?"
        rewritten = self.rewriter.rewrite(query, chat_history=history, skip_llm=True)
        self.assertIn("What about the side effects of the second one?", rewritten)
        self.assertIn("regarding What are the first-line controllers for asthma in children?", rewritten)

    @patch("langchain_ollama.ChatOllama")
    def test_mocked_llm_rewrite(self, mock_chat_ollama):
        """Test LLM-based query rewriting with mocked LLM."""
        mock_response = MagicMock()
        mock_response.content = "What are the side effects of Leukotriene Receptor Antagonists (LTRA) for asthma in children?"
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_response
        mock_chat_ollama.return_value = mock_instance

        history = [
            {"role": "user", "content": "What are the first-line controllers for asthma in children?"},
            {"role": "assistant", "content": "1. Low-dose ICS. 2. Leukotriene receptor antagonists (LTRA)."},
        ]
        query = "What about the side effects of the second one?"
        rewritten = self.rewriter.rewrite(query, chat_history=history)

        self.assertEqual(
            rewritten,
            "What are the side effects of Leukotriene Receptor Antagonists (LTRA) for asthma in children?",
        )

    def test_convenience_function(self):
        """Test rewrite_conversational_query convenience wrapper."""
        history = [
            {"role": "user", "content": "What is ICS?"},
            {"role": "assistant", "content": "ICS stands for Inhaled Corticosteroids."},
        ]
        query = "Can you elaborate on that?"
        res = rewrite_conversational_query(query, chat_history=history, skip_llm=True)
        self.assertIn("Can you elaborate on that?", res)
        self.assertIn("regarding What is ICS?", res)


if __name__ == "__main__":
    unittest.main()
