"""Unit tests for Conversational Query Rewriter module."""

import unittest
from unittest.mock import patch, MagicMock

from medical_rag.query_rewriter import (
    ConversationalQueryRewriter,
    rewrite_conversational_query,
    _normalize_history,
    _heuristic_fallback,
)


class TestConversationalQueryRewriter(unittest.TestCase):

    def test_empty_history_returns_original_query(self):
        rewriter = ConversationalQueryRewriter()
        self.assertEqual(rewriter.rewrite("What is asthma?", chat_history=[]), "What is asthma?")
        self.assertEqual(rewriter.rewrite("  What is asthma?  ", chat_history=None), "What is asthma?")

    def test_normalize_history_dict_and_tuple(self):
        dict_history = [
            {"role": "user", "content": "What are paediatric asthma symptoms?"},
            {"role": "assistant", "content": "Cough and wheeze."},
        ]
        tuple_history = [
            ("What are paediatric asthma symptoms?", "Cough and wheeze.")
        ]
        norm_dict = _normalize_history(dict_history)
        norm_tuple = _normalize_history(tuple_history)

        self.assertEqual(len(norm_dict), 2)
        self.assertEqual(len(norm_tuple), 2)
        self.assertEqual(norm_dict[0]["content"], "What are paediatric asthma symptoms?")
        self.assertEqual(norm_tuple[0]["content"], "What are paediatric asthma symptoms?")

    def test_sliding_window_max_messages(self):
        long_history = [
            {"role": "user", "content": f"Turn {i}"} for i in range(20)
        ]
        norm = _normalize_history(long_history, max_messages=8)
        self.assertEqual(len(norm), 8)
        self.assertEqual(norm[0]["content"], "Turn 12")
        self.assertEqual(norm[-1]["content"], "Turn 19")

    def test_heuristic_fallback_with_what_about(self):
        history = [
            {"role": "user", "content": "What is paediatric asthma treatment?"},
            {"role": "assistant", "content": "Low-dose ICS is recommended."},
        ]
        res = _heuristic_fallback("what about side effects?", history)
        self.assertIn("paediatric asthma treatment", res.lower())
        self.assertIn("side effects", res.lower())

    def test_heuristic_fallback_substitutes_pronoun(self):
        history = [
            {"role": "user", "content": "Tell me about Salbutamol inhalers"},
            {"role": "assistant", "content": "Salbutamol is a short-acting beta agonist."},
        ]
        res = _heuristic_fallback("How is it administered?", history)
        self.assertIn("Salbutamol", res)
        self.assertNotIn(" it ", res)

    def test_standalone_query_unmodified(self):
        history = [
            {"role": "user", "content": "What is paediatric asthma treatment?"},
            {"role": "assistant", "content": "Low-dose ICS is recommended."},
        ]
        res = _heuristic_fallback("What are the diagnostic criteria for severe asthma in adults?", history)
        self.assertEqual(res, "What are the diagnostic criteria for severe asthma in adults?")

    @patch("medical_rag.query_rewriter.ConversationalQueryRewriter.rewrite")
    def test_rewrite_conversational_query_wrapper(self, mock_rewrite):
        mock_rewrite.return_value = "Rewritten query"
        res = rewrite_conversational_query("How is it treated?", chat_history=[])
        self.assertEqual(res, "Rewritten query")


if __name__ == "__main__":
    unittest.main()
