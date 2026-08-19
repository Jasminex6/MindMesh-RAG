import unittest

from medical_rag.evaluation import precision_at_k


class EvaluationTests(unittest.TestCase):
    def test_precision_at_k(self):
        self.assertEqual(precision_at_k([True, True, False, True, False], 5), 0.6)

    def test_precision_requires_k_labels(self):
        with self.assertRaises(ValueError):
            precision_at_k([True], 3)


if __name__ == "__main__":
    unittest.main()
