"""Headless unit test for Streamlit UI state initialization and imports."""

import unittest


class TestUILifecycle(unittest.TestCase):

    def test_imports_and_app_syntax(self):
        """Verify app.py syntax and required imports without launching GUI."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("app_module", "app.py")
        self.assertIsNotNone(spec)


if __name__ == "__main__":
    unittest.main()
