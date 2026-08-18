import unittest

from medical_rag.chunking import SectionAwareChunker
from medical_rag.models import Page, ParsedGuideline, SourceSpec


def guideline(text: str) -> ParsedGuideline:
    source = SourceSpec(
        file_name="test.pdf",
        document="Test guideline",
        publisher="Test publisher",
        source_url="https://example.test/guideline",
        usage_note="Test fixture",
    )
    return ParsedGuideline(
        source=source,
        file_path="test.pdf",
        pages=(Page(1, text, text),),
    )


class ChunkingTests(unittest.TestCase):
    def test_section_boundaries_do_not_share_overlap(self):
        text = "1 Introduction\n" + ("background evidence " * 30) + "\n2 Management\n" + ("treatment recommendation " * 30)
        chunks = SectionAwareChunker(chunk_size=60, overlap=10).chunk(guideline(text), "test topic")
        management = [chunk for chunk in chunks if chunk.section == "2 Management"]
        self.assertTrue(management)
        self.assertNotIn("background evidence", management[0].text)

    def test_chunk_metadata_and_ids_are_complete(self):
        text = "1 Diagnosis\n" + ("objective testing spirometry " * 50)
        chunks = SectionAwareChunker(chunk_size=50, overlap=8).chunk(guideline(text), "asthma")
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(chunks), len({chunk.chunk_id for chunk in chunks}))
        for chunk in chunks:
            self.assertLessEqual(chunk.token_count, 50)
            self.assertEqual(chunk.metadata()["page"], 1)
            self.assertEqual(chunk.metadata()["section"], "1 Diagnosis")

    def test_consecutive_headings_are_kept_with_following_content(self):
        text = "3 Management\n3.1 Acute management\n" + ("clinical recommendation " * 20)
        chunks = SectionAwareChunker(chunk_size=80, overlap=10).chunk(guideline(text), "asthma")
        self.assertGreaterEqual(chunks[0].token_count, 20)
        self.assertIn("3 Management", chunks[0].text)
        self.assertIn("3.1 Acute management", chunks[0].text)
        self.assertIn("clinical recommendation", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
