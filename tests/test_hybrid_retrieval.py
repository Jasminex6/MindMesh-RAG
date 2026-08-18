import unittest
from medical_rag.models import Chunk, SearchResult
from medical_rag.hybrid_retrieval import (
    BM25Retriever,
    reciprocal_rank_fusion,
    CrossEncoderReranker,
    tokenize,
)


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk(
                chunk_id="chunk-001",
                text="Inhaled corticosteroids ICS are the preferred first-line controller therapy for paediatric asthma.",
                document="WHO guideline",
                publisher="WHO",
                source_url="http://who.int",
                topic="asthma management",
                section="Controller therapy",
                page_start=10,
                page_end=10,
                token_count=15,
                boundary_reason="section_boundary",
            ),
            Chunk(
                chunk_id="chunk-002",
                text="Intravenous magnesium sulfate is indicated for severe acute asthma exacerbations in children.",
                document="WHO guideline",
                publisher="WHO",
                source_url="http://who.int",
                topic="acute asthma",
                section="Exacerbations",
                page_start=34,
                page_end=35,
                token_count=16,
                boundary_reason="section_boundary",
            ),
            Chunk(
                chunk_id="chunk-003",
                text="Spirometry with bronchodilator reversibility testing confirms variable expiratory airflow limitation.",
                document="NICE guideline",
                publisher="NICE",
                source_url="http://nice.org.uk",
                topic="asthma diagnosis",
                section="Diagnosis",
                page_start=15,
                page_end=15,
                token_count=14,
                boundary_reason="section_boundary",
            ),
        ]

    def test_bm25_search_returns_relevant_chunk(self):
        bm25 = BM25Retriever(self.chunks)
        results = bm25.search("magnesium sulfate acute asthma", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].metadata["chunk_id"], "chunk-002")

    def test_reciprocal_rank_fusion(self):
        bm25 = BM25Retriever(self.chunks)
        res1 = bm25.search("magnesium sulfate", top_k=3)
        res2 = bm25.search("acute exacerbations", top_k=3)
        fused = reciprocal_rank_fusion([res1, res2], top_k=2)
        self.assertLessEqual(len(fused), 2)
        self.assertEqual(fused[0].metadata["chunk_id"], "chunk-002")

    def test_cross_encoder_reranker(self):
        reranker = CrossEncoderReranker()
        cands = [
            SearchResult(rank=1, score=0.5, text=self.chunks[0].text, metadata=self.chunks[0].metadata()),
            SearchResult(rank=2, score=0.4, text=self.chunks[1].text, metadata=self.chunks[1].metadata()),
        ]
        reranked = reranker.rerank("corticosteroids controller therapy", cands, top_k=2)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0].metadata["chunk_id"], "chunk-001")


if __name__ == "__main__":
    unittest.main()
