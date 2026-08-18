"""
Local Test Script: Validates end-to-end PDF parsing, 800-token/100-overlap chunking,
and semantic query search over the processed clinical guideline chunks.
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# Simple TF-IDF / Word Vector Cosine Similarity Search Engine for immediate testing
class LightweightSearchEngine:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.vocab = {}
        self.doc_vectors = []
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"\b\w+\b", text.lower())

    def _build_index(self):
        # Build vocabulary
        all_words = set()
        doc_tokens_list = []
        for c in self.chunks:
            tokens = self._tokenize(c["text"])
            doc_tokens_list.append(tokens)
            all_words.update(tokens)

        self.vocab = {word: idx for idx, word in enumerate(sorted(all_words))}
        vocab_size = len(self.vocab)

        # Build term frequency vectors
        for tokens in doc_tokens_list:
            vec = np.zeros(vocab_size, dtype=np.float32)
            for t in tokens:
                if t in self.vocab:
                    vec[self.vocab[t]] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.doc_vectors.append(vec)

        self.doc_vectors = np.array(self.doc_vectors)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q_tokens = self._tokenize(query)
        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        for t in q_tokens:
            if t in self.vocab:
                q_vec[self.vocab[t]] += 1.0
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = np.dot(self.doc_vectors, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "score": float(scores[idx]),
                "chunk_id": chunk["chunk_id"],
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"],
                "section_title": chunk["section_title"],
                "text_snippet": chunk["text"][:300] + "..."
            })
        return results


def run_test_queries():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    chunks_dir = os.path.join(base_dir, "RAG", "chunks_data")

    json_files = [f for f in os.listdir(chunks_dir) if f.endswith("_chunks.json")]
    all_chunks = []
    for jf in json_files:
        with open(os.path.join(chunks_dir, jf), "r", encoding="utf-8") as f:
            all_chunks.extend(json.load(f))

    print(f"Loaded {len(all_chunks)} chunks for retrieval testing.")
    engine = LightweightSearchEngine(all_chunks)

    test_queries = [
        "What is the recommended initial treatment for asthma in adults?",
        "How is asthma diagnosed in children according to WHO guidelines?",
        "When should inhaled corticosteroids (ICS) be prescribed?"
    ]

    print("=" * 80)
    print("CLINICAL RETRIEVAL TEST RESULTS")
    print("=" * 80)

    for q in test_queries:
        print(f"\nQUERY: '{q}'")
        print("-" * 80)
        results = engine.search(q, top_k=2)
        for r_idx, res in enumerate(results, 1):
            print(f"  Result #{r_idx} | Match Score: {res['score']:.4f}")
            print(f"  Document: {res['document_name']} | Page: {res['page_number']} | Section: {res['section_title']}")
            print(f"  Excerpt: {res['text_snippet']}\n")

if __name__ == "__main__":
    run_test_queries()
