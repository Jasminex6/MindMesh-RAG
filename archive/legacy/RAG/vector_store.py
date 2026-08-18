"""
Vector Store & Indexing Engine for AI Clinical Decision Support Lite
Indexes chunked clinical text into persistent vector database (ChromaDB / JSON vector index),
preserving metadata (Document Name, Section Title, Page Number, Chunk ID).
"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Any

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class VectorStoreManager:
    def __init__(self, db_dir: str = "RAG/vector_db", collection_name: str = "clinical_guidelines"):
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.embedding_model = None
        self.client = None
        self.collection = None
        self.fallback_chunks = []
        self.vocab = {}
        self.doc_matrix = None

    def initialize(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model and vector database store"""
        os.makedirs(self.db_dir, exist_ok=True)

        if SentenceTransformer is not None:
            try:
                print(f"Loading embedding model: {model_name}...")
                self.embedding_model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Embedding model load notice: {e}")

        if chromadb is not None:
            try:
                self.client = chromadb.PersistentClient(path=self.db_dir)
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
                print(f"ChromaDB initialized at: {self.db_dir}")
            except Exception as e:
                print(f"ChromaDB init notice: {e}")

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Index chunks with embeddings and metadata into ChromaDB & persistent store"""
        if not chunks:
            print("No chunks provided for indexing.")
            return

        print(f"Indexing {len(chunks)} chunks into collection '{self.collection_name}'...")
        self.fallback_chunks = chunks

        # Save persistent JSON index for lightweight zero-dependency loading
        index_file = os.path.join(self.db_dir, "vector_index.json")
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved persistent JSON vector index to: {index_file}")

        # Build TF-IDF / TF vector matrix for fallback semantic search
        all_words = set()
        doc_toks = []
        for c in chunks:
            toks = self._tokenize(c["text"])
            doc_toks.append(toks)
            all_words.update(toks)

        self.vocab = {w: i for i, w in enumerate(sorted(all_words))}
        matrix = np.zeros((len(chunks), len(self.vocab)), dtype=np.float32)
        for d_idx, toks in enumerate(doc_toks):
            for w in toks:
                if w in self.vocab:
                    matrix[d_idx, self.vocab[w]] += 1.0
            norm = np.linalg.norm(matrix[d_idx])
            if norm > 0:
                matrix[d_idx] /= norm
        self.doc_matrix = matrix

        # Also populate ChromaDB if available
        if self.collection is not None:
            try:
                ids = [c["chunk_id"] for c in chunks]
                texts = [c["text"] for c in chunks]
                metadatas = [
                    {
                        "document_name": c["document_name"],
                        "file_path": c["file_path"],
                        "page_number": int(c["page_number"]),
                        "start_page": int(c["start_page"]),
                        "end_page": int(c["end_page"]),
                        "section_title": c["section_title"],
                        "token_count": int(c["token_count"])
                    }
                    for c in chunks
                ]

                if self.embedding_model is not None:
                    embeddings = self.embedding_model.encode(texts).tolist()
                    self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
                else:
                    self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
                print(f"Successfully indexed {len(chunks)} chunks in ChromaDB!")
            except Exception as e:
                print(f"ChromaDB indexing notice: {e}")

    def load_index(self):
        """Load persistent JSON vector index from disk if present"""
        index_file = os.path.join(self.db_dir, "vector_index.json")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            self.index_chunks(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search vector database for top_k relevant evidence chunks"""
        if self.collection is not None:
            try:
                if self.embedding_model is not None:
                    query_embedding = self.embedding_model.encode([query]).tolist()
                    results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
                else:
                    results = self.collection.query(query_texts=[query], n_results=top_k)

                if results and "documents" in results and results["documents"]:
                    docs = results["documents"][0]
                    metas = results["metadatas"][0]
                    distances = results.get("distances", [[]])[0]
                    ids = results["ids"][0]

                    formatted = []
                    for i in range(len(docs)):
                        formatted.append({
                            "chunk_id": ids[i],
                            "score": float(distances[i]) if i < len(distances) else 0.0,
                            "text": docs[i],
                            "metadata": metas[i]
                        })
                    return formatted
            except Exception:
                pass

        # Fallback Vector Matrix Search
        if not self.fallback_chunks or self.doc_matrix is None:
            self.load_index()

        if not self.fallback_chunks or self.doc_matrix is None:
            return []

        q_toks = self._tokenize(query)
        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        for w in q_toks:
            if w in self.vocab:
                q_vec[self.vocab[w]] += 1.0
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        scores = np.dot(self.doc_matrix, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            c = self.fallback_chunks[idx]
            results.append({
                "chunk_id": c["chunk_id"],
                "score": float(scores[idx]),
                "text": c["text"],
                "metadata": {
                    "document_name": c["document_name"],
                    "file_path": c["file_path"],
                    "page_number": c["page_number"],
                    "start_page": c["start_page"],
                    "end_page": c["end_page"],
                    "section_title": c["section_title"],
                    "token_count": c["token_count"]
                }
            })
        return results


def build_index_from_chunks(chunks_dir: str, db_dir: str):
    """Build persistent vector index from JSON chunk files"""
    json_files = [f for f in os.listdir(chunks_dir) if f.endswith("_chunks.json")]
    all_chunks = []

    for jf in json_files:
        with open(os.path.join(chunks_dir, jf), "r", encoding="utf-8") as f:
            all_chunks.extend(json.load(f))

    print(f"Loaded total {len(all_chunks)} chunks from {len(json_files)} file(s).")
    
    manager = VectorStoreManager(db_dir=db_dir)
    manager.initialize()
    manager.index_chunks(all_chunks)


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    chunks_directory = os.path.join(base_dir, "RAG", "chunks_data")
    vector_db_directory = os.path.join(base_dir, "RAG", "vector_db")

    if os.path.exists(chunks_directory):
        build_index_from_chunks(chunks_directory, vector_db_directory)
    else:
        print(f"Chunks directory '{chunks_directory}' not found.")
