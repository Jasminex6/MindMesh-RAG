"""
Master Pipeline Script for Day 1 Requirements:
1. PDF Parsing & Text Cleaning (preserves page numbers, removes headers/footers/artifacts)
2. Section-Aware Chunking (400-800 tokens with metadata schema)
3. Vector Indexing into ChromaDB / Vector Store
"""

import os
import sys
import json
import time

# Ensure RAG module path is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pdf_parser import process_all_sources
from chunker import process_all_chunks
from vector_store import build_index_from_chunks

def run_day1_pipeline():
    start_time = time.time()
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    sources_dir = os.path.join(base_dir, "Docs", "Sources")
    parsed_dir = os.path.join(current_dir, "parsed_data")
    chunks_dir = os.path.join(current_dir, "chunks_data")
    vector_db_dir = os.path.join(current_dir, "vector_db")

    print("=" * 70)
    print("AI Clinical Decision Support Lite - Day 1 Ingestion Pipeline")
    print("=" * 70)
    print(f"1. Sources Directory: {sources_dir}")
    print(f"2. Cleaned Output Directory: {parsed_dir}")
    print(f"3. Chunked Data Directory: {chunks_dir}")
    print(f"4. Vector DB Store Directory: {vector_db_dir}")
    print("=" * 70)

    # Step 1: PDF Parsing & Text Cleaning
    print("\n[STEP 1/3] Parsing PDF Documents & Cleaning Text...")
    parsed_results = process_all_sources(sources_dir, parsed_dir)
    print(f"Step 1 Complete. Parsed {len(parsed_results)} document(s).")

    # Step 2: Section-Aware Chunking
    print("\n[STEP 2/3] Performing Section-Aware Chunking (400-800 tokens)...")
    all_chunks = process_all_chunks(parsed_dir, chunks_dir)
    print(f"Step 2 Complete. Generated {len(all_chunks)} chunk(s) across all documents.")

    # Step 3: Vector Indexing & Metadata Storage
    print("\n[STEP 3/3] Building Vector Database & Indexing Metadata...")
    build_index_from_chunks(chunks_dir, vector_db_dir)
    print(f"Step 3 Complete. Vector database ready at: {vector_db_dir}")

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"SUCCESS: Day 1 Ingestion Pipeline Executed in {elapsed} seconds!")
    print("=" * 70)

if __name__ == "__main__":
    run_day1_pipeline()
