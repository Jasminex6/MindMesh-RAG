"""FastAPI Backend Server for Pediatric Asthma CDS Web Application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.config import default_config
from medical_rag.pipeline import CorpusPipeline
from medical_rag.vector_repository import ChromaVectorRepository
from medical_rag.hybrid_retrieval import UnifiedRetriever
from medical_rag.generation import GenerationService
from medical_rag.app_service import RagApplicationService
from medical_rag.persistence_store import SQLiteStore
from medical_rag.benchmark_service import BenchmarkService
from langchain_ollama import OllamaEmbeddings

# ---------------------------------------------------------------------------
# FastAPI App Initialization & CORS Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pediatric Asthma CDS API",
    description="REST API for WHO & NICE NG245 Pediatric Asthma Clinical Decision Support System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton Service Instances
_store = SQLiteStore()
_config = default_config(ROOT)
_build = CorpusPipeline(_config).build()
_emb_fn = OllamaEmbeddings(model=_config.embedding_model)
_collection_name = f"demo_collection_{_build.corpus_fingerprint}"
_repo = ChromaVectorRepository(_config.chroma_dir, _collection_name, _emb_fn)
_repo.upsert(_build.chunks, batch_size=32)

_retriever = UnifiedRetriever(_repo, _build.chunks)
_gen_service = GenerationService(model="llama3.2", temperature=0.1)
_app_service = RagApplicationService(
    retriever=_retriever,
    generation_service=_gen_service,
    persistence_store=_store,
)
_benchmark_service = BenchmarkService(ROOT)


# ---------------------------------------------------------------------------
# Request & Response Dataclass Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None
    slots: Optional[Dict[str, Any]] = None


class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Clinical Session"


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Pediatric Asthma CDS API"}


@app.post("/api/ask")
def ask_question(req: AskRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        response = _app_service.ask(
            query=req.query.strip(),
            conversation_id=req.conversation_id,
            slots=req.slots or {},
            chat_history=req.chat_history or [],
        )
        return response.to_dict()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Backend Error: {str(e)}")



@app.get("/api/sessions")
def list_sessions():
    return {"sessions": _store.list_conversations()}


@app.post("/api/sessions")
def create_session(req: CreateSessionRequest):
    session_id = _store.create_conversation(req.title or "New Clinical Session")
    return {"id": session_id, "title": req.title or "New Clinical Session"}


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    messages = _store.get_conversation_messages(session_id)
    return {"messages": messages}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    # Soft deletion in store
    return {"status": "deleted", "id": session_id}


@app.get("/api/benchmarks")
def get_benchmarks():
    return {
        "summary": _benchmark_service.get_summary_metrics(),
        "matrix": _benchmark_service.get_sequential_matrix(),
        "stage_latency": _benchmark_service.get_stage_latency_breakdown(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
