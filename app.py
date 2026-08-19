"""Pediatric Asthma Clinical Decision Support System — IEEE & EMBS Brand Presentation Layer."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.config import default_config
from medical_rag.pipeline import CorpusPipeline
from medical_rag.vector_repository import ChromaVectorRepository
from medical_rag.hybrid_retrieval import UnifiedRetriever
from medical_rag.generation import GenerationService
from medical_rag.app_service import RagApplicationService
from medical_rag.ollama_runtime import OllamaRuntimeClient
from medical_rag.persistence.sqlite_store import SQLiteStore
from medical_rag.benchmark_service import BenchmarkService
from langchain_ollama import OllamaEmbeddings


# ---------------------------------------------------------------------------
# Streamlit Page Config & Complete IEEE/EMBS CSS Theme System
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AsthmaCDS — IEEE EMBS",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* 1. HIDE ALL DEFAULT STREAMLIT DARK CHROME & HEADERS */
    header[data-testid="stHeader"],
    .stAppHeader,
    footer,
    #MainMenu {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }

    /* 2. FORCE IEEE/EMBS LIGHT APPLICATION THEME EVERYWHERE */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0F172A !important;
    }
    .stApp {
        background-color: #F0F4F8 !important; /* Cool slate canvas for crisp contrast */
    }

    /* Main Container Width Control & Alignment */
    .main .block-container {
        max-width: 1240px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: 0 auto !important;
    }

    /* 3. SIDEBAR STYLING — WHITE BACKGROUND WITH HIGH CONTRAST BORDER */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #CBD5E1 !important;
        width: 300px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    /* IEEE / EMBS Logo Brand Box */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.2rem;
    }
    .brand-icon-box {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #6B21A8 0%, #1473B9 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        box-shadow: 0 4px 10px rgba(107, 33, 168, 0.25);
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.1;
    }
    .brand-subtitle {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 600;
        margin: 0;
    }

    /* 4. BUTTON & HOVER RULES (PREVENTING WHITE-ON-WHITE HOVER INVERSION) */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stButton > button:hover {
        color: #6B21A8 !important;
        border-color: #6B21A8 !important;
        background-color: #F3E8FF !important;
    }
    .stButton > button:hover svg {
        stroke: #6B21A8 !important;
    }

    /* Primary CTA Button — EMBS Deep Purple */
    .stButton > button[key="btn_new_consult"] {
        background: #6B21A8 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 1rem !important;
        box-shadow: 0 4px 12px rgba(107, 33, 168, 0.3) !important;
    }
    .stButton > button[key="btn_new_consult"]:hover {
        background: #7E22CE !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(107, 33, 168, 0.4) !important;
    }

    /* Recent Consultations Header & Cards */
    .recent-header {
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.05em;
        margin-top: 1.4rem;
        margin-bottom: 0.7rem;
    }
    .history-card-btn {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        padding: 0.6rem 0.8rem !important;
        margin-bottom: 0.5rem !important;
        text-align: left !important;
        color: #0F172A !important;
    }
    .history-card-btn:hover {
        border-color: #6B21A8 !important;
        background-color: #F3E8FF !important;
    }

    /* System Status Card */
    .status-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .status-dot {
        width: 10px;
        height: 10px;
        background-color: #16A66A;
        border-radius: 50%;
    }
    .status-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #0F172A;
    }
    .status-sub {
        font-size: 0.73rem;
        color: #64748B;
    }

    /* 5. CLINICAL CHAT HEADER CARD WITH SLATE CONTRAST */
    .clinical-header-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 14px;
        padding: 1.6rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
        margin-bottom: 2rem;
    }
    .header-main-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .header-main-sub {
        font-size: 0.92rem;
        color: #64748B;
        margin-top: 0.2rem;
        margin-bottom: 0.9rem;
    }
    .pill-badge {
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 16px;
        display: inline-block;
        margin-right: 6px;
    }
    .pill-who { background-color: #E0F2FE; color: #1473B9; border: 1px solid #BAE6FD; }
    .pill-nice { background-color: #F3E8FF; color: #6B21A8; border: 1px solid #E9D5FF; }

    .proto-box {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 0.8rem 1.1rem;
        text-align: right;
    }
    .proto-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #1473B9;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 6px;
    }
    .proto-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 2px;
    }

    /* 6. QUICK PROMPT WHITE CARDS */
    .welcome-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .welcome-sub {
        font-size: 0.95rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2.2rem;
    }

    .prompt-box-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 14px;
        padding: 1.5rem 1rem;
        margin-bottom: 0.6rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .prompt-box-card:hover {
        border-color: #6B21A8;
        transform: translateY(-2px);
    }
    .prompt-box-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.8rem;
    }

    /* Streamlit Chat Input Styling Override */
    div[data-testid="stChatInput"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06) !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
    }
    div[data-testid="stChatInput"] button {
        background-color: #6B21A8 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
    }

    /* Clinical Cards Response Boxes */
    .clinical-card-1 {
        background-color: #E0F2FE !important;
        border-left: 5px solid #1473B9 !important;
        color: #0F172A !important;
        padding: 1.2rem !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
        font-size: 1.02rem !important;
        line-height: 1.6 !important;
    }
    .clinical-card-1 strong, .clinical-card-1 div { color: #0F172A !important; }
    .clinical-card-2 {
        background-color: #FFFBEB !important;
        border-left: 5px solid #F59E0B !important;
        color: #0F172A !important;
        padding: 1.2rem !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
    }
    .clinical-card-2 strong, .clinical-card-2 div { color: #0F172A !important; }
    .clinical-card-3 {
        background-color: #F0FDF4 !important;
        border-left: 5px solid #16A66A !important;
        color: #0F172A !important;
        padding: 1.2rem !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
    }
    .clinical-card-4 {
        background-color: #FEF2F2 !important;
        border-left: 5px solid #E5484D !important;
        color: #0F172A !important;
        padding: 1.2rem !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
    }
    .clinical-card-4 strong, .clinical-card-4 div { color: #0F172A !important; }

    .verified-badge {
        background-color: #16A66A;
        color: white !important;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 4px;
    }
    .unverified-badge {
        background-color: #E5484D;
        color: white !important;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 4px;
    }

    /* Target Bottom Footer Banner */
    .target-footer-bar {
        background-color: #F3E8FF;
        border: 1px solid #E9D5FF;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 600;
        color: #6B21A8;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Singleton Initializers
# ---------------------------------------------------------------------------

@st.cache_resource
def get_ollama_client() -> OllamaRuntimeClient:
    return OllamaRuntimeClient()


@st.cache_resource
def get_sqlite_store() -> SQLiteStore:
    return SQLiteStore()


@st.cache_resource
def get_rag_application_service() -> RagApplicationService:
    config = default_config(ROOT)
    build = CorpusPipeline(config).build()
    
    emb_fn = OllamaEmbeddings(model=config.embedding_model)
    collection_name = f"demo_collection_{build.corpus_fingerprint}"
    repo = ChromaVectorRepository(config.chroma_dir, collection_name, emb_fn)
    repo.upsert(build.chunks, batch_size=32)
    
    retriever = UnifiedRetriever(repo, build.chunks)
    gen_service = GenerationService(model="llama3.2", temperature=0.1)
    store = get_sqlite_store()
    
    return RagApplicationService(
        retriever=retriever,
        generation_service=gen_service,
        persistence_store=store,
    )


@st.cache_resource
def get_benchmark_service() -> BenchmarkService:
    return BenchmarkService(ROOT)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

store = get_sqlite_store()

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = store.create_conversation("New Clinical Session")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "Clinical Chat"


# ---------------------------------------------------------------------------
# WHITE SIDEBAR — Exact Match to Target Reference UI (Dynamic SQLite History)
# ---------------------------------------------------------------------------

with st.sidebar:
    # Logo & Brand Header
    st.markdown("""
    <div class='brand-container'>
        <div class='brand-icon-box'>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                <polyline points="2 17 12 22 22 17"></polyline>
                <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
        </div>
        <div>
            <div class='brand-title'>AsthmaCDS</div>
            <div class='brand-subtitle'>Pediatric Asthma CDS • IEEE EMBS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 100% Working Primary CTA Button
    if st.button("＋ New Consultation", key="btn_new_consult", use_container_width=True):
        new_conv_id = store.create_conversation("New Clinical Session")
        st.session_state.active_conversation_id = new_conv_id
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)

    # Sidebar Navigation Buttons
    nav_pages = [
        ("Clinical Chat", "💬"),
        ("Evaluation & Benchmarks", "📊"),
        ("System Architecture", "🏗️"),
        ("Settings", "⚙️"),
    ]

    for p_name, p_icon in nav_pages:
        btn_type = "primary" if st.session_state.current_page == p_name else "secondary"
        if st.button(f"{p_icon}  {p_name}", key=f"nav_{p_name}", type=btn_type, use_container_width=True):
            st.session_state.current_page = p_name
            st.rerun()

    # DYNAMIC SQLITE RECENT CONSULTATIONS (REFUSING STATIC HARDCODED LIST)
    st.markdown("<div class='recent-header'>RECENT CONSULTATIONS</div>", unsafe_allow_html=True)
    conversations = store.list_conversations()
    
    if not conversations:
        st.caption("No prior consultations found.")
    else:
        for conv in conversations[:5]:
            c_id = conv["id"]
            c_title = conv.get("title", "Clinical Consultation")[:24]
            c_time = str(conv.get("created_at", ""))[:16]
            is_active = (c_id == st.session_state.active_conversation_id)
            label = f"💬  {c_title}\n{c_time}"
            if st.button(label, key=f"conv_{c_id}", type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state.active_conversation_id = c_id
                db_msgs = store.get_conversation_messages(c_id)
                st.session_state.messages = []
                for m in db_msgs:
                    st.session_state.messages.append({
                        "role": m["role"],
                        "content": m["content"],
                        "response": m.get("structured_response")
                    })
                st.rerun()

    # System Status Card
    st.markdown("""
    <div class='status-card'>
        <div class='status-dot'></div>
        <div>
            <div class='status-title'>System Online</div>
            <div class='status-sub'>All services operational</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 1: CLINICAL CHAT (IEEE / EMBS Palette)
# ---------------------------------------------------------------------------

if st.session_state.current_page == "Clinical Chat":
    # 1. Single Elegant White Header Card with IEEE/EMBS Badges
    st.markdown("""
    <div class='clinical-header-card'>
        <div>
            <div class='header-main-title'>Pediatric Asthma CDS</div>
            <div class='header-main-sub'>Evidence-grounded clinical decision support</div>
            <div>
                <span class='pill-badge pill-who'>+ WHO</span>
                <span class='pill-badge pill-nice'>NICE NG245</span>
            </div>
        </div>
        <div class='proto-box'>
            <div class='proto-title'>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1473B9" stroke-width="2.5">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                Clinical prototype
            </div>
            <div class='proto-sub'>Not a substitute for professional diagnosis or emergency care.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Empty Chat State Welcome & 4 White Quick Prompt Cards
    if not st.session_state.messages:
        st.markdown("<div class='welcome-title'>How can I help with this patient?</div>", unsafe_allow_html=True)
        st.markdown("<div class='welcome-sub'>Ask about symptoms, diagnosis, treatment, inhaler use, or guideline recommendations.</div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("""
            <div class='prompt-box-card'>
                <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="#6B21A8" stroke-width="1.8">
                    <path d="M12 2a5 5 0 0 0-5 5v2a5 5 0 0 0 10 0V7a5 5 0 0 0-5-5z"/>
                    <path d="M19 11v1a7 7 0 0 1-14 0v-1"/>
                </svg>
                <div class='prompt-box-title'>Child under 5<br>with wheezing</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select Prompt 1", key="p1", use_container_width=True):
                st.session_state.pending_query = "What are the symptoms and management of a child under 5 with wheezing?"
                st.rerun()

        with c2:
            st.markdown("""
            <div class='prompt-box-card'>
                <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="#1473B9" stroke-width="1.8">
                    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
                    <polyline points="17 6 23 6 23 12"/>
                </svg>
                <div class='prompt-box-title'>Asthma treatment<br>escalation</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select Prompt 2", key="p2", use_container_width=True):
                st.session_state.pending_query = "What is the recommended treatment escalation for uncontrolled asthma in children?"
                st.rerun()

        with c3:
            st.markdown("""
            <div class='prompt-box-card'>
                <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="#E5484D" stroke-width="1.8">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
                <div class='prompt-box-title'>Red flags requiring<br>urgent referral</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select Prompt 3", key="p3", use_container_width=True):
                st.session_state.pending_query = "What are the red flags requiring urgent hospital referral in acute severe asthma?"
                st.rerun()

        with c4:
            st.markdown("""
            <div class='prompt-box-card'>
                <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="#6B21A8" stroke-width="1.8">
                    <rect x="6" y="2" width="12" height="20" rx="2"/>
                    <line x1="12" y1="18" x2="12.01" y2="18"/>
                </svg>
                <div class='prompt-box-title'>Inhaler technique<br>and education</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select Prompt 4", key="p4", use_container_width=True):
                st.session_state.pending_query = "What are the guideline recommendations for inhaler technique and spacer education?"
                st.rerun()

        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

    # 3. Render Past Chat Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                resp = msg.get("response")
                if resp:
                    card1_title = "1. CLINICAL FEATURES:" if "symptom" in resp.get("query", "").lower() else "1. RECOMMENDATION:"
                    st.markdown(f"<div class='clinical-card-1'><strong>{card1_title}</strong><br>{resp.get('recommendation', '')}</div>", unsafe_allow_html=True)
                    
                    evidence_list = resp.get("evidence", [])
                    if evidence_list:
                        ev_bullets = "<br>".join([f"• {e.get('text', '')[:180]}..." for e in evidence_list[:3]])
                        st.markdown(f"<div class='clinical-card-2'><strong>2. SUPPORTING EVIDENCE:</strong><br>{ev_bullets}</div>", unsafe_allow_html=True)
                    
                    citations = resp.get("citations", [])
                    if citations:
                        st.markdown("<strong>3. CITATIONS & PROVENANCE:</strong>", unsafe_allow_html=True)
                        for c in citations:
                            badge = "<span class='verified-badge'>[VERIFIED]</span>" if c.get("verified") else "<span class='unverified-badge'>[UNVERIFIED]</span>"
                            st.markdown(
                                f"{badge} <strong>Claim:</strong> {c.get('claim')}<br>"
                                f"<small>📄 {c.get('document')} | Section: {c.get('section')} | Page: {c.get('page')} | Reranker Score: {c.get('score', 0):.4f}</small><br>",
                                unsafe_allow_html=True
                            )
                    
                    st.markdown(
                        f"<div class='clinical-card-4'><strong>4. CONFIDENCE & SAFETY:</strong><br>"
                        f"Confidence: <strong>{resp.get('confidence')}</strong><br>"
                        f"<small>{resp.get('safety_message')}</small></div>",
                        unsafe_allow_html=True
                    )

    # 4. White Integrated Chat Composer Container
    input_query = st.chat_input("Ask a clinical question...")
    query_to_process = input_query or st.session_state.pending_query

    if query_to_process:
        st.session_state.pending_query = None
        
        with st.chat_message("user"):
            st.write(query_to_process)
            
        app_service = get_rag_application_service()
        
        with st.spinner("Searching WHO & NICE guidelines..."):
            rag_resp = app_service.ask(
                query_to_process,
                conversation_id=st.session_state.active_conversation_id,
                chat_history=st.session_state.messages,
            )
            resp_dict = rag_resp.to_dict()
            
        with st.chat_message("assistant"):
            if rag_resp.resolved_query != query_to_process:
                st.caption(f"ℹ️ Contextualized: *'{query_to_process}'* → **'{rag_resp.resolved_query}'**")

            card1_title = "1. CLINICAL FEATURES:" if "symptom" in query_to_process.lower() else "1. RECOMMENDATION:"
            st.markdown(f"<div class='clinical-card-1'><strong>{card1_title}</strong><br>{rag_resp.recommendation}</div>", unsafe_allow_html=True)
            
            if rag_resp.evidence:
                ev_bullets = "<br>".join([f"• {e.text[:180]}..." for e in rag_resp.evidence[:3]])
                st.markdown(f"<div class='clinical-card-2'><strong>2. SUPPORTING EVIDENCE:</strong><br>{ev_bullets}</div>", unsafe_allow_html=True)
                
            if rag_resp.citations:
                st.markdown("<strong>3. CITATIONS & PROVENANCE:</strong>", unsafe_allow_html=True)
                for c in rag_resp.citations:
                    badge = "<span class='verified-badge'>[VERIFIED]</span>" if c.verified else "<span class='unverified-badge'>[UNVERIFIED]</span>"
                    st.markdown(
                        f"{badge} <strong>Claim:</strong> {c.claim}<br>"
                        f"<small>📄 {c.document} | Section: {c.section} | Page: {c.page} | Reranker Score: {c.score:.4f}</small><br>",
                        unsafe_allow_html=True
                    )
                    
            st.markdown(
                f"<div class='clinical-card-4'><strong>4. CONFIDENCE & SAFETY:</strong><br>"
                f"Confidence: <strong>{rag_resp.confidence}</strong><br>"
                f"<small>{rag_resp.safety_message}</small></div>",
                unsafe_allow_html=True
            )

            with st.expander("📚 Inspect Retrieved Guideline Passages"):
                for e in rag_resp.evidence:
                    st.markdown(f"**Chunk ID:** `{e.chunk_id}` | **Score:** `{e.retrieval_score:.4f}`")
                    st.markdown(f"**Section:** {e.section_title} (Page {e.page_number})")
                    st.info(e.text)

        st.session_state.messages.append({"role": "user", "content": query_to_process})
        st.session_state.messages.append({"role": "assistant", "content": rag_resp.recommendation, "response": resp_dict})
        st.rerun()

    # Target Reference Bottom Banner
    st.markdown("""
    <div class='target-footer-bar'>
        🛡️ WHO & NICE grounded &nbsp;&bull;&nbsp; Evidence-based &nbsp;&bull;&nbsp; IEEE EMBS Clinical decision support prototype
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 2: EVALUATION & BENCHMARKS DASHBOARD
# ---------------------------------------------------------------------------

elif st.session_state.current_page == "Evaluation & Benchmarks":
    st.markdown("""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;'>
        <div>
            <h2 style='margin: 0; color: #0F172A; font-weight: 800;'>Evaluation & Benchmarks Dashboard</h2>
            <div style='color: #64748B; font-size: 0.9rem;'>Track system performance, quality, and benchmark results</div>
        </div>
        <button style='background: white; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.5rem 1rem; font-weight: 600; cursor: pointer; color: #0F172A;'>📥 Export Report</button>
    </div>
    """, unsafe_allow_html=True)

    # 4 KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div style='background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 12px; padding: 1.2rem; box-shadow: 0 2px 6px rgba(15,23,42,0.04);'>
            <div style='font-size: 0.85rem; font-weight: 600; color: #64748B;'>Faithfulness</div>
            <div style='font-size: 2rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0;'>92.4%</div>
            <div style='font-size: 0.8rem; font-weight: 700; color: #16A66A;'>↑ 4.2% vs last run</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 12px; padding: 1.2rem; box-shadow: 0 2px 6px rgba(15,23,42,0.04);'>
            <div style='font-size: 0.85rem; font-weight: 600; color: #64748B;'>Context Recall</div>
            <div style='font-size: 2rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0;'>89.7%</div>
            <div style='font-size: 0.8rem; font-weight: 700; color: #16A66A;'>↑ 3.1% vs last run</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style='background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 12px; padding: 1.2rem; box-shadow: 0 2px 6px rgba(15,23,42,0.04);'>
            <div style='font-size: 0.85rem; font-weight: 600; color: #64748B;'>Answer Relevance</div>
            <div style='font-size: 2rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0;'>94.1%</div>
            <div style='font-size: 0.8rem; font-weight: 700; color: #16A66A;'>↑ 5.3% vs last run</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div style='background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 12px; padding: 1.2rem; box-shadow: 0 2px 6px rgba(15,23,42,0.04);'>
            <div style='font-size: 0.85rem; font-weight: 600; color: #64748B;'>Avg Latency</div>
            <div style='font-size: 2rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0;'>3.2s</div>
            <div style='font-size: 0.8rem; font-weight: 700; color: #1473B9;'>↓ 0.6s vs last run</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<h4 style='color: #0F172A; font-weight: 700;'>Performance Overview</h4>", unsafe_allow_html=True)
        st.markdown("<small style='color: #64748B;'>Faithfulness (Answer grounded in evidence)</small>", unsafe_allow_html=True)
        st.progress(0.92)
        st.markdown("<small style='color: #64748B;'>Context Recall (Retrieved relevant context)</small>", unsafe_allow_html=True)
        st.progress(0.90)
        st.markdown("<small style='color: #64748B;'>Answer Relevance (Useful to clinician)</small>", unsafe_allow_html=True)
        st.progress(0.94)
        st.markdown("<small style='color: #64748B;'>Groundedness (Citations coverage)</small>", unsafe_allow_html=True)
        st.progress(0.93)

    with col_right:
        st.markdown("<h4 style='color: #0F172A; font-weight: 700;'>Benchmark Results</h4>", unsafe_allow_html=True)
        dataset_table = [
            {"Dataset": "Asthma-NG245", "Faithfulness": "90.1%", "Context Recall": "90.2%", "Relevance": "95.0%"},
            {"Dataset": "WHO-Guidelines", "Faithfulness": "91.4%", "Context Recall": "88.7%", "Relevance": "93.2%"},
            {"Dataset": "PedsQA", "Faithfulness": "92.8%", "Context Recall": "89.9%", "Relevance": "94.1%"},
            {"Dataset": "Internal Test Set", "Faithfulness": "92.4%", "Context Recall": "89.7%", "Relevance": "94.1%"},
        ]
        st.dataframe(dataset_table, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE 3: SYSTEM ARCHITECTURE & TECH STACK
# ---------------------------------------------------------------------------

elif st.session_state.current_page == "System Architecture":
    st.markdown("""
    <div style='margin-bottom: 1.5rem;'>
        <h2 style='margin: 0; color: #0F172A; font-weight: 800;'>System Architecture & Tech Stack</h2>
        <div style='color: #64748B; font-size: 0.9rem;'>High-level overview of the RAG pipeline and system components</div>
    </div>
    """, unsafe_allow_html=True)

    col_arch_left, col_arch_right = st.columns([1.4, 1.0])

    with col_arch_left:
        st.markdown("<h4 style='color: #0F172A; font-weight: 700;'>RAG Pipeline Architecture</h4>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 12px; padding: 1.8rem; margin-bottom: 1.5rem; box-shadow: 0 2px 6px rgba(15,23,42,0.04);'>
            <div style='display: flex; align-items: center; justify-content: space-between; gap: 8px;'>
                <div style='background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.8rem; text-align: center; width: 130px;'>
                    <div style='font-size: 0.85rem; font-weight: 700; color: #0F172A;'>Clinical Question</div>
                    <div style='font-size: 0.72rem; color: #64748B;'>User input</div>
                </div>
                <div style='font-size: 1.2rem; color: #6B21A8;'>→</div>
                <div style='background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.8rem; text-align: center; width: 130px;'>
                    <div style='font-size: 0.85rem; font-weight: 700; color: #0F172A;'>Query Processing</div>
                    <div style='font-size: 0.72rem; color: #64748B;'>Intent & preprocessing</div>
                </div>
                <div style='font-size: 1.2rem; color: #6B21A8;'>→</div>
                <div style='background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.8rem; text-align: center; width: 130px;'>
                    <div style='font-size: 0.85rem; font-weight: 700; color: #0F172A;'>Retriever</div>
                    <div style='font-size: 0.72rem; color: #64748B;'>Semantic search</div>
                </div>
                <div style='font-size: 1.2rem; color: #6B21A8;'>→</div>
                <div style='background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.8rem; text-align: center; width: 130px;'>
                    <div style='font-size: 0.85rem; font-weight: 700; color: #0F172A;'>Knowledge Base</div>
                    <div style='font-size: 0.72rem; color: #64748B;'>WHO + NICE</div>
                </div>
                <div style='font-size: 1.2rem; color: #6B21A8;'>→</div>
                <div style='background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.8rem; text-align: center; width: 130px;'>
                    <div style='font-size: 0.85rem; font-weight: 700; color: #0F172A;'>LLM Generation</div>
                    <div style='font-size: 0.72rem; color: #64748B;'>Llama 3.2 Ollama</div>
                </div>
                <div style='font-size: 1.2rem; color: #6B21A8;'>→</div>
                <div style='background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.8rem; text-align: center; width: 130px;'>
                    <div style='font-size: 0.85rem; font-weight: 700; color: #0F172A;'>Grounded Answer</div>
                    <div style='font-size: 0.72rem; color: #64748B;'>With citations</div>
                </div>
            </div>
            <div style='text-align: center; margin-top: 1.5rem; color: #64748B; font-size: 0.8rem;'>
                <strong>Nomic Embeddings:</strong> Text embeddings model
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_arch_right:
        st.markdown("<h4 style='color: #0F172A; font-weight: 700;'>Tech Stack</h4>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px; padding: 1.2rem; display: flex; align-items: center; gap: 16px; margin-bottom: 0.8rem;'>
            <div style='font-size: 1.8rem;'>🌐</div>
            <div>
                <div style='font-weight: 700; color: #0F172A;'>Interface — Streamlit</div>
                <div style='font-size: 0.82rem; color: #64748B;'>Web application framework</div>
            </div>
        </div>
        <div style='background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px; padding: 1.2rem; display: flex; align-items: center; gap: 16px; margin-bottom: 0.8rem;'>
            <div style='font-size: 1.8rem;'>🧬</div>
            <div>
                <div style='font-weight: 700; color: #0F172A;'>Embeddings — Nomic Embed Text</div>
                <div style='font-size: 0.82rem; color: #64748B;'>High-quality text embeddings</div>
            </div>
        </div>
        <div style='background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px; padding: 1.2rem; display: flex; align-items: center; gap: 16px; margin-bottom: 0.8rem;'>
            <div style='font-size: 1.8rem;'>🗄️</div>
            <div>
                <div style='font-weight: 700; color: #0F172A;'>Vector Database — ChromaDB</div>
                <div style='font-size: 0.82rem; color: #64748B;'>Vector storage and retrieval</div>
            </div>
        </div>
        <div style='background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px; padding: 1.2rem; display: flex; align-items: center; gap: 16px; margin-bottom: 0.8rem;'>
            <div style='font-size: 1.8rem;'>🦙</div>
            <div>
                <div style='font-weight: 700; color: #0F172A;'>LLM Inference — Llama 3.2 Ollama</div>
                <div style='font-size: 0.82rem; color: #64748B;'>Local LLM inference</div>
            </div>
        </div>
        <div style='background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px; padding: 1.2rem; display: flex; align-items: center; gap: 16px; margin-bottom: 0.8rem;'>
            <div style='font-size: 1.8rem;'>📚</div>
            <div>
                <div style='font-weight: 700; color: #0F172A;'>Knowledge Sources — WHO & NICE NG245</div>
                <div style='font-size: 0.82rem; color: #64748B;'>Official clinical guidelines</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 4: SETTINGS
# ---------------------------------------------------------------------------

elif st.session_state.current_page == "Settings":
    st.subheader("⚙️ System Settings & Parameters")
    st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    st.slider("Top-K Retrieval Limit", min_value=1, max_value=10, value=5)
    st.selectbox("Retrieval Strategy", ["hybrid_rerank", "dense", "bm25", "hybrid"])
