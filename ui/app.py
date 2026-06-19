"""Golgi — Healthcare Intelligence Search: Streamlit UI."""
from __future__ import annotations

import sys
import os
import time

# Make project root importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Golgi — Healthcare Intelligence Search",
    page_icon="🔬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .intent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .answer-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.25rem 1.5rem;
        line-height: 1.7;
        font-size: 0.97rem;
    }
    .source-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.6rem;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.4rem;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Intent helpers ────────────────────────────────────────────────────────────
_INTENT_META = {
    "drug_info":         ("💊", "Drug Information",    "#fef3c7", "#92400e"),
    "clinical_evidence": ("🔬", "Clinical Evidence",   "#dbeafe", "#1e40af"),
    "guidelines":        ("📋", "Clinical Guidelines", "#d1fae5", "#065f46"),
    "research_funding":  ("🏛️", "Research Funding",   "#ede9fe", "#4c1d95"),
    "general":           ("🌐", "General Search",      "#f1f5f9", "#334155"),
}

_SOURCE_COLORS = {
    "PubMed":        ("#dbeafe", "#1e40af"),
    "NIH RePORTER":  ("#d1fae5", "#065f46"),
    "WHO IRIS":      ("#fef3c7", "#92400e"),
    "FDA":           ("#fee2e2", "#991b1b"),
}


def _intent_badge(intent: str) -> str:
    icon, label, bg, fg = _INTENT_META.get(intent, ("🌐", intent.title(), "#f1f5f9", "#334155"))
    return (
        f'<span class="intent-badge" style="background:{bg};color:{fg};">'
        f"{icon} {label}</span>"
    )


def _source_badge(source: str) -> str:
    bg, fg = _SOURCE_COLORS.get(source, ("#f1f5f9", "#334155"))
    return (
        f'<span class="source-badge" style="background:{bg};color:{fg};">'
        f"{source}</span>"
    )


# ── Pipeline (cached so it only loads once per session) ──────────────────────
@st.cache_resource(show_spinner=False)
def _load_pipeline():
    from orchestration.pipeline import GolgiPipeline
    return GolgiPipeline()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filters")
    st.markdown("**Sources to include**")
    src_pubmed = st.checkbox("PubMed", value=True)
    src_nih    = st.checkbox("NIH Reporter", value=True)
    src_who    = st.checkbox("WHO IRIS", value=True)
    src_fda    = st.checkbox("FDA", value=True)

    st.divider()
    st.header("About Golgi")
    st.markdown(
        """
Golgi is an LLM-powered healthcare search engine that retrieves and synthesises
evidence from **50+ sources** including PubMed, NIH, WHO, and FDA.

**Pipeline steps**
1. 🎯 Intent detection (Gemini 2.0 Flash)
2. 📡 Live source fetching
3. 🔍 Hybrid BM25 + FAISS retrieval
4. 🏆 Cross-encoder reranking
5. ✍️ Grounded answer generation

*Sources cited inline — never hallucinated.*
        """
    )


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧬 Golgi</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Healthcare Intelligence Search · PubMed · NIH · WHO · FDA</div>',
    unsafe_allow_html=True,
)

col_input, col_btn = st.columns([6, 1])
with col_input:
    query = st.text_input(
        label="query",
        placeholder="e.g. What are the side effects of metformin?",
        label_visibility="collapsed",
    )
with col_btn:
    search_clicked = st.button("Search", type="primary", use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if search_clicked and query.strip():
    t_start = time.time()

    with st.spinner("Searching 50+ healthcare sources..."):
        pipeline = _load_pipeline()
        result = pipeline.run(query.strip())

    elapsed_ms = round((time.time() - t_start) * 1000)
    chunks_searched = len(result.get("chunks_used", []))
    sources_found   = len(result.get("sources", []))

    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("⏱ Retrieval time", f"{elapsed_ms} ms")
    m2.metric("📄 Chunks searched", chunks_searched)
    m3.metric("📚 Sources found", sources_found)

    st.divider()

    # Intent badge
    st.markdown(_intent_badge(result["intent"]), unsafe_allow_html=True)

    # Answer card
    st.markdown(
        f'<div class="answer-card">{result["answer"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Sources expander
    with st.expander("📚 Sources", expanded=True):
        for i, src in enumerate(result["sources"], 1):
            title  = src.get("title", "Untitled")
            url    = src.get("url", "")
            source = src.get("source", "")
            score  = src.get("relevance_score", 0.0)

            link = f'<a href="{url}" target="_blank">{title}</a>' if url else title
            badge = _source_badge(source)

            st.markdown(
                f'<div class="source-card">'
                f'<strong>[{i}]</strong> {link}{badge}<br>'
                f'<small style="color:#6b7280;">Relevance score: {score:.3f}</small><br>'
                f'<div style="margin-top:0.3rem;background:#e2e8f0;border-radius:9999px;height:6px;">'
                f'<div style="width:{min(score*100,100):.0f}%;background:#3b82f6;height:6px;border-radius:9999px;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # Retrieval trace expander
    with st.expander("🔍 Retrieval trace"):
        for step in result["retrieval_trace"]:
            st.markdown(f"- {step}")

elif search_clicked and not query.strip():
    st.warning("Please enter a search query.")
