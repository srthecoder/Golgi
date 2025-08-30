#!/usr/bin/env python3
from __future__ import annotations
import os, re, requests, streamlit as st
from exa_py import Exa
from readability import Document
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

ALLOW = [
    "nih.gov","ncbi.nlm.nih.gov","pubmed.ncbi.nlm.nih.gov","cdc.gov","who.int",
    "nejm.org","jamanetwork.com","thelancet.com","bmj.com","nature.com",
    "fda.gov","ema.europa.eu","clinicaltrials.gov","cochranelibrary.com",
]
# EXCLUDE kept for future non-content searches; not used with `text=` to avoid 400s.
EXCLUDE = ["wikipedia.org","twitter.com","x.com","facebook.com","reddit.com","medium.com","quora.com","linkedin.com","youtube.com"]

def _fetch(url: str, timeout=10) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Golgi/1.0"}); r.raise_for_status()
        return r.text
    except Exception:
        return ""

def _clean(html: str) -> str:
    if not html: return ""
    try:
        soup = BeautifulSoup(Document(html).summary(html_partial=True), "html.parser")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script","style","noscript"]): t.extract()
    return re.sub(r"\s+"," ", soup.get_text(" ").strip())

def _topk(query: str, text: str, k: int = 5) -> list[str]:
    if not text: return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    toks = [re.findall(r"\w+", s.lower()) for s in sents]
    if not toks: return []
    bm25 = BM25Okapi(toks)
    qtok = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(qtok)
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [sents[i] for i in idx]

def _get_exa_key() -> str:
    key = os.getenv("EXA_API_KEY") or st.secrets.get("EXA_API_KEY", "")
    if not key:
        st.error("Missing EXA_API_KEY (env var or .streamlit/secrets.toml)."); st.stop()
    return key

@st.cache_resource
def _exa() -> Exa:
    return Exa(api_key=_get_exa_key())

@st.cache_data(show_spinner=False)
def exa_search(query: str, num: int, since: str | None, use_filters: bool):
    exa = _exa()
    opts = dict(
        query=query,
        num_results=num,
        type="auto",
        use_autoprompt=True,
        text={"max_characters": 5000},   # <-- requesting content
    )
    if since: opts["start_published_date"] = since
    # IMPORTANT: with `text`, pass ONLY ONE of includeDomains/excludeDomains.
    if use_filters:
        opts["include_domains"] = ALLOW  # do NOT set exclude_domains here

    try:
        res = exa.search_and_contents(**opts)
        return res.results
    except Exception as e:
        # Fallback: no content (no `text`), allow exclude_domains
        st.warning(f"Exa error (with content): {e}")
        opts.pop("text", None)
        if not use_filters:
            opts["exclude_domains"] = EXCLUDE
        try:
            res = exa.search(**opts)  # returns results without content
            return res.results
        except Exception as e2:
            st.error(f"Exa failed: {e2}")
            return []

def render_results(q: str, results: list):
    if not results:
        st.info("No results."); return
    for r in results:
        url = getattr(r, "url", "")
        title = getattr(r, "title", "") or url
        published = getattr(r, "published_date", None)
        text = getattr(r, "text", None)
        if not text:  # if fallback path returned no content, fetch-clean
            text = _clean(_fetch(url))
        snippet = " ".join(_topk(q, text, 5)) if text else ""
        with st.container(border=True):
            st.markdown(f"**[{title}]({url})**")
            if published: st.caption(str(published))
            if snippet: st.write(snippet)

st.set_page_config(page_title="Golgi — Healthcare Evidence Search", page_icon="🧬", layout="centered")
st.title("Golgi — Healthcare Evidence Search")

q = st.text_input("Query", placeholder="SGLT2 inhibitors CKD 2024 guideline")
c1, c2, c3 = st.columns([1,1,1])
with c1: since = st.text_input("Since (YYYY[-MM[-DD]])", "")
with c2: k = st.number_input("Results", 1, 25, 8, 1)
with c3: use_filters = st.checkbox("Healthcare filters", True)

if st.button("Search", type="primary", use_container_width=True):
    if not q.strip():
        st.warning("Enter a query.")
    else:
        with st.spinner("Searching…"):
            rows = exa_search(q.strip(), num=int(k), since=since.strip() or None, use_filters=use_filters)
        render_results(q, rows)