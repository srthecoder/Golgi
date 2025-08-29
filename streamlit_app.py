#!/usr/bin/env python3
from __future__ import annotations
import os, re, requests
import streamlit as st
from exa_py import Exa
from readability import Document
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

HEALTH_ALLOW = [
    "nih.gov","ncbi.nlm.nih.gov","pubmed.ncbi.nlm.nih.gov","cdc.gov","who.int",
    "nejm.org","jamanetwork.com","thelancet.com","bmj.com","nature.com",
    "fda.gov","ema.europa.eu","clinicaltrials.gov","cochranelibrary.com",
]
HEALTH_EXCLUDE = ["wikipedia.org","twitter.com","x.com","facebook.com","reddit.com",
                  "medium.com","quora.com","linkedin.com","youtube.com"]

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

@st.cache_resource
def _exa_client():
    key = st.secrets.get("EXA_API_KEY", os.getenv("EXA_API_KEY", ""))
    if not key:
        st.stop()
    return Exa(api_key=key)

@st.cache_data(show_spinner=False)
def exa_search(query: str, num: int, since: str | None):
    exa = _exa_client()
    resp = exa.search_and_contents(
        query=query, num_results=num,
        include_domains=HEALTH_ALLOW, exclude_domains=HEALTH_EXCLUDE,
        start_published_date=since or None,
        type="neural", use_autoprompt=True,
        text={"max_characters": 5000},
    )
    out = []
    for r in resp.results:
        url = getattr(r, "url", "")
        title = getattr(r, "title", "") or url
        published = getattr(r, "published_date", None)
        txt = getattr(r, "text", None) or _clean(_fetch(url))
        snippet = " ".join(_topk(query, txt, 5)) if txt else ""
        out.append({"title": title, "url": url, "published": published, "summary": snippet})
    return out

# --- UI ---
st.set_page_config(page_title="Golgi — Healthcare Evidence Search", page_icon="🧬", layout="centered")
st.title("Golgi — Healthcare Evidence Search")

q = st.text_input("Query", placeholder="SGLT2 inhibitors CKD 2024 guideline")
c1, c2, c3 = st.columns([2,1,1])
with c1:
    since = st.text_input("Since (YYYY or YYYY-MM or YYYY-MM-DD)", value="")
with c2:
    k = st.number_input("Results", min_value=1, max_value=25, value=8, step=1)
with c3:
    go = st.button("Search", type="primary", use_container_width=True)

if go:
    if not q.strip():
        st.warning("Enter a query.")
    else:
        with st.spinner("Searching…"):
            rows = exa_search(q.strip(), num=int(k), since=since.strip() or None)
        if not rows:
            st.info("No results.")
        for r in rows:
            with st.container(border=True):
                st.markdown(f"**[{r['title']}]({r['url']})**")
                if r.get("published"):
                    st.caption(str(r["published"]))
                if r.get("summary"):
                    st.write(r["summary"])