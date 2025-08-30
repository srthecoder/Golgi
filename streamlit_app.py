# streamlit_app.py — Golgi (specialized UI: overview, scores, charts, images)
from __future__ import annotations
import os, re, json, math, time, requests, tldextract
import streamlit as st
import altair as alt
from exa_py import Exa
from readability import Document
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

ALLOW = [
    "nih.gov","ncbi.nlm.nih.gov","pubmed.ncbi.nlm.nih.gov","cdc.gov","who.int",
    "nejm.org","jamanetwork.com","thelancet.com","bmj.com","nature.com",
    "fda.gov","ema.europa.eu","clinicaltrials.gov","cochranelibrary.com",
]

SYNONYMS = {
    "mi": ["myocardial infarction","heart attack","stemi","nstemi"],
    "htn": ["hypertension","high blood pressure"],
    "ckd": ["chronic kidney disease","renal insufficiency"],
    "dm": ["diabetes mellitus","t2d","t1d","type 2 diabetes","type 1 diabetes"],
    "anticoag": ["anticoagulation","doac","apixaban","rivaroxaban","warfarin"],
}

# ---------------- core helpers ----------------
def _fetch(url: str, timeout=8) -> str:
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
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [sents[i] for i in order]

def _og_image(url: str) -> str|None:
    try:
        html = _fetch(url, timeout=5)
        if not html: return None
        s = BeautifulSoup(html, "html.parser")
        og = s.find("meta", property="og:image") or s.find("meta", attrs={"name":"twitter:image"})
        if og and og.get("content"): return og["content"]
        # favicon fallback
        base = re.match(r"(https?://[^/]+)", url)
        fav = s.find("link", rel=lambda x: x and "icon" in x.lower()) or {}
        if fav and fav.get("href"): 
            href = fav["href"]
            return href if href.startswith("http") else (base.group(1)+href if base else None)
        return None
    except Exception:
        return None

def _domain(url: str) -> str:
    ext = tldextract.extract(url)
    return ".".join([p for p in [ext.domain, ext.suffix] if p])

def _expand_query(q: str) -> str:
    ql = q.lower()
    extra = sorted({s for k,vs in SYNONYMS.items() if k in ql for s in vs})
    return f"({q}) OR ({' OR '.join(extra)})" if extra else q

def _score(q: str, text: str, url: str, published: str|None) -> float:
    # heuristic: lexical overlap + recency boost + domain prior
    tokens_q = set(re.findall(r"\w+", q.lower()))
    tokens_t = set(re.findall(r"\w+", text.lower())) if text else set()
    overlap = len(tokens_q & tokens_t) / (len(tokens_q) + 1e-6)
    # recency: year in date string
    rec = 0.0
    if published:
        m = re.search(r"(\d{4})", str(published))
        if m:
            yr = int(m.group(1))
            rec = max(0.0, min(1.0, (yr - 2015) / 10.0))  # 2015..2025 -> 0..1
    dom = _domain(url)
    prior = 1.0 if any(dom.endswith(d) for d in ALLOW) else 0.6
    return round(0.6*overlap + 0.3*rec + 0.1*prior, 3)

# ---------------- Exa client ----------------
def _get_exa_key() -> str:
    key = os.getenv("EXA_API_KEY") or st.secrets.get("EXA_API_KEY", "")
    if not key:
        st.error("Missing EXA_API_KEY (env var or .streamlit/secrets.toml).")
        st.stop()
    return key

@st.cache_resource
def _exa() -> Exa:
    return Exa(api_key=_get_exa_key())

@st.cache_data(show_spinner=False)
def exa_search(query: str, num: int, since: str|None, use_filters: bool):
    exa = _exa()
    opts = dict(query=_expand_query(query), num_results=num, type="auto", use_autoprompt=True, text={"max_characters": 5000})
    if since: opts["start_published_date"] = since
    if use_filters: opts["include_domains"] = ALLOW  # only includeDomains when requesting text
    res = exa.search_and_contents(**opts)
    out = []
    for r in res.results:
        url = getattr(r,"url",""); title = getattr(r,"title","") or url
        published = getattr(r,"published_date", None)
        text = getattr(r,"text", None) or _clean(_fetch(url))
        snippet = " ".join(_topk(query, text, 5)) if text else ""
        score = _score(query, text or "", url, published)
        out.append({"title": title, "url": url, "published": published, "summary": snippet, "score": score})
    return out

@st.cache_data(show_spinner=False)
def exa_overview(query: str, since: str|None, use_filters: bool) -> dict:
    exa = _exa()
    opts = dict(query=query, text=True)
    if since: opts["start_published_date"] = since
    if use_filters: opts["include_domains"] = ALLOW
    ans = exa.answer(**opts)
    cites = [{"title": c.title, "url": c.url} for c in (ans.citations or [])]
    return {"answer": ans.answer or "", "citations": cites}

# ---------------- UI ----------------
st.set_page_config(page_title="Golgi — Healthcare Evidence Search", page_icon="🧬", layout="wide")
st.title("Golgi — Healthcare Evidence Search")

with st.sidebar:
    st.subheader("Query helpers")
    chips = st.multiselect("Quick terms", ["guideline","systematic review","randomized controlled trial","contraindications","pregnancy","perioperative","dose"], default=["guideline"])
    since = st.text_input("Since (YYYY[-MM[-DD]])", "")
    k = st.slider("Results", 1, 25, 8)
    use_filters = st.checkbox("Healthcare allowlist", True)
    show_images = st.checkbox("Show images", True)
    show_charts = st.checkbox("Show charts", True)
    show_scores = st.checkbox("Show scores", True)
    want_overview = st.checkbox("High-level overview", True)

base_q = st.text_input("Query", placeholder="SGLT2 inhibitors CKD stage 3", value="")
q = f"{base_q} {' '.join(chips)}".strip()

c1, c2 = st.columns([1,1])
run = c1.button("Search", type="primary", use_container_width=True)
clear = c2.button("Clear", use_container_width=True)
if clear:
    st.experimental_rerun()

if run:
    with st.spinner("Searching…"):
        rows = exa_search(q, num=int(k), since=since.strip() or None, use_filters=use_filters)

    if want_overview:
        with st.status("Generating overview…", expanded=False) as s:
            ov = exa_overview(q, since=since.strip() or None, use_filters=use_filters)
            s.update(label="Overview ready", state="complete")
        st.subheader("Overview")
        st.write(ov["answer"] or "_No summary returned._")
        if ov["citations"]:
            st.caption("Sources:")
            for c in ov["citations"]:
                st.markdown(f"- [{c['title']}]({c['url']})")

    # charts
    if show_charts and rows:
        st.subheader("Analytics")
        # Domain distribution
        domain_counts = {}
        for r in rows:
            d = _domain(r["url"])
            domain_counts[d] = domain_counts.get(d, 0) + 1
        dom_data = [{"domain": k, "count": v} for k,v in sorted(domain_counts.items(), key=lambda x: -x[1])]
        chart = alt.Chart(dom_data).mark_bar().encode(x=alt.X("count:Q", title="Results"), y=alt.Y("domain:N", sort='-x', title="Domain"))
        st.altair_chart(chart, use_container_width=True)

        # Score distribution
        if show_scores:
            score_data = [{"rank": i+1, "score": r["score"]} for i, r in enumerate(rows)]
            score_chart = alt.Chart(score_data).mark_line(point=True).encode(x="rank:Q", y="score:Q")
            st.altair_chart(score_chart, use_container_width=True)

    # results
    st.subheader("Results")
    if not rows:
        st.info("No results.")
    for r in rows:
        with st.container(border=True):
            header = f"**[{r['title']}]({r['url']})**"
            if show_scores: header += f" · _score {r['score']}_"
            st.markdown(header)
            if r.get("published"): st.caption(str(r["published"]))
            cols = st.columns([4,1]) if show_images else [st]
            with (cols[0] if show_images else cols[0]):
                st.write(r["summary"] or "")
            if show_images:
                img = _og_image(r["url"])
                if img: cols[1].image(img, use_column_width=True)