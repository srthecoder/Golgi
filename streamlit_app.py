#!/usr/bin/env python3
# streamlit_app.py — Golgi (researcher view: images kept, no guidelines graph; source counts, confidence, pie)
from __future__ import annotations
import os, re, io, json, csv, requests, tldextract, streamlit as st, altair as alt
from dateutil import parser as dtp
from exa_py import Exa
from readability import Document
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

CLINICAL_ALLOW = [
    "nih.gov","ncbi.nlm.nih.gov","pubmed.ncbi.nlm.nih.gov","cdc.gov","who.int",
    "nejm.org","jamanetwork.com","thelancet.com","bmj.com","nature.com",
    "sciencedirect.com","springer.com","wiley.com","oxfordacademic.com","tandfonline.com",
    "medrxiv.org","biorxiv.org","arxiv.org",
    "fda.gov","ema.europa.eu","hhs.gov","ahrq.gov","health.gov","europa.eu",
    "clinicaltrials.gov","isrctn.com","anzctr.org.au","cochranelibrary.com"
]
EXCLUDE_JUNK = ["twitter.com","x.com","facebook.com","reddit.com","quora.com","linkedin.com","youtube.com","pinterest.com"]

SYNONYMS = {
    "mi": ["myocardial infarction","heart attack","stemi","nstemi"],
    "htn": ["hypertension","high blood pressure"],
    "ckd": ["chronic kidney disease","renal insufficiency"],
    "dm": ["diabetes mellitus","type 2 diabetes","t2d","type 1 diabetes","t1d"],
    "rct": ["randomized controlled trial","randomised controlled trial","clinical trial"],
    "guideline": ["practice guideline","consensus statement","recommendation"],
    "systematic": ["systematic review","meta-analysis","evidence synthesis"],
}

# ---------- utils ----------
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

def _topk(query: str, text: str, k: int = 4) -> list[str]:
    if not text: return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    toks = [re.findall(r"\w+", s.lower()) for s in sents]
    if not toks: return []
    bm25 = BM25Okapi(toks)
    qtok = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(qtok)
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [sents[i] for i in idx]

def _og_image(url: str) -> str | None:
    try:
        html = _fetch(url, timeout=5)
        if not html: return None
        s = BeautifulSoup(html, "html.parser")
        og = s.find("meta", property="og:image") or s.find("meta", attrs={"name":"twitter:image"})
        if og and og.get("content"): return og["content"]
        fav = s.find("link", rel=lambda x: x and "icon" in x.lower())
        if fav and fav.get("href"):
            href = fav["href"]; base = re.match(r"(https?://[^/]+)", url)
            return href if href.startswith("http") else (base.group(1)+href if base else None)
        return None
    except Exception:
        return None

def _domain(url: str) -> str:
    ext = tldextract.extract(url); return ".".join([p for p in [ext.domain, ext.suffix] if p])

def _expand_query(q: str) -> str:
    ql = q.lower()
    extra = sorted({t for k,vs in SYNONYMS.items() if k in ql for t in vs})
    return f"({q}) OR ({' OR '.join(extra)})" if extra else q

def _year_from(val) -> int | None:
    if not val: return None
    try:
        return dtp.parse(str(val), default=dtp.parse("2000-01-01")).year
    except Exception:
        m = re.search(r"(\d{4})", str(val)); return int(m.group(1)) if m else None

def _content_type(title: str, url: str) -> str:
    s = f"{title} {url}".lower()
    if any(k in s for k in ["guideline","consensus","recommendation","practice guideline"]): return "Guideline"
    if any(k in s for k in ["systematic review","meta-analysis","cochrane"]): return "Systematic Review"
    if "clinicaltrials.gov" in s or any(k in s for k in ["randomized","randomised","rct","phase i","phase ii","phase iii"]): return "Trial/Registry"
    return "Article/Other"

def _confidence(q: str, text: str, url: str, published, clinical_mode: bool) -> float:
    tq = set(re.findall(r"\w+", q.lower()))
    tt = set(re.findall(r"\w+", (text or "").lower()))
    overlap = len(tq & tt) / (len(tq) + 1e-6)
    yr = _year_from(published) or 2010
    rec = max(0.0, min(1.0, (yr - 2015) / 10.0))
    dom_prior = 1.0 if (clinical_mode and any(_domain(url).endswith(d) for d in CLINICAL_ALLOW)) else 0.7
    return round(0.6*overlap + 0.3*rec + 0.1*dom_prior, 3)

def _download_blob(rows: list[dict], kind: str):
    if kind == "json":
        return "application/json", json.dumps(rows, indent=2).encode()
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["title","url","published","type","score","summary"])
    w.writeheader()
    for r in rows: w.writerow({k:r.get(k,"") for k in w.fieldnames})
    return "text/csv", buf.getvalue().encode()

# ---------- Exa ----------
def _get_exa_key() -> str:
    key = os.getenv("EXA_API_KEY") or st.secrets.get("EXA_API_KEY", "")
    if not key:
        st.error("Missing EXA_API_KEY (env or .streamlit/secrets.toml)"); st.stop()
    return key

@st.cache_resource
def _exa() -> Exa: return Exa(api_key=_get_exa_key())

@st.cache_data(show_spinner=False)
def exa_search(query: str, num: int, since: str | None, mode: str):
    exa = _exa()
    opts = dict(query=_expand_query(query), num_results=num, type="auto", use_autoprompt=True, text={"max_characters": 5000})
    if since: opts["start_published_date"] = since
    clinical = mode.startswith("Clinical")
    if clinical: opts["include_domains"] = CLINICAL_ALLOW
    else:        opts["exclude_domains"] = EXCLUDE_JUNK
    res = exa.search_and_contents(**opts).results

    rows = []
    for r in res:
        url = getattr(r,"url",""); title = getattr(r,"title","") or url
        published = getattr(r,"published_date", None)
        text = getattr(r,"text", None) or _clean(_fetch(url))
        snippet = " ".join(_topk(query, text, 4)) if text else ""
        ctype = _content_type(title, url)
        score = _confidence(query, text, url, published, clinical)
        rows.append({"title": title, "url": url, "published": published, "type": ctype, "score": score, "summary": snippet})
    return rows

@st.cache_data(show_spinner=False)
def exa_overview(query: str, since: str|None, mode: str, max_citations: int) -> dict:
    exa = _exa()
    opts = dict(query=query, text=True)
    if since: opts["start_published_date"] = since
    try:
        if mode.startswith("Clinical"): opts["include_domains"] = CLINICAL_ALLOW
        else:                            opts["exclude_domains"] = EXCLUDE_JUNK
        ans = exa.answer(**opts)
    except Exception as e:
        st.warning(f"Overview failed with filters: {e}")
        ans = exa.answer(query=query, text=True)
    cites_all = [{"title": c.title, "url": c.url} for c in (getattr(ans, "citations", []) or [])]
    cites = cites_all[:max(0, int(max_citations))]
    return {"answer": getattr(ans, "answer", "") or "", "citations": cites}

# ---------- UI ----------
st.set_page_config(page_title="Golgi — Research Mode", page_icon="🧬", layout="wide")
st.title("Golgi — Healthcare Evidence Search")

with st.sidebar:
    mode = st.radio("Mode", ["Clinical (strict)","Scholar (broad)"], index=0)
    chips = st.multiselect("Quick terms", ["guideline","systematic review","randomized controlled trial","contraindications","pregnancy","perioperative","dose"], default=["guideline"])
    since = st.text_input("Since (YYYY[-MM[-DD]])", "")
    k = st.slider("Results", 1, 50, 15)
    want_overview = st.checkbox("High-level overview", True)
    # evidence-type filter
    all_types = ["Guideline","Systematic Review","Trial/Registry","Article/Other"]
    type_filter = st.multiselect("Filter by evidence type", options=all_types, default=all_types)

base_q = st.text_input("Query", placeholder="SGLT2 inhibitors CKD stage 3")
q = f"{base_q} {' '.join(chips)}".strip()

run = st.button("Search", type="primary", use_container_width=True)
if run:
    with st.spinner("Searching…"):
        rows = exa_search(q, num=int(k), since=since.strip() or None, mode=mode)

    # filter by evidence type
    rows = [r for r in rows if r["type"] in type_filter]

    # overview
    if want_overview:
        with st.status("Generating overview…", expanded=False) as s:
            ov = exa_overview(q, since=since.strip() or None, mode=mode, max_citations=int(k))
            s.update(label="Overview ready", state="complete")
        st.subheader("Overview")
        st.write(ov["answer"] or "_No summary returned._")
        if ov["citations"]:
            st.caption("Sources:")
            for c in ov["citations"]:
                st.markdown(f"- [{c['title']}]({c['url']})")

    # metrics (no guidelines metric)
    st.subheader("Stats")
    total = len(rows)
    avg_score = round(sum(r["score"] for r in rows)/total, 3) if rows else 0.0
    m1, m2 = st.columns(2)
    m1.metric("Results", total)
    m2.metric("Avg confidence", avg_score)

    # analytics: pie (type), bar (domains), line (confidence)
    st.subheader("Analytics")
    # aggregate
    types, domains = {}, {}
    for r in rows:
        types[r["type"]] = types.get(r["type"], 0) + 1
        d = _domain(r["url"]); domains[d] = domains.get(d, 0) + 1

    cols = st.columns(3)
    pie_data = [{"type": k, "count": v} for k, v in sorted(types.items(), key=lambda x: -x[1])]
    if pie_data:
        pie = alt.Chart(alt.Data(values=pie_data)).mark_arc(outerRadius=110).encode(
            theta=alt.Theta("count:Q", stack=True),
            color=alt.Color("type:N", legend=alt.Legend(title="Evidence Type")),
            tooltip=["type:N","count:Q"],
        )
        cols[0].altair_chart(pie, use_container_width=True)
    else:
        cols[0].info("No type data")

    dom_data = [{"domain":k, "count":v} for k,v in sorted(domains.items(), key=lambda x:-x[1])[:20]]
    if dom_data:
        dom_chart = alt.Chart(alt.Data(values=dom_data)).mark_bar().encode(
            x=alt.X("count:Q", title="Results"),
            y=alt.Y("domain:N", sort='-x', title="Domain"),
            tooltip=["domain:N","count:Q"],
        )
        cols[1].altair_chart(dom_chart, use_container_width=True)
    else:
        cols[1].info("No domain data")

    sc_data = [{"rank": i+1, "score": float(r["score"])} for i,r in enumerate(rows)]
    if sc_data:
        sc_chart = alt.Chart(alt.Data(values=sc_data)).mark_line(point=True).encode(
            x=alt.X("rank:Q", title="Rank"),
            y=alt.Y("score:Q", title="Confidence"),
            tooltip=["rank:Q","score:Q"],
        )
        cols[2].altair_chart(sc_chart, use_container_width=True)
    else:
        cols[2].info("No score data")

    # export
    st.subheader("Export")
    mime, blob = _download_blob(rows, "json"); st.download_button("Download JSON", data=blob, file_name="golgi_results.json", mime=mime)
    mime, blob = _download_blob(rows, "csv");  st.download_button("Download CSV",  data=blob, file_name="golgi_results.csv",  mime=mime)

    # results (images kept)
    st.subheader("Results")
    if not rows: st.info("No results.")
    for i, r in enumerate(rows, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. [{r['title']}]({r['url']})** · _{r['type']}_ · **{r['score']}**")
            if r.get("published"): st.caption(str(r["published"]))
            cols2 = st.columns([4,1])
            with cols2[0]: st.write(r["summary"] or "")
            img = _og_image(r["url"])
            if img: cols2[1].image(img, use_column_width=True)