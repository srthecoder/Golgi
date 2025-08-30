#!/usr/bin/env python3
import os, re, io, json, csv, requests, tldextract, streamlit as st, altair as alt
from dateutil import parser as dtp
from exa_py import Exa
from readability import Document
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

# --- page config (only once) ---
st.set_page_config(page_title="Golgi — Healthcare Evidence Search", page_icon="assets/logo.png", layout="centered")

# --- brand CSS + landing look ---
st.markdown("""
<style>
.block-container {padding-top: 4%; text-align: center;}
h1,h2,h3 {text-align:center}
input[type="text"] {text-align:center}
.stTextInput > div > div > input {
  font-size: 1.2rem; padding: 0.8rem 1.2rem; border-radius: 28px; border: 1px solid #d0d0d0;
}
.stButton>button {
  font-size: 1.05rem; padding: 0.6rem 2.4rem; border-radius: 28px; background:#ef4444; color:#fff; border:0;
}
.card {text-align:left}
</style>
""", unsafe_allow_html=True)

# ---------------- domain config ----------------
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
    "guideline": ["practice guideline","consensus statement","recommendation"],
    "systematic": ["systematic review","meta-analysis","evidence synthesis","cochrane"],
    "rct": ["randomized controlled trial","randomised controlled trial","clinical trial","phase iii","phase ii","phase i"],
    "pregnancy": ["pregnant","gestation"],
}

# ---------------- helpers ----------------
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

def _domain(url: str) -> str:
    ext = tldextract.extract(url); return ".".join([p for p in [ext.domain, ext.suffix] if p])

def _expand_query(q: str, boosters: list[str]) -> str:
    # Append boosters + synonyms (OR-expanded)
    q2 = q
    for b in boosters:
        q2 += f" {b}"
    extra = []
    for b in boosters:
        extra += SYNONYMS.get(b, [])
    extra = sorted(set(extra))
    return f"({q2}) OR ({' OR '.join(extra)})" if extra else q2

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

# ---------------- Exa ----------------
def _exa() -> Exa:
    key = os.getenv("EXA_API_KEY") or st.secrets.get("EXA_API_KEY", "")
    if not key: st.error("Missing EXA_API_KEY"); st.stop()
    return Exa(api_key=key)

def exa_search(query: str, num: int, since: str | None, mode: str):
    exa = _exa()
    opts = dict(query=query, num_results=num, type="auto", use_autoprompt=True, text={"max_characters": 5000})
    if since: opts["start_published_date"] = since
    if mode == "Clinical": opts["include_domains"] = CLINICAL_ALLOW   # with text: only one of include/exclude
    else:                   opts["exclude_domains"] = EXCLUDE_JUNK
    res = exa.search_and_contents(**opts).results

    rows = []
    for r in res:
        url = getattr(r,"url",""); title = getattr(r,"title","") or url
        published = getattr(r,"published_date", None)
        text = getattr(r,"text", None) or _clean(_fetch(url))
        snippet = " ".join(_topk(query, text, 4)) if text else ""
        ctype = _content_type(title, url)
        score = _confidence(query, text, url, published, mode=="Clinical")
        rows.append({"title": title, "url": url, "published": published, "type": ctype, "score": score, "summary": snippet})
    return rows

def exa_overview(query: str, since: str | None, mode: str, max_cites: int) -> dict:
    exa = _exa()
    opts = dict(query=query, text=True)
    if since: opts["start_published_date"] = since
    try:
        if mode == "Clinical": opts["include_domains"] = CLINICAL_ALLOW
        else:                  opts["exclude_domains"] = EXCLUDE_JUNK
        ans = exa.answer(**opts)
    except Exception:
        ans = exa.answer(query=query, text=True)
    cites = [{"title": c.title, "url": c.url} for c in (getattr(ans,"citations",[]) or [])][:max_cites]
    return {"answer": getattr(ans,"answer","") or "", "citations": cites}

# ---------------- Landing ----------------
st.image("assets/logo.png", width=220)
st.markdown("#### Making healthcare searchable")

# Search controls (centered)
q = st.text_input(" ", placeholder="SGLT2 inhibitors CKD stage 3")
c1, c2, c3 = st.columns([1,1,1])
with c1: mode = st.radio("Mode", ["Clinical","Scholar"], horizontal=True, index=0)
with c2: since = st.text_input("Since (YYYY[-MM[-DD]])", "")
with c3: k = st.slider("Results", 1, 50, 15)

boosters = st.multiselect("Query boosters (optional)", ["guideline","systematic","rct","pregnancy"], help="Adds terms & synonyms to bias results.")
run = st.button("Search")

if run and q.strip():
    # swap to wide layout feel for results
    st.markdown("<style>.block-container{max-width: 1200px;}</style>", unsafe_allow_html=True)

    # expanded query
    q2 = _expand_query(q.strip(), boosters)

    with st.spinner("Searching…"):
        rows = exa_search(q2, num=int(k), since=since.strip() or None, mode=mode)

    # Evidence-type filter (if user clears all → treat as all)
    all_types = ["Guideline","Systematic Review","Trial/Registry","Article/Other"]
    filt = st.multiselect("Filter by evidence type", all_types, default=all_types)
    if not filt: filt = all_types
    rows = [r for r in rows if r["type"] in filt]

    # Overview (cited)
    ov = exa_overview(q2, since=since.strip() or None, mode=mode, max_cites=int(k))
    st.subheader("Overview")
    st.write(ov["answer"] or "_No summary returned._")
    if ov["citations"]:
        st.caption("Sources:")
        for c in ov["citations"]:
            st.markdown(f"- [{c['title']}]({c['url']})")

    # Stats
    st.subheader("Stats")
    total = len(rows)
    avg_score = round(sum(r["score"] for r in rows)/total, 3) if rows else 0.0
    m1, m2 = st.columns(2)
    m1.metric("Results", total)
    m2.metric("Avg confidence", avg_score)

    # Analytics tabs (pie + source counts + confidence)
    types, domains = {}, {}
    for r in rows:
        types[r["type"]] = types.get(r["type"], 0) + 1
        d = _domain(r["url"]); domains[d] = domains.get(d, 0) + 1

    tab1, tab2, tab3 = st.tabs(["Evidence types", "Source counts", "Confidence curve"])

    pie_data = [{"type": k, "count": v} for k,v in sorted(types.items(), key=lambda x:-x[1])]
    with tab1:
        if pie_data:
            pie = alt.Chart(alt.Data(values=pie_data)).mark_arc().encode(
                theta=alt.Theta("count:Q", stack=True),
                color=alt.Color("type:N", legend=alt.Legend(title="Evidence Type")),
                tooltip=["type:N","count:Q"],
            ).properties(width='container', height=300)
            st.altair_chart(pie, use_container_width=True)
        else:
            st.info("No type data")

    dom_data = [{"domain":k, "count":v} for k,v in sorted(domains.items(), key=lambda x:-x[1])[:20]]
    with tab2:
        if dom_data:
            dom_chart = alt.Chart(alt.Data(values=dom_data)).mark_bar().encode(
                x=alt.X("count:Q", title="Results"),
                y=alt.Y("domain:N", sort='-x', title="Domain"),
                tooltip=["domain:N","count:Q"],
            ).properties(width='container', height=300)
            st.altair_chart(dom_chart, use_container_width=True)
        else:
            st.info("No domain data")

    sc_data = [{"rank": i+1, "score": float(r["score"])} for i,r in enumerate(rows)]
    with tab3:
        if sc_data:
            sc_chart = alt.Chart(alt.Data(values=sc_data)).mark_line(point=True).encode(
                x=alt.X("rank:Q", title="Rank"),
                y=alt.Y("score:Q", title="Confidence"),
                tooltip=["rank:Q","score:Q"],
            ).properties(width='container', height=300)
            st.altair_chart(sc_chart, use_container_width=True)
        else:
            st.info("No score data")

    # Export
    st.subheader("Export")
    mime, blob = _download_blob(rows, "json"); st.download_button("Download JSON", data=blob, file_name="golgi_results.json", mime=mime)
    mime, blob = _download_blob(rows, "csv");  st.download_button("Download CSV",  data=blob, file_name="golgi_results.csv",  mime=mime)

    # Results (cards w/ images)
    st.subheader("Results")
    if not rows: st.info("No results.")
    for i, r in enumerate(rows, 1):
        with st.container():
            st.markdown(f"**{i}. [{r['title']}]({r['url']})** · _{r['type']}_ · **{r['score']}**", help=_domain(r["url"]))
            if r.get("published"): st.caption(str(r["published"]))
            cols = st.columns([4,1])
            with cols[0]: st.write(r["summary"] or "")
            try:
                html = _fetch(r["url"], timeout=5)
                s = BeautifulSoup(html, "html.parser")
                img = (s.find("meta", property="og:image") or s.find("meta", attrs={"name":"twitter:image"}))
                if img and img.get("content"): cols[1].image(img["content"], use_column_width=True)
            except Exception:
                pass