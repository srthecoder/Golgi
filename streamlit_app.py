#!/usr/bin/env python3
import os, re, io, json, csv, requests, tldextract, streamlit as st, altair as alt
from dateutil import parser as dtp
from exa_py import Exa
from readability import Document
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

# -------- page config --------
st.set_page_config(page_title="Golgi — Healthcare Evidence Search", page_icon="assets/logo.jpg", layout="wide")
st.markdown("""
<style>
/* main layout: centered content area; keep sidebar */
.block-container{max-width:1100px;margin:0 auto;padding-top:3%;}
/* clean logo (no rounding/shadow) */
.stImage img{border-radius:0!important; box-shadow:none!important;}
/* googley input/button */
.stTextInput > div > div > input{
  font-size:1.15rem; padding:.8rem 1.2rem; border-radius:28px; text-align:center;
}
.stButton>button{
  font-size:1.05rem; padding:.6rem 2.2rem; border-radius:28px; background:#ef4444; color:#fff; border:0;
}
h1,h2,h3{text-align:center}
.card{text-align:left}
</style>
""", unsafe_allow_html=True)

# -------- domains / synonyms --------
CLINICAL_ALLOW = [
  "nih.gov","ncbi.nlm.nih.gov","pubmed.ncbi.nlm.nih.gov","cdc.gov","who.int",
  "nejm.org","jamanetwork.com","thelancet.com","bmj.com","nature.com",
  "sciencedirect.com","springer.com","wiley.com","oxfordacademic.com","tandfonline.com",
  "medrxiv.org","biorxiv.org","arxiv.org","fda.gov","ema.europa.eu","hhs.gov",
  "ahrq.gov","health.gov","europa.eu","clinicaltrials.gov","isrctn.com",
  "anzctr.org.au","cochranelibrary.com"
]
EXCLUDE_JUNK = ["twitter.com","x.com","facebook.com","reddit.com","quora.com","linkedin.com","youtube.com","pinterest.com"]
SYNONYMS = {
  "guideline": ["practice guideline","consensus statement","recommendation"],
  "systematic": ["systematic review","meta-analysis","evidence synthesis","cochrane"],
  "rct": ["randomized controlled trial","randomised controlled trial","clinical trial","phase iii","phase ii","phase i"],
}

# -------- helpers --------
def _fetch(url, timeout=8):
    try:
        r=requests.get(url,timeout=timeout,headers={"User-Agent":"Golgi/1.0"}); r.raise_for_status(); return r.text
    except Exception: return ""
def _clean(html):
    if not html: return ""
    try: soup=BeautifulSoup(Document(html).summary(html_partial=True),"html.parser")
    except Exception: soup=BeautifulSoup(html,"html.parser")
    for t in soup(["script","style","noscript"]): t.extract()
    return re.sub(r"\s+"," ", soup.get_text(" ").strip())
def _topk(q, text, k=4):
    if not text: return []
    sents=re.split(r"(?<=[.!?])\s+", text)
    toks=[re.findall(r"\w+", s.lower()) for s in sents]
    if not toks: return []
    bm=BM25Okapi(toks); qtok=re.findall(r"\w+", q.lower())
    scores=bm.get_scores(qtok)
    idx=sorted(range(len(scores)), key=lambda i:scores[i], reverse=True)[:k]
    return [sents[i] for i in idx]
def _domain(url):
    ext=tldextract.extract(url); return ".".join([p for p in [ext.domain, ext.suffix] if p])
def _year(val):
    if not val: return None
    try: return dtp.parse(str(val), default=dtp.parse("2000-01-01")).year
    except Exception:
        m=re.search(r"(\d{4})", str(val)); return int(m.group(1)) if m else None
def _ctype(title, url):
    s=f"{title} {url}".lower()
    if any(k in s for k in ["guideline","consensus","recommendation","practice guideline"]): return "Guideline"
    if any(k in s for k in ["systematic review","meta-analysis","cochrane"]): return "Systematic Review"
    if "clinicaltrials.gov" in s or any(k in s for k in ["randomized","randomised","rct","phase i","phase ii","phase iii"]): return "Trial/Registry"
    return "Article/Other"
def _conf(q, text, url, published, clinical):
    tq=set(re.findall(r"\w+", q.lower())); tt=set(re.findall(r"\w+", (text or "").lower()))
    overlap=len(tq & tt)/(len(tq)+1e-6)
    y=_year(published) or 2010; rec=max(0.0,min(1.0,(y-2015)/10.0))
    prior=1.0 if (clinical and any(_domain(url).endswith(d) for d in CLINICAL_ALLOW)) else 0.7
    return round(0.6*overlap + 0.3*rec + 0.1*prior, 3)
def _dl(rows, kind):
    if kind=="json": return "application/json", json.dumps(rows,indent=2).encode()
    buf=io.StringIO(); w=csv.DictWriter(buf, fieldnames=["title","url","published","type","score","summary"]); w.writeheader()
    for r in rows: w.writerow({k:r.get(k,"") for k in w.fieldnames})
    return "text/csv", buf.getvalue().encode()

# -------- Exa --------
def _exa():
    key=os.getenv("EXA_API_KEY") or st.secrets.get("EXA_API_KEY","")
    if not key: st.error("Missing EXA_API_KEY"); st.stop()
    return Exa(api_key=key)
def _expand(q, boosters):
    q2=q + (" " + " ".join(boosters) if boosters else "")
    extra=[]
    for b in boosters: extra += SYNONYMS.get(b, [])
    extra=sorted(set(extra))
    return f"({q2}) OR ({' OR '.join(extra)})" if extra else q2
def exa_search(query, num, since, mode):
    exa=_exa()
    opts=dict(query=query, num_results=num, type="auto", use_autoprompt=True, text={"max_characters":5000})
    if since: opts["start_published_date"]=since
    if mode=="Clinical": opts["include_domains"]=CLINICAL_ALLOW
    else: opts["exclude_domains"]=EXCLUDE_JUNK
    res=exa.search_and_contents(**opts).results
    rows=[]
    for r in res:
        url=getattr(r,"url",""); title=getattr(r,"title","") or url; pub=getattr(r,"published_date",None)
        text=getattr(r,"text",None) or _clean(_fetch(url))
        snippet=" ".join(_topk(query,text,4)) if text else ""
        ctype=_ctype(title,url); score=_conf(query,text,url,pub, mode=="Clinical")
        rows.append({"title":title,"url":url,"published":pub,"type":ctype,"score":score,"summary":snippet})
    return rows
def exa_overview(query, since, mode, max_cites):
    exa=_exa(); opts=dict(query=query, text=True)
    if since: opts["start_published_date"]=since
    try:
        if mode=="Clinical": opts["include_domains"]=CLINICAL_ALLOW
        else: opts["exclude_domains"]=EXCLUDE_JUNK
        ans=exa.answer(**opts)
    except Exception: ans=exa.answer(query=query, text=True)
    cites=[{"title":c.title,"url":c.url} for c in (getattr(ans,"citations",[]) or [])][:max_cites]
    return {"answer":getattr(ans,"answer","") or "", "citations":cites}

# -------- SIDEBAR (back!) --------
with st.sidebar:
    st.image("assets/logo.png", width=140)
    st.markdown("### Golgi")
    mode = st.radio("Mode", ["Clinical","Scholar"], index=0)
    since = st.text_input("Since (YYYY[-MM[-DD]])", "")
    k = st.slider("Results", 1, 50, 15)
    boosters = st.multiselect("Query boosters", ["guideline","systematic","rct"], help="Bias toward evidence types.")
    all_types=["Guideline","Systematic Review","Trial/Registry","Article/Other"]
    type_filter = st.multiselect("Filter by evidence type", all_types, default=all_types)
    if not type_filter: type_filter = all_types
    show_overview = st.checkbox("High-level overview", True)

# -------- MAIN: clean landing --------
st.image("assets/logo.png", width=220)
st.markdown("#### Making healthcare searchable")
query = st.text_input(" ", placeholder="SGLT2 inhibitors CKD stage 3")
run = st.button("Search")

if run and query.strip():
    q2=_expand(query.strip(), boosters)
    with st.spinner("Searching…"):
        rows=exa_search(q2, num=int(k), since=since.strip() or None, mode=mode)

    rows=[r for r in rows if r["type"] in type_filter]

    if show_overview:
        ov=exa_overview(q2, since=since.strip() or None, mode=mode, max_cites=int(k))
        st.subheader("Overview")
        st.write(ov["answer"] or "_No summary returned._")
        if ov["citations"]:
            st.caption("Sources:")
            for c in ov["citations"]: st.markdown(f"- [{c['title']}]({c['url']})")

    # Stats
    st.subheader("Stats")
    total=len(rows); avg=round(sum(r["score"] for r in rows)/total,3) if rows else 0.0
    cA,cB=st.columns(2); cA.metric("Results", total); cB.metric("Avg confidence", avg)

    # Analytics
    types, domains = {}, {}
    for r in rows:
        types[r["type"]] = types.get(r["type"],0)+1
        d=_domain(r["url"]); domains[d]=domains.get(d,0)+1
    tab1,tab2,tab3=st.tabs(["Evidence types","Source counts","Confidence curve"])

    with tab1:
        data=[{"type":k,"count":v} for k,v in sorted(types.items(), key=lambda x:-x[1])]
        if data:
            chart=alt.Chart(alt.Data(values=data)).mark_arc().encode(
                theta=alt.Theta("count:Q"), color=alt.Color("type:N", legend=alt.Legend(title="Evidence Type")),
                tooltip=["type:N","count:Q"]).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else: st.info("No type data")

    with tab2:
        dom=[{"domain":k,"count":v} for k,v in sorted(domains.items(), key=lambda x:-x[1])[:20]]
        if dom:
            chart=alt.Chart(alt.Data(values=dom)).mark_bar().encode(
                x=alt.X("count:Q", title="Results"), y=alt.Y("domain:N", sort='-x', title="Domain"),
                tooltip=["domain:N","count:Q"]).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else: st.info("No domain data")

    with tab3:
        sc=[{"rank":i+1,"score":float(r["score"])} for i,r in enumerate(rows)]
        if sc:
            chart=alt.Chart(alt.Data(values=sc)).mark_line(point=True).encode(
                x=alt.X("rank:Q", title="Rank"), y=alt.Y("score:Q", title="Confidence"),
                tooltip=["rank:Q","score:Q"]).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else: st.info("No score data")

    # Export
    st.subheader("Export")
    mime,blob=_dl(rows,"json"); st.download_button("Download JSON", data=blob, file_name="golgi_results.json", mime=mime)
    mime,blob=_dl(rows,"csv");  st.download_button("Download CSV",  data=blob, file_name="golgi_results.csv",  mime=mime)

    # Results
    st.subheader("Results")
    if not rows: st.info("No results.")
    for i,r in enumerate(rows,1):
        with st.container():
            st.markdown(f"**{i}. [{r['title']}]({r['url']})** · _{r['type']}_ · **{r['score']}**", help=_domain(r["url"]))
            if r.get("published"): st.caption(str(r["published"]))
            cols=st.columns([4,1])
            with cols[0]: st.write(r["summary"] or "")
            try:
                html=_fetch(r["url"], timeout=5)
                s=BeautifulSoup(html,"html.parser")
                img=(s.find("meta",property="og:image") or s.find("meta",attrs={"name":"twitter:image"}))
                if img and img.get("content"): cols[1].image(img["content"], use_column_width=True)
            except Exception: pass