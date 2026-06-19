from __future__ import annotations
import os, re, requests
from typing import List, Dict, Any
from exa_py import Exa
from bs4 import BeautifulSoup
from readability import Document
from rank_bm25 import BM25Okapi

HEALTH_ALLOW = [
    "nih.gov","ncbi.nlm.nih.gov","pubmed.ncbi.nlm.nih.gov","cdc.gov","who.int",
    "nejm.org","jamanetwork.com","thelancet.com","bmj.com","nature.com",
    "fda.gov","ema.europa.eu","clinicaltrials.gov","cochranelibrary.com",
]
HEALTH_EXCLUDE = ["wikipedia.org","twitter.com","x.com","facebook.com","reddit.com",
                  "medium.com","quora.com","linkedin.com","youtube.com"]

def _fetch(url: str, timeout=12) -> str:
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

def _topk(query: str, text: str, k: int = 5) -> List[str]:
    if not text: return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    toks = [re.findall(r"\w+", s.lower()) for s in sents]
    if not toks: return []
    bm25 = BM25Okapi(toks)
    qtok = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(qtok)
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [sents[i] for i in idx]

class GolgiEngine:
    def __init__(self, exa_key: str | None = None):
        key = exa_key or os.getenv("EXA_API_KEY")
        if not key: raise RuntimeError("Missing EXA_API_KEY")
        self.exa = Exa(api_key=key)

    def search(self, query: str, num: int = 8, since: str | None = None) -> List[Dict[str, Any]]:
        resp = self.exa.search_and_contents(
            query=query, num_results=num,
            include_domains=HEALTH_ALLOW, exclude_domains=HEALTH_EXCLUDE,
            start_published_date=since, type="neural",
            text={"max_characters": 5000}, use_autoprompt=True,
        )
        out: List[Dict[str, Any]] = []
        for r in resp.results:
            url = getattr(r, "url", "")
            title = getattr(r, "title", "") or url
            published = getattr(r, "published_date", None)
            txt = getattr(r, "text", None) or _clean(_fetch(url))
            snippet = " ".join(_topk(query, txt, 5)) if txt else ""
            out.append({"title": title, "url": url, "published": published, "summary": snippet})
        return out