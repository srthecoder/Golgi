"""Async fetchers for healthcare data sources: PubMed, NIH RePORTER, openFDA, WHO IRIS."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NIH_REPORTER_URL = "https://api.reporter.nih.gov/v2/projects/search"
FDA_DRUG_URL = "https://api.fda.gov/drug/label.json"
WHO_IRIS_URL = "https://iris.who.int/server/api/discover/search/objects"


async def fetch_pubmed(query: str, max_results: int = 20) -> list[dict[str, Any]]:
    """Fetch articles from PubMed via NCBI E-utilities (esearch + efetch).

    Returns list of dicts with keys: id, title, abstract, url, source, date.
    """
    api_key = os.getenv("PUBMED_API_KEY", "")
    email = os.getenv("PUBMED_EMAIL", "")

    params_search: dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "usehistory": "y",
    }
    if api_key:
        params_search["api_key"] = api_key
    if email:
        params_search["email"] = email

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{PUBMED_BASE}/esearch.fcgi", params=params_search)
            r.raise_for_status()
            search_data = r.json()

        result = search_data.get("esearchresult", {})
        ids = result.get("idlist", [])
        if not ids:
            return []

        params_fetch: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if api_key:
            params_fetch["api_key"] = api_key

        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{PUBMED_BASE}/efetch.fcgi", params=params_fetch)
            r.raise_for_status()
            xml_text = r.text

        return _parse_pubmed_xml(xml_text)

    except Exception as e:
        logger.warning("PubMed fetch failed: %s", e)
        return []


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse PubMed efetch XML into a list of article dicts."""
    import re

    articles = []
    article_blocks = re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml_text, re.DOTALL)

    for block in article_blocks:
        pmid_m = re.search(r"<PMID[^>]*>(\d+)</PMID>", block)
        pmid = pmid_m.group(1) if pmid_m else ""

        title_m = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", block, re.DOTALL)
        title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip() if title_m else ""

        abstract_parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", block, re.DOTALL)
        abstract = " ".join(re.sub(r"<[^>]+>", "", p).strip() for p in abstract_parts)

        year_m = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", block, re.DOTALL)
        date = year_m.group(1) if year_m else ""

        articles.append({
            "id": pmid,
            "title": title,
            "abstract": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "source": "PubMed",
            "date": date,
        })

    return articles


async def fetch_nih_reporter(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Fetch funded research projects from NIH RePORTER API v2.

    Returns list of dicts with keys: id, title, abstract, url, source, date.
    """
    payload = {
        "criteria": {"advanced_text_search": {"operator": "and", "search_field": "all", "search_text": query}},
        "limit": max_results,
        "offset": 0,
        "fields": ["ProjectTitle", "AbstractText", "ContactPiName", "FiscalYear", "CoreProjectNum"],
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(NIH_REPORTER_URL, json=payload)
            r.raise_for_status()
            data = r.json()

        results = data.get("results", [])
        docs = []
        for p in results:
            proj_num = p.get("core_project_num") or p.get("CoreProjectNum", "")
            title = p.get("project_title") or p.get("ProjectTitle", "")
            abstract = p.get("abstract_text") or p.get("AbstractText", "") or ""
            pi = p.get("contact_pi_name") or p.get("ContactPiName", "")
            year = str(p.get("fiscal_year") or p.get("FiscalYear", ""))
            url = f"https://reporter.nih.gov/project-details/{proj_num}" if proj_num else NIH_REPORTER_URL

            docs.append({
                "id": proj_num,
                "title": title,
                "abstract": f"{abstract} PI: {pi}".strip(),
                "url": url,
                "source": "NIH RePORTER",
                "date": year,
            })
        return docs

    except Exception as e:
        logger.warning("NIH RePORTER fetch failed: %s", e)
        return []


async def fetch_fda_drugs(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Fetch drug label information from openFDA drug/label endpoint.

    Returns list of dicts with keys: id, title, abstract, url, source, date.
    """
    # openFDA Lucene syntax: field:term (no phrase quotes for multi-word)
    safe_query = query.replace('"', "")
    params = {
        "search": f"indications_and_usage:{safe_query}",
        "limit": max_results,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(FDA_DRUG_URL, params=params)
            r.raise_for_status()
            data = r.json()

        results = data.get("results", [])
        docs = []
        for item in results:
            openfda = item.get("openfda", {})
            brand = openfda.get("brand_name", [""])[0]
            generic = openfda.get("generic_name", [""])[0]
            drug_name = brand or generic or "Unknown Drug"
            set_id = item.get("set_id", "")

            indications = " ".join(item.get("indications_and_usage", []))
            warnings = " ".join(item.get("warnings", []))
            abstract = f"Indications: {indications[:500]} | Warnings: {warnings[:300]}".strip("| ").strip()

            docs.append({
                "id": set_id,
                "title": drug_name,
                "abstract": abstract,
                "url": f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={set_id}",
                "source": "FDA",
                "date": "",
            })
        return docs

    except Exception as e:
        logger.warning("FDA drug fetch failed: %s", e)
        return []


async def fetch_who_iris(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Fetch publications from WHO IRIS repository via REST search API.

    Returns list of dicts with keys: id, title, abstract, url, source, date.
    """
    params = {
        "query": query,
        "size": max_results,
        "page": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=15, headers={"Accept": "application/json"}) as client:
            r = await client.get(WHO_IRIS_URL, params=params)
            r.raise_for_status()
            data = r.json()

        raw_items = (
            data.get("_embedded", {})
                .get("searchResult", {})
                .get("_embedded", {})
                .get("objects", [])
        )
        docs = []
        for item in raw_items[:max_results]:
            obj = item.get("_embedded", {}).get("indexableObject", {})
            handle = obj.get("handle", "")
            metadata = obj.get("metadata", {})

            title = (metadata.get("dc.title", [{}])[0].get("value", "")
                     or obj.get("name", "Unknown"))
            abstract = metadata.get("dc.description.abstract", [{}])[0].get("value", "")
            date = metadata.get("dc.date.issued", [{}])[0].get("value", "")

            url = f"https://iris.who.int/handle/{handle}" if handle else "https://iris.who.int"
            docs.append({
                "id": handle,
                "title": title,
                "abstract": abstract,
                "url": url,
                "source": "WHO IRIS",
                "date": date,
            })
        return docs

    except Exception as e:
        logger.warning("WHO IRIS fetch failed: %s", e)
        return []


async def fetch_all(query: str) -> list[dict[str, Any]]:
    """Run all four fetchers concurrently and return combined tagged results."""
    results = await asyncio.gather(
        fetch_pubmed(query),
        fetch_nih_reporter(query),
        fetch_fda_drugs(query),
        fetch_who_iris(query),
        return_exceptions=True,
    )

    combined: list[dict[str, Any]] = []
    source_names = ["PubMed", "NIH RePORTER", "FDA", "WHO IRIS"]
    for source_name, result in zip(source_names, results):
        if isinstance(result, Exception):
            logger.warning("%s fetch raised exception: %s", source_name, result)
        elif isinstance(result, list):
            combined.extend(result)

    return combined
