"""Intent detection and source routing for the Golgi pipeline."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from google import genai

logger = logging.getLogger(__name__)

INTENT_LABELS = ("drug_info", "clinical_evidence", "guidelines", "research_funding", "general")

_CLASSIFY_PROMPT = (
    "Classify the following healthcare search query into exactly one of these intents: "
    "drug_info, clinical_evidence, guidelines, research_funding, general.\n"
    "Reply with the intent label only — no explanation, no punctuation.\n\n"
    "Query: {query}"
)


class IntentRouter:
    """Detects query intent with Gemini and fetches from the appropriate sources."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def detect_intent(self, query: str) -> str:
        """Classify query into one of the five intent labels using Gemini.

        Args:
            query: Free-text search query.

        Returns:
            One of: drug_info, clinical_evidence, guidelines, research_funding, general.
        """
        try:
            response = self._client.models.generate_content(
                model="gemini-2.0-flash",
                contents=_CLASSIFY_PROMPT.format(query=query),
            )
            label = response.text.strip().lower()
            if label not in INTENT_LABELS:
                logger.warning("Unexpected intent label %r — defaulting to general", label)
                return "general"
            return label
        except Exception as exc:
            logger.warning("Intent detection failed: %s — defaulting to general", exc)
            return "general"

    def route_and_fetch(self, query: str) -> dict[str, Any]:
        """Detect intent and fetch documents from prioritised sources.

        Args:
            query: Free-text search query.

        Returns:
            Dict with keys docs (list of doc dicts) and intent (str).
        """
        from ingestion.fetchers import (
            fetch_fda_drugs,
            fetch_nih_reporter,
            fetch_pubmed,
            fetch_who_iris,
        )

        intent = self.detect_intent(query)
        logger.info("Detected intent: %s", intent)

        async def _gather(*coros):
            return await asyncio.gather(*coros, return_exceptions=True)

        def run(coro):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(asyncio.run, coro)
                        return future.result()
                return loop.run_until_complete(coro)
            except RuntimeError:
                return asyncio.run(coro)

        if intent == "drug_info":
            results = run(_gather(fetch_fda_drugs(query), fetch_pubmed(query)))
        elif intent == "clinical_evidence":
            results = run(_gather(fetch_pubmed(query), fetch_nih_reporter(query)))
        elif intent == "guidelines":
            results = run(_gather(fetch_who_iris(query), fetch_nih_reporter(query)))
        elif intent == "research_funding":
            results = run(_gather(fetch_nih_reporter(query)))
        else:
            from ingestion.fetchers import fetch_all
            results = [run(fetch_all(query))]

        docs: list[dict[str, Any]] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Fetcher error: %s", r)
            elif isinstance(r, list):
                docs.extend(r)

        logger.info("route_and_fetch: %d docs for intent=%s", len(docs), intent)
        return {"docs": docs, "intent": intent}
