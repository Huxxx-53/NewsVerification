"""Source retrieval.

`SearchProvider` is the interface the rest of the app depends on. To switch to a
different search engine, write a new subclass that returns a list of `Source`
objects and register it in `PROVIDERS` (see README, "Swapping the search provider").
"""
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import requests

from models.verification_result import Source
from utils.helpers import classify_source, get_domain, is_safe_url, truncate


class SearchError(Exception):
    """Raised with a user-friendly message when a search cannot be completed."""


class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[Source]:
        """Return real search results as Source objects, or raise SearchError."""


class TavilySearchProvider(SearchProvider):
    """Tavily Search API (free tier, no credit card): https://tavily.com"""

    API_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 15, topic: str = "general"):
        self.api_key = api_key if api_key is not None else os.getenv("SEARCH_API_KEY", "")
        self.timeout = timeout
        self.topic = topic  # "general" or "news" ("news" results usually include dates)

    def search(self, query: str, max_results: int = 5) -> List[Source]:
        if not self.api_key:
            raise SearchError(
                "The search service is not configured. Please set SEARCH_API_KEY in your .env file."
            )
        payload = {
            "query": query,
            "topic": self.topic,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            response = requests.post(self.API_URL, json=payload, headers=headers, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise SearchError("The search service took too long to respond. Please try again later.")
        except requests.exceptions.RequestException:
            raise SearchError("Unable to retrieve sources at the moment. Please try again later.")

        if response.status_code in (401, 403):
            raise SearchError("The search service rejected the API key. Please check SEARCH_API_KEY.")
        if response.status_code in (429, 432, 433):
            raise SearchError("The search service rate limit or usage quota was reached. Please try again later.")
        if response.status_code >= 400:
            raise SearchError("Unable to retrieve sources at the moment. Please try again later.")

        try:
            data = response.json()
        except ValueError:
            raise SearchError("The search service returned an unreadable response. Please try again later.")

        return self._parse_results(data)

    @staticmethod
    def _parse_results(data: dict) -> List[Source]:
        sources: List[Source] = []
        for item in (data or {}).get("results", []) or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not is_safe_url(url):
                continue  # never keep a link we cannot safely show
            domain = get_domain(url)
            sources.append(
                Source(
                    title=truncate(str(item.get("title") or domain), 200),
                    url=url,
                    source_name=domain,
                    published_date=item.get("published_date") or None,
                    snippet=str(item.get("content") or "").strip(),
                    source_type=classify_source(domain),
                )
            )
        return sources


PROVIDERS = {"tavily": TavilySearchProvider}


def get_search_provider() -> SearchProvider:
    """Build the provider named by SEARCH_PROVIDER (default: tavily)."""
    name = (os.getenv("SEARCH_PROVIDER") or "tavily").strip().lower()
    if name not in PROVIDERS:
        raise SearchError(f"Unknown SEARCH_PROVIDER '{name}'. Available: {', '.join(PROVIDERS)}.")
    return PROVIDERS[name]()
