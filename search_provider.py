"""Search provider abstraction + SerperProvider + LruTtlCache."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import aiohttp

try:
    from .config import SERPER_API_KEY, MAX_RESULTS_PER_QUERY
    from .utils import clean_snippet, extract_domain
except ImportError:
    from config import SERPER_API_KEY, MAX_RESULTS_PER_QUERY
    from utils import clean_snippet, extract_domain

logger = logging.getLogger(__name__)


# ── LRU + TTL Cache ─────────────────────────────────────────────────

class LruTtlCache:
    """LRU cache with per-entry TTL expiration."""

    def __init__(self, maxsize: int = 5000, ttl: int = 86400):
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def _make_key(self, query: str, hl: str, gl: str, num: int, page: int) -> str:
        raw = f"{query.strip().lower()}|{hl}|{gl}|{num}|{page}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, hl: str = "en", gl: str = "us",
            num: int = 10, page: int = 1) -> list[dict] | None:
        key = self._make_key(query, hl, gl, num, page)
        if key not in self._cache:
            return None
        ts, value = self._cache[key]
        if time.time() - ts > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def put(self, query: str, results: list[dict], hl: str = "en",
            gl: str = "us", num: int = 10, page: int = 1) -> None:
        key = self._make_key(query, hl, gl, num, page)
        self._cache[key] = (time.time(), results)
        self._cache.move_to_end(key)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


# ── Search Result type ───────────────────────────────────────────────

class SearchResult:
    """Normalized search result."""

    def __init__(self, title: str, url: str, snippet: str,
                 source_type: str = "organic"):
        self.title = title or ""
        self.url = url or ""
        self.snippet = clean_snippet(snippet) if snippet else ""
        self.domain = extract_domain(url)
        self.source_type = source_type

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "domain": self.domain,
            "source_type": self.source_type,
        }


# ── Search Provider ABC ──────────────────────────────────────────────

class SearchProvider(ABC):
    """Abstract base class for all search providers."""

    @abstractmethod
    async def search(self, query: str, num: int = 8,
                     hl: str | None = None, gl: str | None = None,
                     page: int = 1) -> list[dict]:
        ...


# ── Serper Provider ──────────────────────────────────────────────────

class SerperProvider(SearchProvider):
    """Google Search via Serper.dev API with caching and retry."""

    ENDPOINT = "https://google.serper.dev/search"
    MAX_RETRIES = 2
    BACKOFF_TIMES = [0.5, 1.5]

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or SERPER_API_KEY
        self.cache = LruTtlCache()

    async def search(self, query: str, num: int = MAX_RESULTS_PER_QUERY,
                     hl: str | None = None, gl: str | None = None,
                     page: int = 1) -> list[dict]:
        """Execute search. Returns list of result dicts."""
        if not hl:
            # Use 30% Chinese character ratio threshold (aligned with utils.parse_question)
            chars = query.replace(" ", "")
            chinese_ratio = sum(1 for c in chars if '\u4e00' <= c <= '\u9fff') / max(len(chars), 1)
            hl = "zh-cn" if chinese_ratio > 0.3 else "en"
        if not gl:
            gl = "cn" if hl == "zh-cn" else "us"

        # Check cache
        cached = self.cache.get(query, hl, gl, num, page)
        if cached is not None:
            logger.debug(f"Cache hit for query: {query[:50]}")
            return cached

        # Execute search with retry
        results = await self._search_with_retry(query, num, hl, gl, page)

        # Cache results
        result_dicts = [r.to_dict() for r in results]
        self.cache.put(query, result_dicts, hl, gl, num, page)
        return result_dicts

    async def _search_with_retry(self, query: str, num: int,
                                 hl: str, gl: str,
                                 page: int) -> list[SearchResult]:
        """Search with exponential backoff retry on transient errors."""
        last_err = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return await self._do_search(query, num, hl, gl, page)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(self.BACKOFF_TIMES[attempt])
            except SerperHTTPError as e:
                last_err = e
                if e.status in (429, 500, 502, 503, 504) and attempt < self.MAX_RETRIES:
                    await asyncio.sleep(self.BACKOFF_TIMES[attempt])
                else:
                    break

        logger.error(f"Search failed after retries: {last_err}")
        return []

    async def _do_search(self, query: str, num: int,
                         hl: str, gl: str, page: int) -> list[SearchResult]:
        """Single search request to Serper API."""
        payload = {
            "q": query,
            "num": num,
            "hl": hl,
            "gl": gl,
            "page": page,
            "autocorrect": True,
        }
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.ENDPOINT,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise SerperHTTPError(resp.status, body)
                data = await resp.json()

        return self._parse_response(data)

    def _parse_response(self, data: dict) -> list[SearchResult]:
        """Parse Serper API response into SearchResult objects."""
        results: list[SearchResult] = []

        # Answer box
        ab = data.get("answerBox")
        if ab:
            snippet = ab.get("answer") or ab.get("snippet") or ab.get("title", "")
            results.append(SearchResult(
                title=ab.get("title", "Answer Box"),
                url=ab.get("link", ""),
                snippet=snippet,
                source_type="answerBox",
            ))

        # Knowledge graph
        kg = data.get("knowledgeGraph")
        if kg:
            desc = kg.get("description", "")
            title = kg.get("title", "")
            attrs = kg.get("attributes", {})
            attr_text = "; ".join(f"{k}: {v}" for k, v in attrs.items()) if attrs else ""
            snippet = f"{desc} {attr_text}".strip()
            results.append(SearchResult(
                title=title,
                url=kg.get("descriptionLink", ""),
                snippet=snippet,
                source_type="knowledgeGraph",
            ))

        # Organic results
        for item in data.get("organic", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source_type="organic",
            ))

        return results


class SerperHTTPError(Exception):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"Serper HTTP {status}: {body[:200]}")


# ── Factory + module-level singleton ─────────────────────────────────

def create_search_provider() -> SearchProvider:
    """Create the search provider (Serper)."""
    return SerperProvider()


# Module-level singleton, created via factory
default_provider: SearchProvider = create_search_provider()

