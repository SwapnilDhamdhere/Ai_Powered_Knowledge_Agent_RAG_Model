import httpx
import asyncio
from typing import List, Dict
from bs4 import BeautifulSoup
from app.core.logger import logger
from app.utils.helpers import clean_text

DUCKDUCKGO_HTML = "https://html.duckduckgo.com/html/"

async def _duckduckgo_search_page(query: str, max_results: int = 8) -> List[Dict]:
    """Scrape DuckDuckGo HTML results. Returns title, url, snippet."""
    params = {"q": query}
    headers = {"User-Agent": "ai-knowledge-agent/1.0"}
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            r = await client.post(DUCKDUCKGO_HTML, data=params)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")

            results = []
            for res in soup.select(".result")[:max_results]:
                a = res.select_one("a.result__a") or res.select_one("a")
                snippet_el = res.select_one(".result__snippet") or res.select_one(".result__extras")
                title = a.get_text(strip=True) if a else ""
                href = a["href"] if a and a.has_attr("href") else ""
                snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
                results.append({"title": title, "url": href, "snippet": clean_text(snippet)})
            return results
    except Exception as e:
        logger.error("DuckDuckGo search failed: %s", e)
        return []

async def _fetch_page_text(url: str, timeout: float = 10.0) -> str:
    """Fetch and clean visible text from a webpage."""
    headers = {"User-Agent": "ai-knowledge-agent/1.0"}
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            r = await client.get(url)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            for s in soup(["script", "style", "noscript"]):
                s.decompose()
            text = soup.get_text(separator=" ", strip=True)
            return clean_text(text)
    except Exception as e:
        logger.debug("Failed to fetch page %s: %s", url, e)
        return ""

async def web_search(
    query: str,
    max_results: int = 6,
    fetch_full_pages: bool = False,
    page_fetch_limit: int = 3,
) -> List[Dict]:
    """Perform web search and return list of dicts: title, url, snippet, content."""
    results = await _duckduckgo_search_page(query, max_results=max_results)

    if not fetch_full_pages:
        for r in results:
            r["content"] = r.get("snippet", "")
        return results

    to_fetch = results[:page_fetch_limit]
    tasks = [_fetch_page_text(r["url"]) for r in to_fetch]
    fetched_texts = await asyncio.gather(*tasks, return_exceptions=True)

    for i, r in enumerate(to_fetch):
        content = fetched_texts[i] if isinstance(fetched_texts[i], str) else ""
        r["content"] = content or r.get("snippet", "")

    for r in results[page_fetch_limit:]:
        r["content"] = r.get("snippet", "")

    return results
