from typing import List, Dict
from ddgs import DDGS
from app.core.logger import logger
from app.services.ollama_service import generate_answer

async def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Perform a quick DuckDuckGo search (using ddgs).
    Returns simplified sources with minimal metadata.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        sources = []
        for r in results:
            sources.append({
                "document": r.get("title", "") or r.get("body", ""),
                "url": r.get("href", ""),
                "chunks_used": [0],
                "relevance": 0.8  # dummy relevance for now
            })
        logger.info(f"🌍 DDGS results fetched: {len(sources)} items")
        return sources

    except Exception as e:
        logger.error(f"❌ Web search failed: {e}")
        return [{
            "document": f"Error: {e}",
            "url": "",
            "chunks_used": [],
            "relevance": 0.0
        }]


def clean_text(text: str) -> str:
    """Basic text cleaner."""
    return (text or "").strip()

async def summarize_text(text: str, max_length: int = 300) -> str:
    """
    Summarize a long text to a shorter version.
    Uses GPT-OSS (or any LLM) for summarization.
    """
    if len(text) <= max_length:
        return text

    prompt = f"Summarize the following text in a concise way (max {max_length} chars):\n\n{text}"
    try:
        summary = await generate_answer(context=text, query=prompt)
        return summary.strip()
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        return text[:max_length]  # fallback: truncate