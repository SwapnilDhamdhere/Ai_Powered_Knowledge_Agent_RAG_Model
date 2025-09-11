import httpx
from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import OllamaConnectionError
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import List

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
async def _call_ollama_batch(inputs: List[str]):
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(settings.OLLAMA_EMBEDDINGS_URL, json={"model": settings.OLLAMA_EMBEDDINGS_MODEL, "input": inputs})
        resp.raise_for_status()
        return resp.json()

async def generate_embedding(text: str) -> List[float]:
    """Compatibility wrapper for single text input."""
    data = await _call_ollama_batch([text])
    embeddings = data.get("embeddings") or data.get("embedding")
    if not embeddings:
        raise OllamaConnectionError("No embedding returned")
    # If embeddings is a list of lists, return first
    if isinstance(embeddings[0], list):
        return embeddings[0]
    return embeddings

async def generate_embeddings_batch(texts: List[str], batch_size: int = None) -> List[List[float]]:
    batch_size = batch_size or settings.EMBEDDINGS_BATCH_SIZE
    embeddings = []
    for i in range(0, len(texts), batch_size):
        sub = texts[i:i+batch_size]
        data = await _call_ollama_batch(sub)
        batch_embeddings = data.get("embeddings") or data.get("embedding")
        if not batch_embeddings:
            raise OllamaConnectionError("No embeddings in batch response")
        # If Ollama returns single embedding per input: append directly
        # If response is nested, flatten accordingly
        if isinstance(batch_embeddings[0], list):
            embeddings.extend(batch_embeddings)
        else:
            embeddings.extend([[v] for v in batch_embeddings])  # fallback
    return embeddings