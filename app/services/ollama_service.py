import httpx
import json
from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import OllamaConnectionError

# Ollama API endpoint (local)
OLLAMA_BASE_URL = "http://localhost:11434"

async def generate_answer(context: str, query: str) -> str:
    """
    Generate a contextual answer using Ollama LLM.
    Returns "NO_ANSWER" if the LLM cannot answer based on the context.
    """
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
                settings.OLLAMA_CHAT_URL,
                json={
                    "model": settings.OLLAMA_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful AI assistant. "
                                "Always return clean plain text only. "
                                "Do not use Markdown, tables, checklists, headings, or special formatting. "
                                "If you do not know the answer based on the context, reply 'NO_ANSWER'."
                            )
                        },
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
                    ],
                    "stream": False
                }
            )

            response.raise_for_status()
            raw_response = response.text.strip()

            # Parse JSON response
            try:
                data = json.loads(raw_response)
            except json.JSONDecodeError:
                logger.error(f"Unexpected Ollama response: {raw_response}")
                raise OllamaConnectionError("Ollama returned an invalid JSON response")

            answer = data.get("message", {}).get("content", "").strip()
            if not answer:
                answer = "NO_ANSWER"  # fallback if empty
            return answer

    except httpx.RequestError as e:
        logger.error(f"❌ Ollama connection failed: {e}")
        raise OllamaConnectionError(f"Ollama LLM connection error: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ Ollama HTTP error: {e.response.text}")
        raise OllamaConnectionError(f"Ollama LLM API error: {e.response.text}")
    except Exception as e:
        logger.error(f"❌ Failed to generate answer from Ollama: {e}")
        raise OllamaConnectionError(f"Ollama LLM API error: {e}")

async def ollama_health_check() -> bool:
    """
    Check if Ollama API is up and running.
    Returns True if healthy, False otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                logger.info("✅ Ollama is running and reachable")
                return True
            else:
                logger.warning(
                    f"⚠️ Ollama health check failed. Status: {response.status_code}"
                )
                return False
    except Exception as e:
        logger.error(f"❌ Ollama is not reachable: {e}")
        return False