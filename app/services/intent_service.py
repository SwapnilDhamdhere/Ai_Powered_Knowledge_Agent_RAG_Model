import json
import httpx
from app.core.logger import logger
from app.core.config import settings

class IntentService:
    @staticmethod
    async def classify_intent(query: str) -> str:
        """
        Classify the intent of a user query using Ollama LLM.
        Returns 'General' if unable to detect.
        """
        if not query or not query.strip():
            logger.warning("Empty query received for intent classification.")
            return "General"

        prompt = IntentService._build_prompt(query)

        try:
            async with httpx.AsyncClient(timeout=getattr(settings, "REQUEST_TIMEOUT", 20)) as client:
                response = await client.post(
                    f"{settings.OLLAMA_HOST}/api/chat",
                    json={
                        "model": settings.OLLAMA_INTENT_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False
                    }
                )

                response.raise_for_status()
                raw_response = response.text.strip()

                # Parse JSON from Ollama
                try:
                    logger.info("Searching intent detection...")
                    data = json.loads(raw_response)
                    content = data.get("message", {}).get("content", "").strip()
                    return content or "General"
                except json.JSONDecodeError:
                    logger.warning(f"Ollama returned non-JSON: {raw_response}")
                    return raw_response.splitlines()[-1].strip()

        except httpx.RequestError as e:
            logger.error(f"Ollama connection failed: {e}")
            return "ErrorConnection"
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.text}")
            return "ErrorHTTP"
        except Exception as e:
            logger.exception(f"Unexpected error during intent detection: {e}")
            return "General"

    @staticmethod
    def _build_prompt(query: str) -> str:
        return f"""
                You are an intent classification system.
                Your task is to analyze the following user query and determine its true primary intent in one or two words.

                Focus on capturing the actual purpose or motivation behind the query, using natural, meaningful terms (not from a fixed list).

                Respond with only the intent label — short and descriptive
                (e.g., Information Request, Error Diagnosis, System Control, Cost Inquiry, Product Comparison, etc.).

                User Query: "{query}"

                Intent:
        """