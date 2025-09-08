import httpx
from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import OllamaConnectionError

# async def generate_embedding(text: str) -> list:
#     """
#     Generate embeddings for a given text using Ollama embeddings API.
#     """
#     try:
#         async with httpx.AsyncClient(timeout=60) as client:
#             response = await client.post(
#                 settings.OLLAMA_EMBEDDINGS_URL,
#                 json={"model": settings.OLLAMA_EMBEDDINGS_MODEL, "input": text},
#             )
#             response.raise_for_status()
#             data = response.json()
#             return data.get("embedding")
#     except Exception as e:
#         logger.error(f"Failed to generate embedding: {e}")
#         raise OllamaConnectionError(f"Ollama embeddings API error: {e}")

async def generate_embedding(text: str) -> list:
    """
    Generate embeddings for a given text using Ollama embeddings API.
    """
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                settings.OLLAMA_EMBEDDINGS_URL,
                json={
                    "model": settings.OLLAMA_EMBEDDINGS_MODEL,
                    "input": text
                },
            )
            response.raise_for_status()
            data = response.json()

            # ✅ FIX: Extract the correct embedding from Ollama response
            if "embeddings" in data and len(data["embeddings"]) > 0:
                embedding = data["embeddings"][0]

                # ✅ Safety check: ensure it's a list of floats
                if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                    return embedding
                else:
                    logger.error(f"Invalid embedding format: {embedding}")
                    raise OllamaConnectionError("Invalid embedding format received from Ollama")
            else:
                logger.error(f"Unexpected Ollama embedding response: {data}")
                raise OllamaConnectionError("No embeddings found in Ollama response")

    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise OllamaConnectionError(f"Ollama embeddings API error: {e}")