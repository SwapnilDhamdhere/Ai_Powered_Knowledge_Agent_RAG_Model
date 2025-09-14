from qdrant_client import AsyncQdrantClient
from app.core.config import settings
from app.core.logger import logger

_async_client = None

def get_async_qdrant_client() -> AsyncQdrantClient:
    global _async_client
    if _async_client is None:
        logger.info("Creating AsyncQdrantClient...")
        _async_client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            prefer_grpc=False  # set True if you configured gRPC
        )
    return _async_client