from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.core.logger import logger
from app.core.config import settings

# Build Qdrant URL dynamically
QDRANT_URL = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"

# Create a single reusable Qdrant client instance
qdrant_client = QdrantClient(url=QDRANT_URL)

def ensure_collection():
    """Ensure that the target Qdrant collection exists."""
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if settings.QDRANT_COLLECTION not in collection_names:
            logger.info(f"Creating missing collection: {settings.QDRANT_COLLECTION}")

            # Use VectorParams instead of a raw dict ✅
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.QDRANT_VECTOR_SIZE,
                    distance=Distance.COSINE if settings.QDRANT_DISTANCE.upper() == "COSINE" else Distance.EUCLID
                )
            )
            logger.info(f"Collection '{settings.QDRANT_COLLECTION}' created successfully ✅")
        else:
            logger.info(f"Collection '{settings.QDRANT_COLLECTION}' already exists ✅")

    except Exception as e:
        logger.error(f"Failed to ensure Qdrant collection: {e}")
        raise

def init_qdrant():
    """
    Backward compatibility: Initialize Qdrant client and ensure the collection.
    """
    logger.info("Initializing Qdrant client...")
    ensure_collection()
    logger.info("Qdrant client initialized ✅")
