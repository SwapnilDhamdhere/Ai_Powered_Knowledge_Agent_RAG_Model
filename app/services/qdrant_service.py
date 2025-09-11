from typing import List, Optional
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from app.core.config import settings
from app.core.logger import logger
from app.db.async_qdrant import get_async_qdrant_client
from app.core.exceptions import QdrantConnectionError

# Global async client
client = get_async_qdrant_client()


async def ensure_collection():
    """
    Ensure collection exists and create with HNSW config if missing.
    """
    try:
        existing = await client.get_collections()
        names = [c.name for c in existing.collections]

        if settings.QDRANT_COLLECTION in names:
            logger.info("Qdrant collection exists.")
            return

        logger.info("Creating Qdrant collection with HNSW settings...")

        await client.recreate_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.QDRANT_VECTOR_SIZE,
                distance=Distance.COSINE
                if settings.QDRANT_DISTANCE.upper() == "COSINE"
                else Distance.EUCLID,
            ),
            hnsw_config=HnswConfigDiff(
                m=settings.QDRANT_HNSW_M,
                ef_construct=settings.QDRANT_HNSW_EF_CONSTRUCT,
                full_scan_threshold=settings.QDRANT_FULL_SCAN_THRESHOLD,
            ),
        )
        logger.info(
            "Qdrant collection created (m=%d, ef_construct=%d, full_scan_threshold=%d).",
            settings.QDRANT_HNSW_M,
            settings.QDRANT_HNSW_EF_CONSTRUCT,
            settings.QDRANT_FULL_SCAN_THRESHOLD,
        )
    except Exception as e:
        logger.exception("Failed to ensure Qdrant collection: %s", e)
        raise QdrantConnectionError(str(e))


async def upsert_points(points: List[PointStruct], batch_size: Optional[int] = None):
    """
    Upsert points in batches.
    """
    try:
        batch_size = batch_size or settings.QDRANT_UPSERT_BATCH_SIZE
        n = len(points)
        for i in range(0, n, batch_size):
            chunk = points[i : i + batch_size]
            await client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=chunk,
            )
        logger.info("Upserted %d points to Qdrant", n)
    except Exception as e:
        logger.exception("Qdrant upsert failed: %s", e)
        raise QdrantConnectionError(str(e))


async def semantic_search(
    query_vector: List[float], top_k: int = 8, filter_payload: dict = None
):
    """
    Perform semantic search using AsyncQdrantClient.
    """
    try:
        search_filter = None
        if filter_payload:
            search_filter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filter_payload.items()
                ]
            )
        result = await client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
        )
        return result
    except Exception as e:
        logger.exception("Qdrant search failed: %s", e)
        raise QdrantConnectionError(str(e))
