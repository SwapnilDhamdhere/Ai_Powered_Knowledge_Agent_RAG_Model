from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import QdrantConnectionError

# Initialize Qdrant client
qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

async def init_qdrant_collection():
    """
    Create a Qdrant collection if it doesn't exist.
    """
    try:
        collections = qdrant_client.get_collections().collections
        existing = [c.name for c in collections]

        if settings.QDRANT_COLLECTION not in existing:
            logger.info(f"Creating Qdrant collection: {settings.QDRANT_COLLECTION}")
            qdrant_client.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.QDRANT_VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
        else:
            logger.info(f"Qdrant collection already exists: {settings.QDRANT_COLLECTION}")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collection: {e}")
        raise QdrantConnectionError(f"Qdrant initialization failed: {e}")

# async def insert_embeddings(vectors: list, payloads: list):
#     """
#     Insert embeddings into Qdrant with associated metadata.
#     """
#     try:
#         points = [
#             PointStruct(id=i, vector=vectors[i], payload=payloads[i])
#             for i in range(len(vectors))
#         ]
#         qdrant_client.upsert(
#             collection_name=settings.QDRANT_COLLECTION,
#             points=points,
#         )
#         logger.info(f"Inserted {len(points)} embeddings into Qdrant.")
#     except Exception as e:
#         logger.error(f"Failed to insert embeddings: {e}")
#         raise QdrantConnectionError(f"Qdrant insert failed: {e}")

async def insert_embeddings(vectors: list, payloads: list):
    """
    Insert embeddings into Qdrant with validation and associated metadata.
    """
    try:
        # ✅ Safety check: Ensure embeddings are valid
        for idx, vec in enumerate(vectors):
            if not isinstance(vec, list):
                logger.error(f"Invalid vector at index {idx}: Expected list, got {type(vec)}")
                raise QdrantConnectionError(f"Invalid vector format at index {idx}")

            if not all(isinstance(x, (float, int)) for x in vec):
                logger.error(f"Invalid vector at index {idx}: Vector contains non-float values")
                raise QdrantConnectionError(f"Invalid vector values at index {idx}")

            if len(vec) != settings.QDRANT_VECTOR_SIZE:
                logger.error(
                    f"Vector size mismatch at index {idx}: "
                    f"Expected {settings.QDRANT_VECTOR_SIZE}, got {len(vec)}"
                )
                raise QdrantConnectionError(
                    f"Vector size mismatch at index {idx}: Expected {settings.QDRANT_VECTOR_SIZE}, got {len(vec)}"
                )

        # ✅ Create Qdrant points
        points = [
            PointStruct(
                id=payloads[i]["id"],  # Use UUID instead of index for better uniqueness
                vector=vectors[i],
                payload=payloads[i]
            )
            for i in range(len(vectors))
        ]

        # ✅ Insert into Qdrant
        qdrant_client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
        )
        logger.info(f"Inserted {len(points)} embeddings into Qdrant successfully.")

    except Exception as e:
        logger.error(f"Failed to insert embeddings into Qdrant: {e}")
        raise QdrantConnectionError(f"Qdrant insert failed: {e}")

async def semantic_search(query_vector: list, top_k: int = 5, filter_payload: dict = None):
    """
    Perform semantic search in Qdrant using a query vector.
    """
    try:
        search_filter = None
        if filter_payload:
            search_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in filter_payload.items()
                ]
            )

        result = qdrant_client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
        )
        return result
    except Exception as e:
        logger.error(f"Qdrant semantic search failed: {e}")
        raise QdrantConnectionError(f"Qdrant search failed: {e}")
