
from app.core.config import settings
from app.services.qdrant_service import semantic_search, keyword_search, merge_results
from app.services.embeddings_service import generate_embedding
async def search_documents(query: str, top_k: int = 5):
    """
    Unified search pipeline with support for semantic-only or hybrid.
    """
    mode = settings.SEARCH_MODE

    if mode == "semantic":
        # ✅ Only semantic search
        query_vector = await generate_embedding(query)
        return await semantic_search(query_vector, top_k=top_k)

    elif mode == "hybrid":
        # ✅ Run both semantic + keyword search
        query_vector = await generate_embedding(query)
        semantic_results = await semantic_search(query_vector, top_k=top_k)
        keyword_results = await keyword_search(query, top_k=top_k)

        # Merge + rerank
        return merge_results(semantic_results, keyword_results, top_k=top_k)

    else:
        raise ValueError(f"Invalid SEARCH_MODE: {mode}. Use 'semantic' or 'hybrid'.")