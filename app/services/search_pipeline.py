
from app.core.config import settings
from app.services.qdrant_service import semantic_search, keyword_search, merge_results

def search_documents(query: str, top_k: int = 5):
    """
    Unified search pipeline with support for semantic-only or hybrid.
    """
    mode = settings.SEARCH_MODE

    if mode == "semantic":
        # ✅ Only semantic search
        return semantic_search(query, top_k=top_k)

    elif mode == "hybrid":
        # ✅ Run both semantic + keyword search
        semantic_results = semantic_search(query, top_k=top_k)
        keyword_results = keyword_search(query, top_k=top_k)

        # Merge + rerank
        return merge_results(semantic_results, keyword_results, top_k=top_k)

    else:
        raise ValueError(f"Invalid SEARCH_MODE: {mode}. Use 'semantic' or 'hybrid'.")