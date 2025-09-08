from fastapi import APIRouter, HTTPException
from app.models.query_model import AskQuery, AskResponse
from app.services.search_service import search_knowledge_base
from app.utils.helpers import clean_text
from app.core.logger import logger

router = APIRouter()

@router.post("/", response_model=AskResponse)
async def ask_ai(query: AskQuery):
    """
    Ask a question and get AI-generated answers based on documents stored in Qdrant.
    """
    try:
        # ✅ Log the incoming query
        logger.info(f"Received query: {query.query}")

        # ✅ Clean the query text for better embeddings & search results
        cleaned_query = clean_text(query.query)

        # ✅ Perform semantic search & LLM-based answer generation
        result = await search_knowledge_base(cleaned_query)

        # ✅ Return the structured AI response
        return AskResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process query '{query.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch answer: {e}")
