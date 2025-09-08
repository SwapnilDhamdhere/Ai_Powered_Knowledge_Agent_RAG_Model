from pydantic import BaseModel
from typing import List


class AskQuery(BaseModel):
    """
    Request model for the /api/ask endpoint.
    """
    query: str

class SourceInfo(BaseModel):
    """
    Represents a reference to a document chunk used in the answer.
    """
    document: str
    chunks_used: List[int]
    relevance: float  # Optional: can represent semantic similarity score

class AskResponse(BaseModel):
    """
    Response model for the /api/ask endpoint.
    """
    answer: str
    sources: List[SourceInfo]  # Structured list of document references
    generated_by: str          # "Hybrid (Docs + AI)" or "AI-only"
    confidence: float          # Optional: overall confidence score of the answer