from pydantic import BaseModel, Field
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
    chunks_used: List[int] = Field(default_factory=list)  # ✅ safe default
    relevance: float = 0.0  # ✅ default to 0.0 so None won’t break validation


class AskResponse(BaseModel):
    """
    Response model for the /api/ask endpoint.
    """
    answer: str
    sources: List[SourceInfo] = Field(default_factory=list)  # ✅ safe default
    generated_by: str          # "Hybrid (Docs + AI)" or "AI-only"
    confidence: float = 0.0    # ✅ default confidence
