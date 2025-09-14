import re
import numpy as np
from typing import List
from app.services.embeddings_service import generate_embeddings_batch
from app.core.config import settings


async def semantic_chunk_text(text: str, max_tokens: int = 512, similarity_threshold: float = 0.75) -> List[str]:
    """
    Split text into semantically coherent chunks using embeddings similarity.

    Args:
        text: Raw input text.
        max_tokens: Max tokens/characters per chunk.
        similarity_threshold: Cosine similarity threshold to decide new chunk.

    Returns:
        List of semantic chunks.
    """
    # 1. Sentence split
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    # 2. Generate embeddings for all sentences
    embeddings = await generate_embeddings_batch(sentences, batch_size=settings.EMBEDDINGS_BATCH_SIZE)

    # 3. Build chunks dynamically
    chunks, current_chunk = [], sentences[0]
    last_vec = embeddings[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(last_vec, embeddings[i])
        candidate = current_chunk + " " + sentences[i]

        if sim < similarity_threshold or len(candidate) > max_tokens:
            # commit old chunk
            chunks.append(current_chunk.strip())
            current_chunk = sentences[i]
        else:
            current_chunk = candidate

        last_vec = embeddings[i]

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def cosine_similarity(vec1, vec2) -> float:
    """Compute cosine similarity between two vectors."""
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))