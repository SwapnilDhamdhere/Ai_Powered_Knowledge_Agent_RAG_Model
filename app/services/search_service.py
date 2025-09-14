# import numpy as np
# from collections import defaultdict
# from typing import List, Dict
#
# from app.services.embeddings_service import generate_embedding, generate_embeddings_batch
# from app.services.qdrant_service import semantic_search
# from app.services.ollama_service import generate_answer
# from app.services.web_service import web_search, clean_text
# from app.core.logger import logger
# from app.core.config import settings
# from app.models.query_model import AskResponse, SourceInfo
#
# # 🔧 Tuning parameters
# MIN_RELEVANCE = 0.6   # Minimum similarity score to keep a chunk
# TOP_K = 8             # Default top-k search
# MIN_CHUNKS = 3        # Minimum chunks required to consider knowledge-based answer
#
#
# def build_sources(raw_results: List[Dict]) -> List[SourceInfo]:
#     sources = []
#     for item in raw_results:
#         sources.append(
#             SourceInfo(
#                 document=item.get("document") or item.get("url") or "Unknown source",
#                 chunks_used=item.get("chunks_used", []),
#                 relevance=item.get("relevance", 0.0),
#             )
#         )
#     return sources
#
#
# async def search_knowledge_base(query: str, mode: str = None, top_k: int = TOP_K, min_chunks: int = MIN_CHUNKS) -> AskResponse:
#     mode = (mode or settings.SEARCH_MODE).lower()
#     logger.info(f"Search mode: {mode}")
#
#     # ONLINE mode
#     if mode == "online":
#         return await _online_search(query, top_k)
#
#     # SEMANTIC / HYBRID
#     logger.info("Generating embedding for query...")
#     query_vector = await generate_embedding(query)
#
#     logger.info(f"Searching Qdrant (top_k={top_k})...")
#     search_results = await semantic_search(query_vector, top_k=top_k)
#
#     if not search_results or len(search_results) < min_chunks:
#         logger.warning(f"Found only {len(search_results)} chunks, retrying with top_k={min_chunks}...")
#         search_results = await semantic_search(query_vector, top_k=min_chunks)
#
#     if not search_results:
#         return await _ai_then_web_fallback(query)
#
#     # Collect high-relevance chunks
#     top_chunks = []
#     source_map = defaultdict(list)
#     relevance_map = defaultdict(list)
#     for hit in search_results:
#         score = getattr(hit, "score", 0)
#         if score < MIN_RELEVANCE:
#             continue
#         content = hit.payload.get("content")
#         source = hit.payload.get("source")
#         chunk_index = hit.payload.get("chunk_index")
#         if content:
#             top_chunks.append(content)
#         if source:
#             source_map[source].append(chunk_index)
#             relevance_map[source].append(score)
#
#     if not top_chunks:
#         return await _ai_then_web_fallback(query)
#
#     # Generate answer using GPT-OSS
#     context = "\n\n".join(top_chunks)
#     logger.info("Generating contextual answer using GPT-OSS...")
#     answer = await generate_answer(context=context, query=query)
#     if not answer.strip() or answer.strip().upper() == "NO_ANSWER":
#         return await _ai_then_web_fallback(query)
#
#     # Build sources
#     raw_sources = [
#         {"document": doc, "chunks_used": sorted(set(idxs)), "relevance": round(max(relevance_map[doc]), 2)}
#         for doc, idxs in source_map.items()
#     ]
#     sources = build_sources(raw_sources)
#     overall_confidence = round(max([s.relevance for s in sources], default=0.0), 2)
#
#     return AskResponse(
#         answer=answer.strip() or "NO_ANSWER",
#         sources=sources,
#         generated_by="Hybrid (Docs + AI)",
#         confidence=overall_confidence
#     )
#
# # ------------------------------
# # Helper functions
# # ------------------------------
# async def _online_search(query: str, top_k: int) -> AskResponse:
#     web_results = await web_search(query, max_results=top_k)
#     logger.info(f"_online_search results: {len(web_results)} items")
#
#     if not web_results:
#         answer = await generate_answer(context="", query=query)
#         return AskResponse(
#             answer=answer.strip() or "NO_ANSWER",
#             sources=[],
#             generated_by="AI-only",
#             confidence=0.0,
#         )
#
#     context_texts = []
#     raw_sources = []
#
#     for r in web_results:
#         text = clean_text(r.get("content") or r.get("snippet") or r.get("title") or "")
#         if text:
#             context_texts.append(text)
#         raw_sources.append({
#             "document": r.get("title") or r.get("url") or "Unknown",
#             "chunks_used": [],
#             "relevance": 0.0
#         })
#
#     full_context = "\n\n".join(context_texts)
#     answer = await generate_answer(context=full_context, query=query)
#     if not answer.strip():
#         answer = full_context if full_context else "NO_ANSWER"
#
#     return AskResponse(
#         answer=answer,
#         sources=build_sources(raw_sources),
#         generated_by="Online + AI",
#         confidence=0.0
#     )
#
#
# async def _ai_then_web_fallback(query: str) -> AskResponse:
#     """Fallback: Use all web content + AI to generate answer if Qdrant fails."""
#     logger.info("Attempting AI + Web fallback...")
#
#     web_results = await web_search(query, max_results=TOP_K)
#     logger.info(f"Web search returned {len(web_results)} items")
#
#     if not web_results:
#         answer = await generate_answer(context="", query=query)
#         return AskResponse(
#             answer=answer.strip() or "NO_ANSWER",
#             sources=[],
#             generated_by="AI-only",
#             confidence=0.0
#         )
#
#     context_texts = []
#     raw_sources = []
#
#     for r in web_results:
#         # Use snippet or content; fallback to title only if nothing else
#         text = (r.get("content") or r.get("snippet") or "").strip()
#         if text:  # only include non-empty text
#             context_texts.append(clean_text(text))
#         raw_sources.append({
#             "document": r.get("title") or r.get("url") or "Unknown",
#             "chunks_used": [],
#             "relevance": 0.0
#         })
#
#     if not context_texts:
#         # As last resort, pass titles/urls as context
#         context_texts = [r.get("title") or r.get("url") or "" for r in web_results]
#
#     full_context = "\n\n".join(context_texts)
#     print(f"Full context: {full_context}")
#
#     # Generate answer using GPT
#     answer = await generate_answer(context=full_context, query=query)
#     if not answer.strip() or answer.strip().upper() == "NO_ANSWER":
#         # fallback to raw web content if GPT returns NO_ANSWER
#         answer = full_context if full_context else "NO_ANSWER"
#
#     return AskResponse(
#         answer=answer,
#         sources=build_sources(raw_sources),
#         generated_by="Web + AI fallback",
#         confidence=0.0
#     )

import numpy as np
from collections import defaultdict
from typing import List, Dict

from app.services.embeddings_service import generate_embedding
from app.services.qdrant_service import semantic_search
from app.services.ollama_service import generate_answer
from app.services.web_service import web_search, clean_text, summarize_text
from app.core.logger import logger
from app.core.config import settings
from app.models.query_model import AskResponse, SourceInfo

# 🔧 Tuning parameters
MIN_RELEVANCE = 0.6   # Minimum similarity score to keep a chunk
TOP_K = 8             # Default top-k search
MIN_CHUNKS = 3        # Minimum chunks required to consider knowledge-based answer
MAX_CHUNK_LENGTH = 1000  # Max chars per chunk for summarization

def build_sources(raw_results: List[Dict]) -> List[SourceInfo]:
    """Deduplicate and build sources"""
    seen = set()
    sources = []
    for item in raw_results:
        doc = item.get("document") or "Unknown source"
        if doc in seen:
            continue
        seen.add(doc)
        sources.append(
            SourceInfo(
                document=doc,
                chunks_used=item.get("chunks_used", []),
                relevance=item.get("relevance", 0.0),
            )
        )
    return sources

async def search_knowledge_base(query: str, mode: str = None, top_k: int = TOP_K, min_chunks: int = MIN_CHUNKS) -> AskResponse:
    mode = (mode or settings.SEARCH_MODE).lower()
    logger.info(f"Search mode: {mode}")

    # ONLINE mode
    if mode == "online":
        return await _online_search(query, top_k)

    # SEMANTIC / HYBRID
    logger.info("Generating embedding for query...")
    query_vector = await generate_embedding(query)

    logger.info(f"Searching Qdrant (top_k={top_k})...")
    search_results = await semantic_search(query_vector, top_k=top_k)

    if not search_results or len(search_results) < min_chunks:
        logger.warning(f"Found only {len(search_results)} chunks, retrying with top_k={min_chunks}...")
        search_results = await semantic_search(query_vector, top_k=min_chunks)

    if search_results:
        # Qdrant succeeded → generate answer using GPT with Qdrant context
        top_chunks, source_map, relevance_map = [], defaultdict(list), defaultdict(list)
        for hit in search_results:
            score = getattr(hit, "score", 0)
            if score < MIN_RELEVANCE:
                continue
            content = hit.payload.get("content")
            source = hit.payload.get("source")
            chunk_index = hit.payload.get("chunk_index")
            if content:
                top_chunks.append(content)
            if source:
                source_map[source].append(chunk_index)
                relevance_map[source].append(score)

        context = "\n\n".join(top_chunks)
        answer = await generate_answer(context=context, query=query)

        if not answer.strip() or answer.strip().upper() == "NO_ANSWER":
            # If AI with Qdrant fails → fallback to AI-only
            return await _ai_only(query)

        raw_sources = [
            {"document": doc, "chunks_used": sorted(set(idxs)), "relevance": round(max(relevance_map[doc]), 2)}
            for doc, idxs in source_map.items()
        ]
        sources = build_sources(raw_sources)
        overall_confidence = round(max([s.relevance for s in sources], default=0.0), 2)

        return AskResponse(
            answer=answer.strip() or "NO_ANSWER",
            sources=sources,
            generated_by="Hybrid (Docs + AI)",
            confidence=overall_confidence
        )
    else:
        # No Qdrant results → AI-only first
        return await _ai_only(query)

async def _ai_only(query: str) -> AskResponse:
    """Fallback using only AI without web search."""
    logger.info("Attempting AI-only fallback...")
    answer = await generate_answer(context="", query=query)
    if not answer.strip() or answer.strip().upper() == "NO_ANSWER":
        # Only if AI-only fails → fallback to AI + Web
        return await _ai_then_web_fallback(query)

    return AskResponse(
        answer=answer.strip(),
        sources=[],
        generated_by="AI-only",
        confidence=0.0
    )

# ------------------------------
# Helper functions
# ------------------------------
async def _online_search(query: str, top_k: int) -> AskResponse:
    web_results = await web_search(query, max_results=top_k)
    logger.info(f"_online_search results: {len(web_results)} items")

    if not web_results:
        answer = await generate_answer(context="", query=query)
        return AskResponse(
            answer=answer.strip() or "NO_ANSWER",
            sources=[],
            generated_by="AI-only",
            confidence=0.0,
        )

    context_texts = []
    raw_sources = []

    for r in web_results:
        text = clean_text(r.get("content") or r.get("snippet") or r.get("title") or "")
        if text:
            if len(text) > MAX_CHUNK_LENGTH:
                text = await summarize_text(text)
            context_texts.append(text)
        raw_sources.append({
            "document": r.get("title") or r.get("url") or "Unknown",
            "chunks_used": [],
            "relevance": 0.0
        })

    full_context = "\n\n".join(context_texts)
    answer = await generate_answer(context=full_context, query=query)
    if not answer.strip():
        answer = full_context if full_context else "NO_ANSWER"

    confidence = round(min(len(web_results)/TOP_K, 1.0), 2)  # better confidence scoring

    return AskResponse(
        answer=answer,
        sources=build_sources(raw_sources),
        generated_by="Online + AI",
        confidence=confidence
    )


async def _ai_then_web_fallback(query: str) -> AskResponse:
    """Fallback: Use all web content + AI to generate answer if Qdrant fails."""
    logger.info("Attempting AI + Web fallback...")

    web_results = await web_search(query, max_results=TOP_K)
    logger.info(f"Web search returned {len(web_results)} items")

    if not web_results:
        answer = await generate_answer(context="", query=query)
        return AskResponse(
            answer=answer.strip() or "NO_ANSWER",
            sources=[],
            generated_by="AI-only",
            confidence=0.0
        )

    context_texts = []
    raw_sources = []

    for r in web_results:
        text = (r.get("content") or r.get("snippet") or "").strip()
        if text:
            if len(text) > MAX_CHUNK_LENGTH:
                text = await summarize_text(text)
            context_texts.append(clean_text(text))
        raw_sources.append({
            "document": r.get("title") or r.get("url") or "Unknown",
            "chunks_used": [],
            "relevance": 0.0
        })

    if not context_texts:
        context_texts = [r.get("title") or r.get("url") or "" for r in web_results]

    full_context = "\n\n".join(context_texts)

    # Generate answer using GPT
    answer = await generate_answer(context=full_context, query=query)
    if not answer.strip() or answer.strip().upper() == "NO_ANSWER":
        answer = full_context if full_context else "NO_ANSWER"

    confidence = round(min(len(web_results)/TOP_K, 1.0), 2)  # better confidence scoring

    return AskResponse(
        answer=answer,
        sources=build_sources(raw_sources),
        generated_by="Web + AI fallback",
        confidence=confidence
    )
