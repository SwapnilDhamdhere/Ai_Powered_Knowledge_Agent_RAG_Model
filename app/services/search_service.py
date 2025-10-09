from collections import defaultdict
from typing import List, Dict

from app.services.intent_service import IntentService
from app.services.ollama_service import generate_answer
from app.services.web_service import web_search, clean_text, summarize_text
from app.core.logger import logger
from app.core.config import settings
from app.models.query_model import AskResponse, SourceInfo
from app.services.search_pipeline import search_documents
from app.utils.bm25_reranker import BM25Reranker
from app.utils.query_optimizer import QueryOptimizer
from app.utils.context_assembler import ContextAssembler

# 🔧 Tuning parameters
MIN_RELEVANCE = settings.MIN_RELEVANCE if hasattr(settings, "MIN_RELEVANCE") else 0.6   # Minimum similarity score to keep a chunk
TOP_K = settings.TOP_K if hasattr(settings, "TOP_K") else 8                             # Default top-k search
MIN_CHUNKS = settings.MIN_CHUNKS if hasattr(settings, "MIN_CHUNKS") else 3              # Minimum chunks required to consider knowledge-based answer
MAX_CHUNK_LENGTH = 1000                                                                 # Max chars per chunk for summarization

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
    """
    Searches the knowledge base using semantic/hybrid retrieval with query optimization,
    BM25 reranking, and structured context assembly.
    """
    mode = (mode or settings.SEARCH_MODE).lower()
    logger.info(f"Search mode: {mode}")

    query_optimizer = QueryOptimizer()
    assembler = ContextAssembler(max_chunks=6, similarity_threshold=0.85, neighbor_gap=1)

    # 🌐 ONLINE SEARCH
    if mode == "online":
        return await _online_search(query, top_k)

    # 🧠 SEMANTIC / HYBRID SEARCH
    optimized_query = query_optimizer.optimize(query)
    logger.info(f"Searching Qdrant with optimized query: {optimized_query} (top_k={top_k})...")
    search_results = await search_documents(optimized_query, top_k=top_k)

    if not search_results or len(search_results) < min_chunks:
        logger.warning(f"Found only {len(search_results)} chunks, retrying with top_k={min_chunks}...")
        search_results = await search_documents(optimized_query, top_k=min_chunks)

    if not search_results:
        logger.warning("No Qdrant results found — falling back to AI-only mode.")
        return await _ai_only(query)

    # 🧩 BM25 RERANKING
    try:
        logger.info("Applying BM25 reranking...")
        corpus = [hit.payload.get("content", "") for hit in search_results if hit.payload.get("content")]
        if corpus:
            reranker = BM25Reranker(corpus)
            search_results = reranker.rerank(optimized_query, search_results)
        else:
            logger.warning("Empty corpus for BM25 — skipping rerank.")
    except Exception as e:
        logger.exception(f"BM25 reranking failed: {e}")

    # 🧱 CONTEXT ASSEMBLY
    top_chunks, source_map, relevance_map = [], defaultdict(list), defaultdict(list)

    for hit in search_results:
        score = getattr(hit, "score", 0)
        if score < MIN_RELEVANCE:
            continue

        payload = hit.payload
        content = payload.get("content")
        source = payload.get("source")
        section_path = payload.get("section_path") or "N/A"
        chunk_index = payload.get("chunk_index")

        if content:
            top_chunks.append(content)

        if source:
            source_map[source].append({
                "chunk_index": chunk_index,
                "section_path": section_path,
                "score": score
            })
            relevance_map[source].append(score)

    try:
        logger.info("Assembling structured context using ContextAssembler...")
        context = assembler.assemble(search_results)
    except Exception as e:
        logger.exception(f"Context assembly failed: {e}")
        context = "\n\n".join(top_chunks)

    # 💬 GENERATE ANSWER
    intent = await IntentService.classify_intent(query)
    answer = await generate_answer(context=context, query=query, intent=intent)
    if not answer.strip() or answer.strip().upper() == "NO_ANSWER":
        logger.warning("AI returned no useful answer — falling back to AI-only mode.")
        return await _ai_only(query)

    # 📚 BUILD SOURCES
    raw_sources = [
        {
            "document": doc,
            "sections": sorted({item["section_path"] for item in chunks}),
            "chunks_used": sorted({item["chunk_index"] for item in chunks}),
            "relevance": round(max(item["score"] for item in chunks), 2)
        }
        for doc, chunks in source_map.items()
    ]

    sources = build_sources(raw_sources)
    overall_confidence = round(max([s.relevance for s in sources], default=0.0), 2)

    return AskResponse(
        answer=answer.strip(),
        sources=sources,
        generated_by="Hybrid (Docs + AI)",
        confidence=overall_confidence
    )

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