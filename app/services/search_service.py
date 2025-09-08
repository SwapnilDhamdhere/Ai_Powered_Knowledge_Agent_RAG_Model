from app.services.embeddings_service import generate_embedding
from app.services.qdrant_service import semantic_search
from app.services.ollama_service import generate_answer
from app.core.logger import logger
#og version
# async def search_knowledge_base(query: str, top_k: int = 5):
#     """
#     Perform semantic search + contextual answer generation.
#     """
#     # Step 1: Generate query embedding
#     logger.info("Generating embedding for query...")
#     query_vector = await generate_embedding(query)
#
#     # Step 2: Semantic search in Qdrant
#     logger.info("Searching Qdrant for relevant chunks...")
#     search_results = await semantic_search(query_vector, top_k=top_k)
#
#     if not search_results:
#         logger.warning("No relevant documents found.")
#         return {"answer": "No relevant information found.", "sources": []}
#
#     # Step 3: Combine top-k chunks into a context string
#     top_chunks = [hit.payload.get("content") for hit in search_results]
#     sources = [hit.payload.get("source") for hit in search_results]
#     context = "\n\n".join(top_chunks)
#     # logger.info("context ..." + context)
#
#     # Step 4: Generate contextual answer
#     logger.info("Generating contextual answer using Ollama...")
#     logger.info("Context: " + context + " Question: " + query)
#
    # answer = await generate_answer(context=context, query=query)
#
#     return {
#         "answer": answer,
#         "sources": list(filter(None, sources))  # Remove None values
#     }

#1st change
# from collections import defaultdict
# from app.services.embeddings_service import generate_embedding
# from app.services.qdrant_service import semantic_search
# from app.services.ollama_service import generate_answer
# from app.core.logger import logger
#
# async def search_knowledge_base(query: str, top_k: int = 8, min_chunks: int = 3):
#     """
#     Perform semantic search in Qdrant and generate an AI-enhanced answer.
#     Returns:
#         {
#             "answer": str,
#             "sources": [
#                 {"document": str, "chunks_used": [int, int]},
#                 ...
#             ],
#             "generated_by": "Hybrid (Docs + AI)" | "AI-only"
#         }
#     """
#     logger.info("Generating embedding for query...")
#     query_vector = await generate_embedding(query)
#
#     logger.info(f"Searching Qdrant (top_k={top_k})...")
#     search_results = await semantic_search(query_vector, top_k=top_k)
#
#     # ✅ Fallback if fewer chunks found than expected
#     if not search_results or len(search_results) < min_chunks:
#         logger.warning(f"Found only {len(search_results)} chunks, retrying with top_k={min_chunks}...")
#         search_results = await semantic_search(query_vector, top_k=min_chunks)
#
#     # ✅ No documents found → fallback to AI-only response
#     if not search_results:
#         logger.warning("No relevant information found in Qdrant.")
#         return {
#             "answer": "No relevant information found in the knowledge base.",
#             "sources": [],
#             "generated_by": "AI-only"
#         }
#
#     # ✅ Prepare context and collect references
#     top_chunks = []
#     source_map = defaultdict(list)
#
#     for hit in search_results:
#         content = hit.payload.get("content")
#         source = hit.payload.get("source")
#         chunk_index = hit.payload.get("chunk_index")
#
#         if content:
#             top_chunks.append(content)
#         if source:
#             source_map[source].append(chunk_index)
#
#     context = "\n\n".join(top_chunks)
#
#     # ✅ Generate contextual answer using LLM
#     logger.info("Generating contextual answer using GPT-OSS...")
#     answer = await generate_answer(context=context, query=query)
#
#     # ✅ Build structured sources response
#     sources = [
#         {"document": doc, "chunks_used": sorted(set(idxs))}
#         for doc, idxs in source_map.items()
#     ]
#
#     return {
#         "answer": answer,
#         "sources": sources,
#         "generated_by": "Hybrid (Docs + AI)" if context.strip() else "AI-only"
#     }

#2ed step
from collections import defaultdict
from app.services.embeddings_service import generate_embedding
from app.services.qdrant_service import semantic_search
from app.services.ollama_service import generate_answer
from app.core.logger import logger

# Tune this threshold based on your embeddings
MIN_RELEVANCE = 0.6  # only consider chunks above this similarity
TOP_K = 8
MIN_CHUNKS = 3  # minimum chunks to consider for hybrid generation


async def search_knowledge_base(query: str, top_k: int = TOP_K, min_chunks: int = MIN_CHUNKS):
    """
    Perform semantic search in Qdrant and generate an AI-enhanced answer.

    Returns:
        {
            "answer": str,
            "sources": [
                {"document": str, "chunks_used": [int, int], "relevance": float},
                ...
            ],
            "generated_by": "Hybrid (Docs + AI)" | "AI-only",
            "confidence": float  # max similarity score of included chunks
        }
    """
    logger.info("Generating embedding for query...")
    query_vector = await generate_embedding(query)

    logger.info(f"Searching Qdrant (top_k={top_k})...")
    search_results = await semantic_search(query_vector, top_k=top_k)

    # Fallback if fewer chunks found than expected
    if not search_results or len(search_results) < min_chunks:
        logger.warning(f"Found only {len(search_results)} chunks, retrying with top_k={min_chunks}...")
        search_results = await semantic_search(query_vector, top_k=min_chunks)

    # No chunks found → AI-only
    if not search_results:
        logger.warning("No relevant chunks found in Qdrant.")
        answer = await generate_answer(context="", query=query)
        return {
            "answer": answer,
            "sources": [],
            "generated_by": "AI-only",
            "confidence": 0.0
        }

    # Prepare context and collect references with relevance filtering
    top_chunks = []
    source_map = defaultdict(list)
    relevance_map = defaultdict(list)

    for hit in search_results:
        score = getattr(hit, "score", 0)  # similarity from Qdrant
        if score < MIN_RELEVANCE:
            continue  # skip low-relevance chunks

        content = hit.payload.get("content")
        source = hit.payload.get("source")
        chunk_index = hit.payload.get("chunk_index")

        if content:
            top_chunks.append(content)
        if source:
            source_map[source].append(chunk_index)
            relevance_map[source].append(score)

    # If no high-relevance chunks → AI-only fallback
    if not top_chunks:
        logger.info("No high-relevance chunks found → AI-only response")
        answer = await generate_answer(context="", query=query)
        return {
            "answer": answer,
            "sources": [],
            "generated_by": "AI-only",
            "confidence": 0.0
        }

    # Combine top chunks into context
    context = "\n\n".join(top_chunks)

    # Generate answer using LLM
    logger.info("Generating contextual answer using GPT-OSS...")
    answer = await generate_answer(context=context, query=query)

    # Build structured sources with max relevance per document
    sources = [
        {
            "document": doc,
            "chunks_used": sorted(set(idxs)),
            "relevance": round(max(relevance_map[doc]), 2)  # max similarity as confidence per doc
        }
        for doc, idxs in source_map.items()
    ]

    # Overall confidence is max across all sources
    overall_confidence = round(max([s["relevance"] for s in sources], default=0.0), 2)

    return {
        "answer": answer,
        "sources": sources,
        "generated_by": "Hybrid (Docs + AI)",
        "confidence": overall_confidence
    }
