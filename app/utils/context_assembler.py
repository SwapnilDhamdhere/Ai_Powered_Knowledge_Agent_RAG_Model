import logging
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger("ai-knowledge-agent")


class ContextAssembler:
    """
    Build a structured context from search results (Qdrant ScoredPoint or dict).
    - Groups by section_path (if present).
    - Sorts by chunk_index within sections.
    - Stitches nearby chunk indices into single section blocks.
    - Deduplicates near-duplicate blocks.
    - Returns a final assembled context string.
    """

    def __init__(self, max_chunks: int = 6, similarity_threshold: float = 0.80, neighbor_gap: int = 1):
        self.max_chunks = max_chunks
        self.similarity_threshold = similarity_threshold
        self.neighbor_gap = neighbor_gap

    # ---------------------------
    # Helpers to support ScoredPoint or dict
    # ---------------------------
    def _to_simple(self, hit: Any) -> Dict[str, Any]:
        """
        Normalize a hit (ScoredPoint or dict) to a simple dict:
        { "id": .., "payload": {...}, "score": float }
        """
        if isinstance(hit, dict):
            payload = hit.get("payload", {}) or {}
            _id = hit.get("id") or hit.get("point_id")
            score = hit.get("score", 0.0)
        else:
            # ScoredPoint / Pydantic style object
            payload = getattr(hit, "payload", {}) or {}
            _id = getattr(hit, "id", getattr(hit, "point_id", None))
            score = getattr(hit, "score", 0.0)

        return {"id": _id, "payload": payload, "score": float(score)}

    def is_similar(self, a: str, b: str) -> bool:
        """Check textual similarity to remove duplicates."""
        if not a or not b:
            return False
        return SequenceMatcher(None, a, b).ratio() > self.similarity_threshold

    def deduplicate(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove near-duplicate text blocks (keeps first occurrence)."""
        unique = []
        for block in blocks:
            text = block.get("text", "")
            if not any(self.is_similar(text, u["text"]) for u in unique):
                unique.append(block)
        return unique

    # ---------------------------
    # Grouping / stitching logic
    # ---------------------------
    def group_by_section(self, search_results: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns mapping: section_path -> list of items:
        { "id", "chunk_index", "text", "score", "source" }
        """
        grouped = defaultdict(list)

        for hit in search_results:
            item = self._to_simple(hit)
            payload = item["payload"] or {}
            # try different payload keys for compatibility
            section = payload.get("section_path") or payload.get("section") or payload.get("sectionPath") or ""
            source = payload.get("source") or payload.get("doc_title") or payload.get("document") or ""
            content = payload.get("content") or payload.get("text") or payload.get("body") or ""
            chunk_idx = payload.get("chunk_index")
            # normalize chunk index if possible
            try:
                chunk_idx = int(chunk_idx) if chunk_idx is not None else None
            except Exception:
                chunk_idx = None

            if not content:
                continue

            key = section.strip() or f"__no_section__::{source or 'unknown'}"
            grouped[key].append(
                {
                    "id": item["id"],
                    "chunk_index": chunk_idx if chunk_idx is not None else float("inf"),
                    "text": content.strip(),
                    "score": item["score"],
                    "source": source,
                }
            )

        return grouped

    def stitch_neighbors(self, grouped: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        For each section, sort by chunk_index and merge adjacent/nearby chunk texts into stitched blocks.
        Returns list of blocks with keys:
            - section
            - text (merged)
            - source
            - chunk_indices (list)
            - avg_score
        """
        stitched = []

        for section, items in grouped.items():
            # sort reliably (chunk_index might be inf if unknown)
            items_sorted = sorted(items, key=lambda x: (x["chunk_index"] if x["chunk_index"] is not None else float("inf")))
            # merge adjacent sequences
            current_group = []
            last_idx = None

            def flush_current():
                if not current_group:
                    return
                texts = [it["text"] for it in current_group]
                merged_text = "\n".join(texts)
                scores = [it["score"] for it in current_group]
                sources = [it["source"] for it in current_group if it.get("source")]
                stitched.append({
                    "section": section,
                    "text": merged_text,
                    "source": sources[0] if sources else "",
                    "chunk_indices": [it["chunk_index"] for it in current_group],
                    "avg_score": float(sum(scores) / len(scores)) if scores else 0.0
                })

            for it in items_sorted:
                idx = it["chunk_index"]
                if last_idx is None:
                    current_group = [it]
                else:
                    # if indices are numeric and gap is small, join; else new group
                    if isinstance(idx, (int, float)) and isinstance(last_idx, (int, float)) and (idx - last_idx) <= self.neighbor_gap:
                        current_group.append(it)
                    else:
                        flush_current()
                        current_group = [it]
                last_idx = idx
            flush_current()

        return stitched

    # ---------------------------
    # Public assemble API
    # ---------------------------
    def assemble(self, search_results: List[Any]) -> str:
        """
        Main entrypoint.
        Returns a structured string to be used as LLM context.
        """
        if not search_results:
            logger.warning("No search results to assemble context.")
            return ""

        # 1) normalize -> group
        grouped = self.group_by_section(search_results)

        if not grouped:
            logger.warning("Grouping produced no sections.")
            return ""

        # 2) stitch neighbors
        stitched = self.stitch_neighbors(grouped)

        if not stitched:
            logger.warning("No stitched blocks produced.")
            return ""

        # 3) deduplicate by text
        deduped = self.deduplicate(stitched)

        # 4) sort by avg_score (desc) and limit to max_chunks
        deduped_sorted = sorted(deduped, key=lambda b: b.get("avg_score", 0.0), reverse=True)
        selected = deduped_sorted[: self.max_chunks]

        # 5) build final context text with simple metadata headers
        blocks = []
        for b in selected:
            header = f"### Section: {b['section']}\nSource: {b.get('source','Unknown')}\nChunkIndices: {b.get('chunk_indices')}\nAvgScore: {b.get('avg_score'):.3f}\n"
            blocks.append(header + b["text"])

        context = "\n\n---\n\n".join(blocks)
        logger.info(f"Assembled context with {len(selected)} blocks (from {len(stitched)} stitched blocks).")
        return context
