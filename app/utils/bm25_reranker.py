# without library
# import math
# from collections import Counter
# from typing import List, Dict
#
#
# class BM25Reranker:
#     def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
#         """
#         Initialize BM25 reranker.
#
#         :param corpus: List of document texts.
#         :param k1: BM25 k1 parameter (controls term frequency scaling).
#         :param b: BM25 b parameter (controls document length normalization).
#         """
#         self.corpus = corpus
#         self.k1 = k1
#         self.b = b
#         self.doc_count = len(corpus)
#         self.avg_doc_len = sum(len(doc.split()) for doc in corpus) / self.doc_count
#         self.doc_freqs = []
#         self.term_doc_freq = {}
#         self._initialize()
#
#     def _initialize(self):
#         """Precompute term frequencies and document frequencies."""
#         for doc in self.corpus:
#             freqs = Counter(doc.lower().split())
#             self.doc_freqs.append(freqs)
#             for term in freqs.keys():
#                 self.term_doc_freq.setdefault(term, 0)
#                 self.term_doc_freq[term] += 1
#
#     def _idf(self, term: str) -> float:
#         """Compute inverse document frequency for a term."""
#         n_qi = self.term_doc_freq.get(term, 0)
#         if n_qi == 0:
#             return 0
#         return math.log((self.doc_count - n_qi + 0.5) / (n_qi + 0.5) + 1.0)
#
#     def score(self, query: str, index: int) -> float:
#         """Compute BM25 score for a document given a query."""
#         score = 0.0
#         doc_freqs = self.doc_freqs[index]
#         doc_len = sum(doc_freqs.values())
#         for term in query.lower().split():
#             if term not in doc_freqs:
#                 continue
#             tf = doc_freqs[term]
#             idf = self._idf(term)
#             numerator = tf * (self.k1 + 1)
#             denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
#             score += idf * (numerator / denominator)
#         return score
#
#     def rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
#         """
#         Re-rank documents using BM25 score.
#
#         :param query: Query string.
#         :param docs: List of dicts with {"id": ..., "payload": {"content": ...}}.
#         :return: Sorted list of documents with BM25 scores added.
#         """
#         for doc in docs:
#             doc_text = doc["payload"]["content"]
#             if doc_text not in self.corpus:
#                 self.corpus.append(doc_text)
#                 self._initialize()
#             doc_index = self.corpus.index(doc_text)
#             doc["bm25_score"] = self.score(query, doc_index)
#
#         return sorted(docs, key=lambda x: x["bm25_score"], reverse=True)

from rank_bm25 import BM25Okapi
import logging
from typing import List, Any

logger = logging.getLogger("ai-knowledge-agent")


class BM25Reranker:
    def __init__(self, corpus: List[str]):
        """
        Initialize BM25 with a given corpus.
        :param corpus: List of document contents (strings) in the same order as docs to be reranked.
        """
        logger.info("Initializing BM25 with corpus of size: %d", len(corpus))
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus = corpus

    def _set_score_on_doc(self, doc: Any, score: float) -> None:
        """
        Attach bm25 score into doc in a safe way:
        - If dict-like: place under payload['bm25_score']
        - If object with payload dict: payload['bm25_score']
        - Else attach attribute 'bm25_score'
        """
        try:
            if isinstance(doc, dict):
                payload = doc.setdefault("payload", {})
                payload["bm25_score"] = float(score)
            else:
                payload = getattr(doc, "payload", None)
                if isinstance(payload, dict):
                    payload["bm25_score"] = float(score)
                else:
                    setattr(doc, "bm25_score", float(score))
        except Exception:
            try:
                setattr(doc, "bm25_score", float(score))
            except Exception:
                logger.debug("Could not set bm25 score on doc of type %s", type(doc))

    def _get_score_from_doc(self, doc: Any) -> float:
        try:
            if isinstance(doc, dict):
                return float(doc.get("payload", {}).get("bm25_score", 0.0))
            payload = getattr(doc, "payload", None)
            if isinstance(payload, dict) and "bm25_score" in payload:
                return float(payload["bm25_score"])
            return float(getattr(doc, "bm25_score", 0.0))
        except Exception:
            return 0.0

    def rerank(self, query: str, docs: List[Any]) -> List[Any]:
        """
        Rerank documents using BM25 scores. `docs` must be in the SAME order as the `corpus`
        passed to constructor.
        """
        logger.info("Reranking %d documents using BM25", len(docs))
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        for i, doc in enumerate(docs):
            s = float(scores[i]) if i < len(scores) else 0.0
            self._set_score_on_doc(doc, s)

        # return docs sorted by new bm25 score descending
        sorted_docs = sorted(docs, key=lambda x: self._get_score_from_doc(x), reverse=True)
        return sorted_docs
