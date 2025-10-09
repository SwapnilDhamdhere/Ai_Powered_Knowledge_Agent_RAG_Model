from sentence_transformers import SentenceTransformer
import re
import logging
from typing import List
import nltk
from nltk.corpus import stopwords

logger = logging.getLogger("ai-knowledge-agent")

# Download stopwords if not available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class QueryOptimizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", top_k: int = 5):
        """
        Initialize the Query Optimizer.
        :param model_name: Sentence Transformer model for semantic understanding.
        :param top_k: Number of key concepts to extract.
        """
        logger.info(f"Loading query optimization model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.stop_words = set(stopwords.words("english"))

    def clean(self, query: str) -> str:
        """Basic normalization: lowercase, remove punctuation."""
        query = query.lower().strip()
        query = re.sub(r"[^a-z0-9\s]", "", query)
        return query

    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords by removing stopwords and ranking by embedding similarity.
        """
        tokens = [t for t in query.split() if t not in self.stop_words and len(t) > 2]
        if not tokens:
            return query.split()

        token_embeddings = self.model.encode(tokens)
        query_embedding = self.model.encode([query])[0]

        # Cosine similarity ranking
        similarities = [
            (tokens[i], float(token_embeddings[i] @ query_embedding) /
             (self._norm(token_embeddings[i]) * self._norm(query_embedding) + 1e-8))
            for i in range(len(tokens))
        ]

        # Sort by relevance and pick top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_keywords = [kw for kw, _ in similarities[:self.top_k]]

        return top_keywords

    def _norm(self, vec):
        return sum(v ** 2 for v in vec) ** 0.5

    def optimize(self, query: str) -> str:
        """
        Optimize query before embeddings:
        - Clean
        - Remove noise words
        - Focus query on key concepts
        """
        cleaned = self.clean(query)
        keywords = self.extract_keywords(cleaned)
        optimized_query = " ".join(keywords)

        logger.info(f"Optimized query: {optimized_query}")
        return optimized_query