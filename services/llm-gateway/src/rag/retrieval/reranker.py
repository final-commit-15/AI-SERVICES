import structlog
from typing import List, Dict, Any, Optional

logger = structlog.get_logger()


class Reranker:
    """Cross-encoder reranker for improving retrieval quality."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model
        self._model = None

    async def _load_model(self):
        """Lazy load the reranker model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info("reranker_model_loaded", model=self.model_name)
            except ImportError:
                logger.warning("sentence_transformers_not_installed_reranker_disabled")
                self._model = False

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder."""
        await self._load_model()

        if not self._model:
            return documents

        if not documents:
            return []

        try:
            # Prepare pairs for reranking
            pairs = [(query, doc.get("content", "")) for doc in documents]

            # Get scores
            scores = self._model.predict(pairs)

            # Sort by score
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            if top_k:
                scored_docs = scored_docs[:top_k]

            # Return reranked documents with scores
            return [
                {**doc, "rerank_score": float(score)}
                for doc, score in scored_docs
            ]
        except Exception as e:
            logger.error("rerank_failed", error=str(e))
            return documents