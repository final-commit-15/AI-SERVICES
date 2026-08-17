from typing import List, Dict


class Reranker:
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider

    async def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        # Simple pass-through; could use cross-encoder or LLM
        return documents