from typing import List
from libs.embeddings_common import EmbeddingClient


class EmbeddingService:
    def __init__(self, client: EmbeddingClient):
        self.client = client

    async def embed_chunks(self, chunks: List) -> List[List[float]]:
        texts = [chunk.text for chunk in chunks]
        response = await self.client.embed(texts)
        return response.embeddings