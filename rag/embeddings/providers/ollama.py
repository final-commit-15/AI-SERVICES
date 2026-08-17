from libs.embeddings_common.base import EmbeddingProvider
from libs.embeddings_common.models import EmbeddingRequest, EmbeddingResponse
from ollama import AsyncClient
from typing import List


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, host: str, model: str):
        self.client = AsyncClient(host=host)
        self.model = model

    @property
    def dimension(self) -> int:
        return 768   # for nomic-embed-text; adjust as needed

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model = request.model or self.model
        embeddings = []
        for text in request.texts:
            resp = await self.client.embeddings(model=model, prompt=text)
            embeddings.append(resp["embedding"])
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            dimension=self.dimension
        )

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = await self.embed(EmbeddingRequest(texts=texts))
        return resp.embeddings