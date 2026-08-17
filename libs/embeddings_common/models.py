from pydantic import BaseModel
from typing import List, Optional


class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = None
    truncate: bool = True


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int