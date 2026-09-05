from pydantic import BaseModel, Field
from typing import List, Optional


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List["EmbeddingData"]
    model: str
    usage: "UsageInfo"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class BatchEmbeddingRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = None
    batch_size: int = 100


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int
    usage: "UsageInfo"


from .chat import UsageInfo