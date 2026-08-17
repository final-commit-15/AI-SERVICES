from abc import ABC, abstractmethod
from typing import List, Dict, Tuple


class VectorStore(ABC):
    @abstractmethod
    async def add(self, vectors: List[List[float]], metadata: List[Dict], ids: List[str]):
        pass

    @abstractmethod
    async def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[Dict, float]]:
        pass