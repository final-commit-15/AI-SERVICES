import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from . import VectorStore


class FAISSVectorStore(VectorStore):
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: List[Dict] = []
        self.ids: List[str] = []
        self.content_map: Dict[str, str] = {}   # id -> content

    async def add(self, vectors: List[List[float]], metadata: List[Dict], ids: List[str]):
        # Deduplicate: only add vectors whose ID is not already present
        new_vectors = []
        new_metadata = []
        new_ids = []
        for vec, meta, id_ in zip(vectors, metadata, ids):
            if id_ not in self.content_map:
                new_vectors.append(vec)
                new_metadata.append(meta)
                new_ids.append(id_)
                # Store content for later retrieval
                self.content_map[id_] = meta.get("content", "")
        if new_vectors:
            arr = np.array(new_vectors).astype(np.float32)
            self.index.add(arr)
            self.metadata.extend(new_metadata)
            self.ids.extend(new_ids)

    async def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[Dict, float]]:
        if self.index.ntotal == 0:
            return []
        # Limit top_k to available vectors
        k = min(top_k, self.index.ntotal)
        arr = np.array([query_vector]).astype(np.float32)
        distances, indices = self.index.search(arr, k)
        results = []
        seen_ids = set()
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                chunk_id = self.ids[idx]
                # Skip duplicates (based on ID) – in case FAISS returns same vector twice (rare)
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                # Enrich metadata with content and score
                meta["content"] = self.content_map.get(chunk_id, "")
                meta["score"] = float(dist)
                results.append((meta, float(dist)))
        return results