from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    metadata: Dict
    index: int


class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: List[Dict]) -> List[Chunk]:
        chunks = []
        for doc in documents:
            text = doc["content"]
            metadata = doc.get("metadata", {})
            # Simple sliding window
            start = 0
            idx = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "chunk_index": idx},
                    index=idx
                ))
                idx += 1
                start += self.chunk_size - self.overlap
        return chunks