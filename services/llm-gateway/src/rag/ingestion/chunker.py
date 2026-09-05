import structlog
from typing import List, Dict, Any
from pathlib import Path

from ..pipeline import DocumentChunk

logger = structlog.get_logger()


class TextChunker:
    """Text chunker with overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk text into overlapping segments."""
        if not text or len(text) <= self.chunk_size:
            return [DocumentChunk(text=text, metadata=metadata)]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end
                for sep in ['. ', '! ', '? ', '\n\n', '\n']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep != -1:
                        end = last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index
                chunk_metadata["chunk_start"] = start
                chunk_metadata["chunk_end"] = end
                chunks.append(DocumentChunk(text=chunk_text, metadata=chunk_metadata))
                chunk_index += 1

            start = end - self.overlap
            if start >= len(text):
                break

        return chunks