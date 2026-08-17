from typing import List, Dict
from .loader import DocumentLoader
from .chunker import TextChunker, Chunk


class IngestionPipeline:
    def __init__(self, chunker: TextChunker = None):
        self.chunker = chunker or TextChunker()

    async def run(self, source: str) -> List[Chunk]:
        docs = DocumentLoader.load_from_file(source)
        chunks = self.chunker.chunk(docs)
        # Add content and source to metadata for each chunk
        for chunk in chunks:
            chunk.metadata["content"] = chunk.text
            chunk.metadata["source"] = source
        return chunks