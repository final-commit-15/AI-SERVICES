import structlog
from typing import List, Dict, Optional, Any
from pathlib import Path

from ..pipeline import DocumentChunk
from .chunker import TextChunker
from .loader import DocumentLoader

logger = structlog.get_logger()


class IngestionPipeline:
    """Document ingestion pipeline."""

    def __init__(self, chunker: TextChunker, loader: DocumentLoader = None):
        self.chunker = chunker
        self.loader = loader or DocumentLoader()

    async def run(
        self,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Run ingestion pipeline on a source."""
        # Load document
        documents = await self.loader.load(source)
        if not documents:
            return []

        all_chunks = []
        base_metadata = metadata or {}

        for doc in documents:
            # Merge metadata
            merged_metadata = {**base_metadata, **doc.get("metadata", {})}
            merged_metadata["source"] = source

            # Chunk the document
            chunks = self.chunker.chunk(doc["content"], merged_metadata)
            all_chunks.extend(chunks)

        return all_chunks