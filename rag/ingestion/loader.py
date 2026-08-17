from typing import List, Any
from pathlib import Path


class DocumentLoader:
    """Load documents from various sources (files, URLs, etc.)."""

    @staticmethod
    def load_from_file(file_path: str) -> List[dict]:
        """Load text content from a file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return [{"content": content, "metadata": {"source": file_path}}]