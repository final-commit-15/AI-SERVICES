import structlog
from typing import List, Dict, Any
from pathlib import Path

logger = structlog.get_logger()


class DocumentLoader:
    """Document loader for various file types."""

    async def load(self, source: str) -> List[Dict[str, Any]]:
        """Load document from file path or URL."""
        path = Path(source)

        if path.is_file():
            return await self._load_file(path)
        elif path.is_dir():
            return await self._load_directory(path)
        else:
            # Treat as text
            return [{"content": source, "metadata": {"source": "text"}}]

    async def _load_file(self, path: Path) -> List[Dict[str, Any]]:
        """Load a single file."""
        suffix = path.suffix.lower()

        if suffix == ".txt" or suffix == ".md":
            content = path.read_text(encoding='utf-8')
            return [{"content": content, "metadata": {"filename": path.name, "type": "text"}}]

        elif suffix == ".pdf":
            return await self._load_pdf(path)

        elif suffix in [".docx", ".doc"]:
            return await self._load_docx(path)

        elif suffix in [".html", ".htm"]:
            return await self._load_html(path)

        elif suffix == ".json":
            return await self._load_json(path)

        elif suffix == ".csv":
            return await self._load_csv(path)

        else:
            # Try as text
            try:
                content = path.read_text(encoding='utf-8')
                return [{"content": content, "metadata": {"filename": path.name, "type": "text"}}]
            except:
                logger.warning("unsupported_file_type", file=str(path))
                return []

    async def _load_pdf(self, path: Path) -> List[Dict[str, Any]]:
        """Load PDF file."""
        try:
            import pdfplumber
            docs = []
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        docs.append({
                            "content": text,
                            "metadata": {"filename": path.name, "page": i + 1, "type": "pdf"}
                        })
            return docs
        except ImportError:
            logger.warning("pdfplumber_not_installed")
            return []

    async def _load_docx(self, path: Path) -> List[Dict[str, Any]]:
        """Load DOCX file."""
        try:
            from docx import Document
            doc = Document(path)
            content = "\n".join([p.text for p in doc.paragraphs if p.text])
            return [{"content": content, "metadata": {"filename": path.name, "type": "docx"}}]
        except ImportError:
            logger.warning("python-docx_not_installed")
            return []

    async def _load_html(self, path: Path) -> List[Dict[str, Any]]:
        """Load HTML file."""
        try:
            from bs4 import BeautifulSoup
            content = path.read_text(encoding='utf-8')
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text()
            return [{"content": text, "metadata": {"filename": path.name, "type": "html"}}]
        except ImportError:
            logger.warning("beautifulsoup4_not_installed")
            return []

    async def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSON file."""
        import json
        content = path.read_text(encoding='utf-8')
        data = json.loads(content)
        return [{"content": json.dumps(data, indent=2), "metadata": {"filename": path.name, "type": "json"}}]

    async def _load_csv(self, path: Path) -> List[Dict[str, Any]]:
        """Load CSV file."""
        import csv
        docs = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                content = "\n".join([f"{k}: {v}" for k, v in row.items()])
                docs.append({"content": content, "metadata": {"filename": path.name, "row": i, "type": "csv"}})
        return docs

    async def _load_directory(self, path: Path) -> List[Dict[str, Any]]:
        """Load all supported files in directory."""
        docs = []
        for file_path in path.rglob("*"):
            if file_path.is_file():
                file_docs = await self._load_file(file_path)
                docs.extend(file_docs)
        return docs