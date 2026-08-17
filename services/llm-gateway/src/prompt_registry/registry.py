from typing import Optional, Dict, Any
import yaml
from pathlib import Path
from .loader import PromptLoader


class PromptRegistry:
    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir
        self.loader = PromptLoader(templates_dir)

    async def get_prompt(self, name: str, version: str = "latest", **variables) -> str:
        """Retrieve and render a prompt template."""
        template = await self.loader.load_template(name, version)
        if variables:
            return template.format(**variables)
        return template

    async def list_templates(self) -> list[str]:
        return await self.loader.list_templates()