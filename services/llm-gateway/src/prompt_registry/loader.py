import yaml
from pathlib import Path
from typing import Dict, Optional
import asyncio


class PromptLoader:
    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir

    async def load_template(self, name: str, version: str = "latest") -> str:
        # For simplicity: look for {name}.yaml, use 'default' version.
        file_path = self.templates_dir / f"{name}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt template {name} not found")
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        # If version specified, try to get that version.
        versions = data.get("versions", {})
        if version == "latest":
            # get highest version?
            if versions:
                latest = max(versions.keys())
                return versions[latest]["template"]
            else:
                return data.get("template", "")
        else:
            if version not in versions:
                raise ValueError(f"Version {version} not found for template {name}")
            return versions[version]["template"]

    async def list_templates(self) -> list[str]:
        return [p.stem for p in self.templates_dir.glob("*.yaml")]