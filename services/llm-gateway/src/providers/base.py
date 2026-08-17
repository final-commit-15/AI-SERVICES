from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def chat(self,
                   messages: list[dict[str, Any]],
                   *,
                   thinking: bool = False,
                   temperature: float = 0.0,
                   response_format: Optional[Dict[str, Any]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Send a chat completion request."""
        pass

    @abstractmethod
    async def generate(self,
                       prompt: str,
                       *,
                       temperature: float = 0.0,
                       max_tokens: Optional[int] = None,
                       **kwargs) -> str:
        """Simple text generation."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass