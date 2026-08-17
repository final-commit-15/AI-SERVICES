from typing import Optional, Dict, Any
from ..providers.base import LLMProvider
from ..providers.local.ollama_provider import OllamaProvider
from ..providers.openai.openai_provider import OpenAIProvider
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class ModelRouter:
    def __init__(self):
        self.providers = {
            "ollama": OllamaProvider(),
            "openai": OpenAIProvider() if settings.openai_api_key else None,
            # "anthropic": AnthropicProvider() if settings.anthropic_api_key else None,
        }
        self.default_provider = settings.default_model_provider

    def get_provider(self, provider_name: Optional[str] = None) -> LLMProvider:
        provider = provider_name or self.default_provider
        provider_obj = self.providers.get(provider)
        if not provider_obj:
            logger.warning(f"Provider {provider} not found, falling back to {self.default_provider}")
            provider_obj = self.providers[self.default_provider]
        return provider_obj

    def route(self,
              task_type: str = "general",
              complexity: str = "medium",
              latency: str = "normal",
              provider: Optional[str] = None,
              **kwargs) -> LLMProvider:
        """Select a provider based on task parameters."""
        # Simple logic: if provider specified, use it; else choose based on task_type.
        if provider:
            return self.get_provider(provider)

        # Example: use openai for reasoning/complex, ollama for simple
        if task_type == "reasoning" and settings.openai_api_key:
            return self.get_provider("openai")
        elif task_type == "coding" and settings.openai_api_key:
            return self.get_provider("openai")  # or specific coding model
        else:
            return self.get_provider("ollama")