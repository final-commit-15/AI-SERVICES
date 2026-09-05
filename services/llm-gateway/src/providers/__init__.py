from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .groq import GroqProvider
from .together import TogetherProvider
from .openrouter import OpenRouterProvider

from libs.schemas_common.providers import ProviderName

PROVIDER_MAP = {
    ProviderName.OLLAMA: OllamaProvider,
    ProviderName.OPENAI: OpenAIProvider,
    ProviderName.ANTHROPIC: AnthropicProvider,
    ProviderName.GEMINI: GeminiProvider,
    ProviderName.GROQ: GroqProvider,
    ProviderName.TOGETHER: TogetherProvider,
    ProviderName.OPENROUTER: OpenRouterProvider,
}


def get_provider_class(provider_name: ProviderName):
    """Get provider class by name."""
    return PROVIDER_MAP.get(provider_name)


def create_provider(provider_name: ProviderName, **kwargs):
    """Create provider instance."""
    provider_class = get_provider_class(provider_name)
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}")
    return provider_class(**kwargs)


__all__ = [
    "OllamaProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "GroqProvider",
    "TogetherProvider",
    "OpenRouterProvider",
    "PROVIDER_MAP",
    "get_provider_class",
    "create_provider",
]