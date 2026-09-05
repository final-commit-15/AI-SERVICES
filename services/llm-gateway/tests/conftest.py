import pytest
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from services.llm_gateway.src.main import app
from services.llm_gateway.src.config.settings import Settings


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings."""
    return Settings(
        environment="test",
        debug=True,
        ollama_host="http://localhost:11434",
        ollama_model="test-model",
        openai_api_key="test-key",
        anthropic_api_key="test-key",
        gemini_api_key="test-key",
        groq_api_key="test-key",
        together_api_key="test-key",
        openrouter_api_key="test-key",
        jwt_secret_key="test-secret-key-for-testing-only-32-chars",
        auth_enabled=False,
        rate_limit_enabled=False,
        cache_enabled=False,
    )


@pytest.fixture
def mock_router():
    """Create mock router."""
    router = MagicMock()
    router.route = MagicMock()
    router.record_request = AsyncMock()
    router.get_provider_stats = MagicMock(return_value={})
    router.get_all_models = MagicMock(return_value=[])
    return router


@pytest.fixture
def mock_provider():
    """Create mock provider."""
    provider = AsyncMock()
    provider.provider_name = "test"
    provider.capabilities = MagicMock(
        supports_streaming=True,
        supports_tools=True,
        supports_vision=False,
        supports_embeddings=True,
        supports_audio=False,
        supports_image_gen=False,
        supports_responses_api=False,
    )
    provider.models = []
    provider.chat = AsyncMock()
    provider.chat_stream = AsyncMock()
    provider.generate_embeddings = AsyncMock()
    provider.health_check = AsyncMock(return_value=True)
    provider.list_models = AsyncMock(return_value=[])
    provider.calculate_cost = MagicMock(return_value=0.0)
    return provider


@pytest.fixture
def sample_chat_request():
    """Sample chat request."""
    return {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "test-model",
        "temperature": 0.7,
        "stream": False,
    }


@pytest.fixture
def sample_chat_response():
    """Sample chat response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "provider": "test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }