import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.llm_gateway.src.providers.ollama import OllamaProvider
from services.llm_gateway.src.providers.openai import OpenAIProvider
from services.llm_gateway.src.providers.anthropic import AnthropicProvider
from services.llm_gateway.src.providers.gemini import GeminiProvider
from services.llm_gateway.src.providers.groq import GroqProvider
from services.llm_gateway.src.providers.together import TogetherProvider
from services.llm_gateway.src.providers.openrouter import OpenRouterProvider

from libs.schemas_common.chat import ChatRequest, ChatMessage, MessageRole
from libs.schemas_common.embeddings import EmbeddingRequest
from libs.schemas_common.providers import ProviderName, ModelInfo


class TestOllamaProvider:
    @pytest.fixture
    def provider(self):
        return OllamaProvider(host="http://localhost:11434", default_model="test-model")

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        with patch("ollama.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.list = AsyncMock(return_value={"models": []})
            mock_client.return_value = mock_instance
            
            await provider.initialize()
            
            assert provider._client is not None
            mock_instance.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self, provider):
        with patch.object(provider, "_client") as mock_client:
            mock_client.list = AsyncMock(return_value={"models": []})
            provider._client = mock_client
            
            result = await provider.health_check()
            
            assert result is True
            mock_client.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_models(self, provider):
        with patch.object(provider, "_client") as mock_client:
            mock_client.list = AsyncMock(return_value={
                "models": [{"name": "test-model"}]
            })
            provider._client = mock_client
            
            models = await provider.list_models()
            
            assert len(models) == 1
            assert models[0].id == "test-model"

    @pytest.mark.asyncio
    async def test_chat(self, provider):
        with patch.object(provider, "_client") as mock_client:
            mock_client.chat = AsyncMock(return_value={
                "message": {"role": "assistant", "content": "Hello!"},
                "prompt_eval_count": 10,
                "eval_count": 5,
            })
            provider._client = mock_client
            
            request = ChatRequest(
                messages=[ChatMessage(role=MessageRole.USER, content="Hi")],
                model="test-model",
            )
            response = await provider.chat(request)
            
            assert response.choices[0].message.content == "Hello!"
            assert response.provider == "ollama"

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, provider):
        with patch.object(provider, "_client") as mock_client:
            mock_client.embeddings = AsyncMock(return_value={"embedding": [0.1] * 768})
            provider._client = mock_client
            
            request = EmbeddingRequest(input=["test text"])
            response = await provider.generate_embeddings(request)
            
            assert len(response.data) == 1
            assert len(response.data[0].embedding) == 768


class TestOpenAIProvider:
    @pytest.fixture
    def provider(self):
        return OpenAIProvider(api_key="test-key", default_model="gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.models.list = AsyncMock(return_value=MagicMock(data=[]))
            mock_client_class.return_value = mock_client
            
            await provider.initialize()
            
            assert provider._client is not None

    @pytest.mark.asyncio
    async def test_health_check(self, provider):
        with patch.object(provider, "_client") as mock_client:
            mock_client.models.list = AsyncMock(return_value=MagicMock(data=[]))
            provider._client = mock_client
            
            result = await provider.health_check()
            
            assert result is True

    @pytest.mark.asyncio
    async def test_chat(self, provider):
        with patch.object(provider, "_client") as mock_client:
            mock_response = MagicMock()
            mock_response.id = "chatcmpl-test"
            mock_response.created = 1234567890
            mock_response.model = "gpt-4o-mini"
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].index = 0
            mock_response.choices[0].message.role = "assistant"
            mock_response.choices[0].message.content = "Hello!"
            mock_response.choices[0].message.tool_calls = None
            mock_response.choices[0].finish_reason = "stop"
            mock_response.choices[0].logprobs = None
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15
            mock_response.usage.prompt_tokens_details = None
            mock_response.usage.completion_tokens_details = None
            mock_response.system_fingerprint = None
            
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            provider._client = mock_client
            
            request = ChatRequest(
                messages=[ChatMessage(role=MessageRole.USER, content="Hi")],
                model="gpt-4o-mini",
            )
            response = await provider.chat(request)
            
            assert response.choices[0].message.content == "Hello!"
            assert response.provider == "openai"


class TestAnthropicProvider:
    @pytest.fixture
    def provider(self):
        return AnthropicProvider(api_key="test-key", default_model="claude-3-5-sonnet-20241022")

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        await provider.initialize()
        assert provider._client is not None

    @pytest.mark.asyncio
    async def test_list_models(self, provider):
        await provider.initialize()
        models = await provider.list_models()
        
        assert len(models) >= 3
        assert any(m.id == "claude-3-5-sonnet-20241022" for m in models)

    @pytest.mark.asyncio
    async def test_generate_embeddings_raises(self, provider):
        await provider.initialize()
        request = EmbeddingRequest(input=["test"])
        
        with pytest.raises(NotImplementedError):
            await provider.generate_embeddings(request)


class TestGeminiProvider:
    @pytest.fixture
    def provider(self):
        return GeminiProvider(api_key="test-key", default_model="gemini-1.5-flash")

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        with patch("google.generativeai.configure") as mock_configure:
            with patch("google.generativeai.GenerativeModel") as mock_model:
                await provider.initialize()
                mock_configure.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_list_models(self, provider):
        await provider.initialize()
        models = await provider.list_models()
        
        assert len(models) >= 4
        assert any(m.id == "gemini-1.5-pro" for m in models)
        assert any(m.id == "text-embedding-004" for m in models)


class TestGroqProvider:
    @pytest.fixture
    def provider(self):
        return GroqProvider(api_key="test-key", default_model="llama-3.1-70b-versatile")

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        with patch("groq.AsyncGroq") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            await provider.initialize()
            
            assert provider._client is not None

    @pytest.mark.asyncio
    async def test_list_models(self, provider):
        await provider.initialize()
        models = await provider.list_models()
        
        assert len(models) >= 4
        assert any(m.id == "llama-3.1-70b-versatile" for m in models)


class TestTogetherProvider:
    @pytest.fixture
    def provider(self):
        return TogetherProvider(api_key="test-key", default_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        with patch("together.AsyncTogether") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            await provider.initialize()
            
            assert provider._client is not None

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, provider):
        with patch.object(provider, "_client") as mock_client:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(index=0, embedding=[0.1] * 768)]
            mock_response.model = "test-model"
            mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)
            
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            provider._client = mock_client
            
            request = EmbeddingRequest(input=["test"])
            response = await provider.generate_embeddings(request)
            
            assert len(response.data) == 1


class TestOpenRouterProvider:
    @pytest.fixture
    def provider(self):
        return OpenRouterProvider(api_key="test-key", default_model="anthropic/claude-3.5-sonnet")

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            with patch("httpx.AsyncClient") as mock_http:
                mock_resp = AsyncMock()
                mock_resp.json = AsyncMock(return_value={"data": []})
                mock_http.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_resp)
                
                await provider.initialize()
                
                assert provider._client is not None

    @pytest.mark.asyncio
    async def test_list_models(self, provider):
        with patch("httpx.AsyncClient") as mock_http:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value={"data": [
                {"id": "test-model", "name": "Test", "context_length": 4096, 
                 "pricing": {"prompt": "0.001", "completion": "0.002"},
                 "supported_parameters": ["stream"], "architecture": {"input_modalities": []}, "tags": []}
            ]})
            mock_http.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_resp)
            
            await provider.initialize()
            models = await provider.list_models()
            
            assert len(models) >= 1