import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from services.llm_gateway.src.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        with patch("services.llm_gateway.src.dependencies.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.providers = {}
            mock_get_router.return_value = mock_router
            
            response = await client.get("/v1/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "components" in data

    @pytest.mark.asyncio
    async def test_readiness(self, client):
        response = await client.get("/v1/health/ready")
        
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    @pytest.mark.asyncio
    async def test_liveness(self, client):
        response = await client.get("/v1/health/live")
        
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestModelsEndpoints:
    @pytest.mark.asyncio
    async def test_list_models(self, client):
        with patch("services.llm_gateway.src.dependencies.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.providers = {}
            mock_router.get_all_models = MagicMock(return_value=[])
            mock_router.get_provider_stats = MagicMock(return_value={})
            mock_get_router.return_value = mock_router
            
            response = await client.get("/v1/models")
            
            assert response.status_code == 200
            data = response.json()
            assert "providers" in data
            assert "models" in data


class TestChatEndpoints:
    @pytest.mark.asyncio
    async def test_chat_completion(self, client):
        with patch("services.llm_gateway.src.dependencies.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_provider = AsyncMock()
            mock_provider.provider_name = "test"
            mock_provider.capabilities = MagicMock(
                supports_streaming=False,
                supports_tools=False,
            )
            
            mock_response = MagicMock()
            mock_response.id = "chatcmpl-test"
            mock_response.created = 1234567890
            mock_response.model = "test-model"
            mock_response.provider = "test"
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].index = 0
            mock_response.choices[0].message = MagicMock()
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
            
            mock_provider.chat = AsyncMock(return_value=mock_response)
            mock_router.route = MagicMock(return_value=mock_provider)
            mock_router.record_request = AsyncMock()
            mock_router.providers = {}
            mock_get_router.return_value = mock_router
            
            with patch("services.llm_gateway.src.caching.cache.get_cache", return_value=None):
                with patch("services.llm_gateway.src.guardrails.guardrails.apply_input_guardrails", return_value=True):
                    with patch("services.llm_gateway.src.guardrails.guardrails.apply_output_guardrails", return_value=True):
                        response = await client.post(
                            "/v1/chat",
                            json={
                                "messages": [{"role": "user", "content": "Hello"}],
                                "model": "test-model",
                            }
                        )
            
            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_stream(self, client):
        with patch("services.llm_gateway.src.dependencies.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_provider = AsyncMock()
            mock_provider.provider_name = "test"
            mock_provider.capabilities = MagicMock(supports_streaming=True)
            
            async def mock_stream(request):
                yield MagicMock(
                    id="chatcmpl-test",
                    created=1234567890,
                    model="test-model",
                    provider="test",
                    choices=[MagicMock(
                        index=0,
                        delta=MagicMock(role="assistant", content="Hello"),
                        finish_reason=None,
                    )]
                )
                yield MagicMock(
                    id="chatcmpl-test",
                    created=1234567890,
                    model="test-model",
                    provider="test",
                    choices=[MagicMock(
                        index=0,
                        delta=MagicMock(role=None, content=" World"),
                        finish_reason="stop",
                    )]
                )
            
            mock_provider.chat_stream = mock_stream
            mock_router.route = MagicMock(return_value=mock_provider)
            mock_router.record_request = AsyncMock()
            mock_router.providers = {}
            mock_get_router.return_value = mock_router
            
            with patch("services.llm_gateway.src.caching.cache.get_cache", return_value=None):
                with patch("services.llm_gateway.src.guardrails.guardrails.apply_input_guardrails", return_value=True):
                    with patch("services.llm_gateway.src.guardrails.guardrails.apply_output_guardrails", return_value=True):
                        response = await client.post(
                            "/v1/chat",
                            json={
                                "messages": [{"role": "user", "content": "Hello"}],
                                "model": "test-model",
                                "stream": True,
                            }
                        )
            
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]


class TestEmbeddingsEndpoints:
    @pytest.mark.asyncio
    async def test_create_embeddings(self, client):
        with patch("services.llm_gateway.src.dependencies.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_provider = AsyncMock()
            mock_provider.provider_name = "test"
            
            mock_response = MagicMock()
            mock_response.data = [MagicMock(index=0, embedding=[0.1] * 768)]
            mock_response.model = "test-model"
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=0, total_tokens=10)
            
            mock_provider.generate_embeddings = AsyncMock(return_value=mock_response)
            mock_router.route = MagicMock(return_value=mock_provider)
            mock_router.record_request = AsyncMock()
            mock_router.providers = {}
            mock_get_router.return_value = mock_router
            
            with patch("services.llm_gateway.src.caching.cache.get_cache", return_value=None):
                response = await client.post(
                    "/v1/embeddings",
                    json={
                        "input": "test text",
                        "model": "test-model",
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["data"]) == 1


class TestRAGEndpoints:
    @pytest.mark.asyncio
    async def test_rag_query(self, client):
        with patch("services.llm_gateway.src.dependencies.get_rag_pipeline") as mock_get_rag:
            mock_rag = AsyncMock()
            mock_rag.query = AsyncMock(return_value=[
                {"content": "Test document", "metadata": {"source": "test.txt"}}
            ])
            mock_get_rag.return_value = mock_rag
            
            response = await client.post(
                "/v1/rag/query",
                json={"query": "test query"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "documents" in data


class TestMemoryEndpoints:
    @pytest.mark.asyncio
    async def test_create_conversation(self, client):
        with patch("services.llm_gateway.src.dependencies.get_memory_manager") as mock_get_memory:
            mock_memory = AsyncMock()
            mock_memory.create_conversation = AsyncMock(return_value={
                "id": "conv-123",
                "user_id": "user-123",
                "messages": [],
                "message_count": 0,
            })
            mock_get_memory.return_value = mock_memory
            
            response = await client.post(
                "/v1/memory/conversations",
                json={"user_id": "user-123"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "conv-123"


class TestToolsEndpoints:
    @pytest.mark.asyncio
    async def test_list_tools(self, client):
        with patch("services.llm_gateway.src.dependencies.get_tool_registry") as mock_get_tools:
            mock_tools = MagicMock()
            mock_tools.list_tools = MagicMock(return_value=[])
            mock_get_tools.return_value = mock_tools
            
            response = await client.get("/v1/tools")
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)