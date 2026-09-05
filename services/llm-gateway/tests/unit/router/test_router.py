import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.llm_gateway.src.router.router import ModelRouter, RoutingRule, RoutingStrategy
from libs.schemas_common.chat import TaskType
from libs.schemas_common.providers import ProviderName, ProviderConfig, ModelInfo


class TestModelRouter:
    @pytest.fixture
    def router(self):
        return ModelRouter()

    @pytest.fixture
    def mock_provider_state(self):
        provider = MagicMock()
        provider.provider_name = ProviderName.OLLAMA
        provider.capabilities = MagicMock(
            supports_streaming=True,
            supports_tools=True,
            supports_vision=False,
            supports_embeddings=True,
        )
        provider.models = [
            ModelInfo(id="test-model", provider=ProviderName.OLLAMA, name="Test", context_window=4096, max_output_tokens=2048)
        ]
        provider.calculate_cost = MagicMock(return_value=0.0)
        provider.health_check = AsyncMock(return_value=True)
        
        config = ProviderConfig(
            name=ProviderName.OLLAMA,
            enabled=True,
            default_model="test-model",
            models=["test-model"],
        )
        
        return {
            "provider": provider,
            "config": config,
            "health": {"status": "healthy", "latency_ms": 100},
            "request_count": 0,
            "total_latency_ms": 0.0,
            "error_count": 0,
            "last_used": 0.0,
        }

    def test_initialization(self, router):
        assert router.providers == {}
        assert router.fallback_chain == [
            ProviderName.OLLAMA,
            ProviderName.OPENAI,
            ProviderName.ANTHROPIC,
            ProviderName.GROQ,
            ProviderName.TOGETHER,
            ProviderName.OPENROUTER,
        ]

    def test_routing_rules_initialized(self, router):
        assert TaskType.CHAT in router.routing_rules
        assert TaskType.CODING in router.routing_rules
        assert TaskType.REASONING in router.routing_rules
        assert TaskType.VISION in router.routing_rules
        assert TaskType.EMBEDDING in router.routing_rules

    @pytest.mark.asyncio
    async def test_initialize_providers_ollama_only(self, router):
        with patch.object(router, "_is_provider_configured", return_value=True):
            with patch("services.llm_gateway.src.providers.get_provider_class") as mock_get_class:
                mock_provider = AsyncMock()
                mock_provider.provider_name = ProviderName.OLLAMA
                mock_provider.capabilities = MagicMock(
                    supports_streaming=True,
                    supports_tools=True,
                    supports_vision=False,
                    supports_embeddings=True,
                )
                mock_provider.models = []
                mock_provider.initialize = AsyncMock()
                mock_provider.health_check = AsyncMock(return_value=True)
                mock_provider.list_models = AsyncMock(return_value=[])
                mock_provider.calculate_cost = MagicMock(return_value=0.0)
                
                mock_get_class.return_value = lambda **kwargs: mock_provider
                
                await router.initialize_providers()
                
                assert ProviderName.OLLAMA in router.providers

    def test_get_candidates(self, router, mock_provider_state):
        router.providers[ProviderName.OLLAMA] = type('State', (), mock_provider_state)()
        
        rule = router.routing_rules[TaskType.CHAT]
        candidates = router._get_candidates(rule, None, None, False)
        
        assert ProviderName.OLLAMA in candidates

    def test_select_provider_priority(self, router, mock_provider_state):
        router.providers[ProviderName.OLLAMA] = type('State', (), mock_provider_state)()
        router.providers[ProviderName.OPENAI] = type('State', (), mock_provider_state)()
        
        rule = router.routing_rules[TaskType.CHAT]
        rule.strategy = RoutingStrategy.PRIORITY
        rule.preferred_providers = [ProviderName.OLLAMA, ProviderName.OPENAI]
        
        selected = router._select_provider(
            [ProviderName.OLLAMA, ProviderName.OPENAI], rule, TaskType.CHAT
        )
        
        assert selected == ProviderName.OLLAMA

    def test_select_provider_cost_optimized(self, router, mock_provider_state):
        ollama_state = type('State', (), mock_provider_state)()
        ollama_state.provider.calculate_cost = MagicMock(return_value=0.001)
        
        openai_state = type('State', (), mock_provider_state)()
        openai_state.provider.calculate_cost = MagicMock(return_value=0.01)
        
        router.providers[ProviderName.OLLAMA] = ollama_state
        router.providers[ProviderName.OPENAI] = openai_state
        
        rule = router.routing_rules[TaskType.CHAT]
        rule.strategy = RoutingStrategy.COST_OPTIMIZED
        
        selected = router._select_provider(
            [ProviderName.OLLAMA, ProviderName.OPENAI], rule, TaskType.CHAT
        )
        
        assert selected == ProviderName.OLLAMA

    def test_route_explicit_provider(self, router, mock_provider_state):
        router.providers[ProviderName.OLLAMA] = type('State', (), mock_provider_state)()
        
        provider = router.route(task_type=TaskType.CHAT, provider=ProviderName.OLLAMA)
        
        assert provider == router.providers[ProviderName.OLLAMA].provider

    def test_get_provider_for_model(self, router, mock_provider_state):
        router.providers[ProviderName.OLLAMA] = type('State', (), mock_provider_state)()
        
        provider = router.get_provider_for_model("test-model")
        
        assert provider is not None

    def test_get_all_models(self, router, mock_provider_state):
        router.providers[ProviderName.OLLAMA] = type('State', (), mock_provider_state)()
        
        models = router.get_all_models()
        
        assert len(models) == 1
        assert models[0].id == "test-model"

    def test_get_provider_stats(self, router, mock_provider_state):
        router.providers[ProviderName.OLLAMA] = type('State', (), mock_provider_state)()
        
        stats = router.get_provider_stats()
        
        assert "ollama" in stats
        assert stats["ollama"]["request_count"] == 0
        assert stats["ollama"]["health"] == "healthy"