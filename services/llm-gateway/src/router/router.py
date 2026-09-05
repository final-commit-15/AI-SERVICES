import structlog
import random
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from libs.llm_common.base import BaseProvider
from libs.schemas_common.chat import TaskType
from libs.schemas_common.providers import ProviderName, ModelInfo, ProviderConfig
from ..providers import get_provider_class, PROVIDER_MAP
from ..config.settings import settings

logger = structlog.get_logger()


class RoutingStrategy(str, Enum):
    PRIORITY = "priority"
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"


@dataclass
class RoutingRule:
    task_type: TaskType
    preferred_providers: List[ProviderName]
    preferred_models: Dict[ProviderName, str]
    strategy: RoutingStrategy = RoutingStrategy.PRIORITY
    fallback_enabled: bool = True
    max_cost_per_1k: Optional[float] = None
    max_latency_ms: Optional[int] = None
    required_capabilities: List[str] = None


@dataclass
class ProviderState:
    provider: BaseProvider
    config: ProviderConfig
    health: Dict[str, Any]
    request_count: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0
    last_used: float = 0.0


class ModelRouter:
    """Intelligent model router with task-based routing, fallback, and cost optimization."""

    def __init__(self):
        self.providers: Dict[ProviderName, ProviderState] = {}
        self.routing_rules: Dict[TaskType, RoutingRule] = {}
        self.fallback_chain: List[ProviderName] = []
        self.round_robin_counters: Dict[TaskType, int] = {}
        self._initialize_default_rules()
        self._parse_fallback_chain()

    def _initialize_default_rules(self):
        """Initialize default routing rules from settings."""
        self.routing_rules = {
            TaskType.CHAT: RoutingRule(
                task_type=TaskType.CHAT,
                preferred_providers=[ProviderName.OLLAMA, ProviderName.OPENAI, ProviderName.GROQ],
                preferred_models={
                    ProviderName.OLLAMA: settings.ollama_model,
                    ProviderName.OPENAI: settings.openai_model,
                    ProviderName.GROQ: settings.groq_model,
                },
                strategy=RoutingStrategy.COST_OPTIMIZED if settings.router_cost_optimization else RoutingStrategy.PRIORITY,
            ),
            TaskType.CODING: RoutingRule(
                task_type=TaskType.CODING,
                preferred_providers=[ProviderName.OPENAI, ProviderName.ANTHROPIC, ProviderName.TOGETHER],
                preferred_models={
                    ProviderName.OPENAI: "gpt-4o",
                    ProviderName.ANTHROPIC: "claude-3-5-sonnet-20241022",
                    ProviderName.TOGETHER: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                },
                strategy=RoutingStrategy.PRIORITY,
            ),
            TaskType.REASONING: RoutingRule(
                task_type=TaskType.REASONING,
                preferred_providers=[ProviderName.ANTHROPIC, ProviderName.OPENAI],
                preferred_models={
                    ProviderName.ANTHROPIC: "claude-3-5-sonnet-20241022",
                    ProviderName.OPENAI: "o1-preview",
                },
                strategy=RoutingStrategy.PRIORITY,
            ),
            TaskType.SUMMARIZATION: RoutingRule(
                task_type=TaskType.SUMMARIZATION,
                preferred_providers=[ProviderName.OLLAMA, ProviderName.OPENAI, ProviderName.GROQ],
                preferred_models={
                    ProviderName.OLLAMA: settings.ollama_model,
                    ProviderName.OPENAI: "gpt-4o-mini",
                    ProviderName.GROQ: "llama-3.1-8b-instant",
                },
                strategy=RoutingStrategy.COST_OPTIMIZED,
            ),
            TaskType.VISION: RoutingRule(
                task_type=TaskType.VISION,
                preferred_providers=[ProviderName.OLLAMA, ProviderName.OPENAI, ProviderName.ANTHROPIC, ProviderName.GEMINI],
                preferred_models={
                    ProviderName.OLLAMA: settings.ollama_vision_model,
                    ProviderName.OPENAI: "gpt-4o",
                    ProviderName.ANTHROPIC: "claude-3-5-sonnet-20241022",
                    ProviderName.GEMINI: "gemini-1.5-pro",
                },
                required_capabilities=["vision"],
                strategy=RoutingStrategy.PRIORITY,
            ),
            TaskType.EMBEDDING: RoutingRule(
                task_type=TaskType.EMBEDDING,
                preferred_providers=[ProviderName.OLLAMA, ProviderName.OPENAI, ProviderName.GEMINI, ProviderName.TOGETHER],
                preferred_models={
                    ProviderName.OLLAMA: settings.ollama_embedding_model,
                    ProviderName.OPENAI: settings.openai_embedding_model,
                    ProviderName.GEMINI: settings.gemini_embedding_model,
                    ProviderName.TOGETHER: settings.together_embedding_model,
                },
                required_capabilities=["embeddings"],
                strategy=RoutingStrategy.COST_OPTIMIZED,
            ),
            TaskType.SPEECH: RoutingRule(
                task_type=TaskType.SPEECH,
                preferred_providers=[ProviderName.OPENAI],
                preferred_models={
                    ProviderName.OPENAI: settings.speech_stt_model,
                },
                required_capabilities=["audio"],
                strategy=RoutingStrategy.PRIORITY,
            ),
            TaskType.IMAGE_GEN: RoutingRule(
                task_type=TaskType.IMAGE_GEN,
                preferred_providers=[ProviderName.OPENAI],
                preferred_models={
                    ProviderName.OPENAI: settings.image_gen_model,
                },
                required_capabilities=["image_gen"],
                strategy=RoutingStrategy.PRIORITY,
            ),
            TaskType.GENERAL: RoutingRule(
                task_type=TaskType.GENERAL,
                preferred_providers=[ProviderName.OLLAMA, ProviderName.OPENAI, ProviderName.GROQ],
                preferred_models={
                    ProviderName.OLLAMA: settings.ollama_model,
                    ProviderName.OPENAI: settings.openai_model,
                    ProviderName.GROQ: settings.groq_model,
                },
                strategy=RoutingStrategy.COST_OPTIMIZED if settings.router_cost_optimization else RoutingStrategy.PRIORITY,
            ),
        }

    def _parse_fallback_chain(self):
        """Parse fallback chain from settings."""
        if settings.router_fallback_chain:
            self.fallback_chain = [
                ProviderName(p.strip())
                for p in settings.router_fallback_chain.split(",")
                if p.strip() in [p.value for p in ProviderName]
            ]
        else:
            self.fallback_chain = [
                ProviderName.OLLAMA,
                ProviderName.OPENAI,
                ProviderName.ANTHROPIC,
                ProviderName.GROQ,
                ProviderName.TOGETHER,
                ProviderName.OPENROUTER,
            ]

    async def initialize_providers(self):
        """Initialize all configured providers."""
        provider_configs = {
            ProviderName.OLLAMA: {
                "host": settings.ollama_host,
                "default_model": settings.ollama_model,
                "embedding_model": settings.ollama_embedding_model,
                "vision_model": settings.ollama_vision_model,
                "timeout": settings.ollama_timeout,
            },
            ProviderName.OPENAI: {
                "api_key": settings.openai_api_key,
                "organization": settings.openai_org_id,
                "default_model": settings.openai_model,
                "embedding_model": settings.openai_embedding_model,
                "timeout": settings.openai_timeout,
                "max_retries": settings.openai_max_retries,
            },
            ProviderName.ANTHROPIC: {
                "api_key": settings.anthropic_api_key,
                "default_model": settings.anthropic_model,
                "timeout": settings.anthropic_timeout,
                "max_retries": settings.anthropic_max_retries,
            },
            ProviderName.GEMINI: {
                "api_key": settings.gemini_api_key,
                "default_model": settings.gemini_model,
                "embedding_model": settings.gemini_embedding_model,
                "timeout": settings.gemini_timeout,
                "max_retries": settings.gemini_max_retries,
            },
            ProviderName.GROQ: {
                "api_key": settings.groq_api_key,
                "default_model": settings.groq_model,
                "timeout": settings.groq_timeout,
                "max_retries": settings.groq_max_retries,
            },
            ProviderName.TOGETHER: {
                "api_key": settings.together_api_key,
                "default_model": settings.together_model,
                "embedding_model": settings.together_embedding_model,
                "timeout": settings.together_timeout,
                "max_retries": settings.together_max_retries,
            },
            ProviderName.OPENROUTER: {
                "api_key": settings.openrouter_api_key,
                "base_url": settings.openrouter_base_url,
                "default_model": settings.openrouter_model,
                "timeout": settings.openrouter_timeout,
                "max_retries": settings.openrouter_max_retries,
            },
        }

        for provider_name, config in provider_configs.items():
            if self._is_provider_configured(provider_name, config):
                try:
                    provider_class = get_provider_class(provider_name)
                    provider = provider_class(**config)
                    await provider.initialize()

                    provider_config = ProviderConfig(
                        name=provider_name,
                        enabled=True,
                        default_model=config.get("default_model", ""),
                        models=[m.id for m in provider.models],
                    )

                    self.providers[provider_name] = ProviderState(
                        provider=provider,
                        config=provider_config,
                        health={"status": "healthy", "latency_ms": 0},
                    )
                    logger.info("provider_initialized", provider=provider_name.value)
                except Exception as e:
                    logger.error("provider_initialization_failed", provider=provider_name.value, error=str(e))

    def _is_provider_configured(self, provider_name: ProviderName, config: Dict[str, Any]) -> bool:
        """Check if provider has required configuration."""
        if provider_name == ProviderName.OLLAMA:
            return True  # Ollama doesn't need API key
        return bool(config.get("api_key"))

    async def close_providers(self):
        """Close all provider connections."""
        for state in self.providers.values():
            try:
                await state.provider.close()
            except Exception as e:
                logger.error("provider_close_failed", provider=state.config.name.value, error=str(e))

    def route(
        self,
        task_type: TaskType = TaskType.GENERAL,
        provider: Optional[ProviderName] = None,
        model: Optional[str] = None,
        required_capabilities: List[str] = None,
        max_cost: Optional[float] = None,
        prefer_speed: bool = False,
    ) -> BaseProvider:
        """Route request to best provider."""
        # If provider explicitly specified, use it
        if provider and provider in self.providers:
            return self._get_available_provider(provider)

        # Get routing rule for task type
        rule = self.routing_rules.get(task_type, self.routing_rules[TaskType.GENERAL])

        # Override capabilities if specified
        if required_capabilities:
            rule.required_capabilities = required_capabilities

        # Get candidate providers
        candidates = self._get_candidates(rule, model, max_cost, prefer_speed)

        if not candidates:
            # Fallback to any available provider
            candidates = list(self.providers.keys())

        # Select provider based on strategy
        selected = self._select_provider(candidates, rule, task_type)

        if not selected and rule.fallback_enabled:
            # Try fallback chain
            selected = self._try_fallback(rule, model, max_cost, prefer_speed)

        if not selected:
            raise ValueError(f"No available provider for task type: {task_type}")

        # Update stats
        self.providers[selected].request_count += 1
        self.providers[selected].last_used = time.time()

        return self.providers[selected].provider

    def _get_candidates(
        self,
        rule: RoutingRule,
        model: Optional[str],
        max_cost: Optional[float],
        prefer_speed: bool,
    ) -> List[ProviderName]:
        """Get candidate providers matching criteria."""
        candidates = []

        for provider_name in rule.preferred_providers:
            if provider_name not in self.providers:
                continue

            state = self.providers[provider_name]
            provider = state.provider

            # Check if model matches
            if model and not any(m.id == model for m in provider.models):
                continue

            # Check capabilities
            if rule.required_capabilities:
                if not all(getattr(provider.capabilities, f"supports_{cap}", False) for cap in rule.required_capabilities):
                    continue

            # Check health
            if state.health.get("status") == "unhealthy":
                continue

            # Check cost
            if max_cost and provider.calculate_cost(1000, 1000) > max_cost:
                continue

            candidates.append(provider_name)

        return candidates

    def _select_provider(
        self,
        candidates: List[ProviderName],
        rule: RoutingRule,
        task_type: TaskType,
    ) -> Optional[ProviderName]:
        """Select provider based on strategy."""
        if not candidates:
            return None

        if rule.strategy == RoutingStrategy.PRIORITY:
            return candidates[0]

        elif rule.strategy == RoutingStrategy.COST_OPTIMIZED:
            return min(candidates, key=lambda p: self._estimate_cost(p))

        elif rule.strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            return min(candidates, key=lambda p: self.providers[p].health.get("latency_ms", 999999))

        elif rule.strategy == RoutingStrategy.ROUND_ROBIN:
            counter = self.round_robin_counters.get(task_type, 0)
            selected = candidates[counter % len(candidates)]
            self.round_robin_counters[task_type] = counter + 1
            return selected

        elif rule.strategy == RoutingStrategy.WEIGHTED:
            # Weight by inverse of cost and latency
            weights = {}
            for p in candidates:
                cost = self._estimate_cost(p)
                latency = self.providers[p].health.get("latency_ms", 1000)
                weights[p] = 1.0 / (cost * 0.001 + latency * 0.0001 + 0.01)
            return max(candidates, key=lambda p: weights[p])

        return candidates[0]

    def _estimate_cost(self, provider_name: ProviderName) -> float:
        """Estimate cost per 1k tokens for provider."""
        state = self.providers[provider_name]
        return state.provider.calculate_cost(1000, 1000)

    def _try_fallback(
        self,
        rule: RoutingRule,
        model: Optional[str],
        max_cost: Optional[float],
        prefer_speed: bool,
    ) -> Optional[ProviderName]:
        """Try fallback chain."""
        for provider_name in self.fallback_chain:
            if provider_name not in self.providers:
                continue
            if provider_name in rule.preferred_providers:
                continue  # Already tried

            state = self.providers[provider_name]
            if state.health.get("status") == "unhealthy":
                continue

            provider = state.provider
            if model and not any(m.id == model for m in provider.models):
                continue

            if rule.required_capabilities:
                if not all(getattr(provider.capabilities, f"supports_{cap}", False) for cap in rule.required_capabilities):
                    continue

            logger.info("fallback_provider_selected", provider=provider_name.value, task_type=rule.task_type.value)
            return provider_name

        return None

    def _get_available_provider(self, provider_name: ProviderName) -> BaseProvider:
        """Get provider if available, otherwise raise."""
        if provider_name not in self.providers:
            raise ValueError(f"Provider not available: {provider_name}")

        state = self.providers[provider_name]
        if state.health.get("status") == "unhealthy":
            logger.warning("provider_unhealthy_but_requested", provider=provider_name.value)

        return state.provider

    async def update_health(self, provider_name: ProviderName, health_data: Dict[str, Any]):
        """Update provider health status."""
        if provider_name in self.providers:
            self.providers[provider_name].health = health_data

    async def record_request(
        self,
        provider_name: ProviderName,
        latency_ms: float,
        success: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        """Record request metrics for routing decisions."""
        if provider_name in self.providers:
            state = self.providers[provider_name]
            state.total_latency_ms += latency_ms
            if not success:
                state.error_count += 1
            # Update rolling average latency
            if state.request_count > 0:
                state.health["latency_ms"] = state.total_latency_ms / state.request_count

    def get_provider_for_model(self, model: str) -> Optional[BaseProvider]:
        """Find provider that serves a specific model."""
        for state in self.providers.values():
            if any(m.id == model for m in state.provider.models):
                return state.provider
        return None

    def get_all_models(self) -> List[ModelInfo]:
        """Get all models from all providers."""
        models = []
        for state in self.providers.values():
            models.extend(state.provider.models)
        return models

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers."""
        return {
            name.value: {
                "request_count": state.request_count,
                "error_count": state.error_count,
                "avg_latency_ms": state.health.get("latency_ms", 0),
                "health": state.health.get("status", "unknown"),
                "models": [m.id for m in state.provider.models],
            }
            for name, state in self.providers.items()
        }