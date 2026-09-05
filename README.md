# AgentForge AI Services

![AgentForge AI Services](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Production-grade unified AI Gateway for AgentForge** — A single API to rule all LLM providers with intelligent routing, fallback, caching, and observability.

---

## 🎯 Overview

AgentForge AI Services is the central AI gateway for the AgentForge ecosystem. Every backend request goes through this service instead of directly calling LLM providers, providing:

- **Unified API** — One consistent interface for 7+ LLM providers
- **Intelligent Routing** — Task-based model selection with automatic fallback
- **Production Features** — Streaming, caching, guardrails, rate limiting, auth
- **Observability** — Health checks, metrics, cost tracking, usage analytics
- **Extensibility** — Clean provider interface for adding new models

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AI SERVICES GATEWAY                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   /v1/chat   │  │ /v1/responses│  │   /v1/embeddings     │  │
│  │  /v1/rag     │  │  /v1/memory  │  │   /v1/tools          │  │
│  │  /v1/speech  │  │  /v1/vision  │  │   /v1/images         │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         └─────────────────┼──────────────────────┘              │
│                           ▼                                     │
│              ┌────────────────────────┐                         │
│              │      Model Router      │                         │
│              │  (Task-based + Fallback)│                        │
│              └───────────┬────────────┘                         │
│                          │                                      │
│         ┌────────────────┼────────────────┐                    │
│         ▼                ▼                ▼                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Ollama    │  │   OpenAI    │  │  Anthropic  │            │
│  │  (Local)    │  │  (Cloud)    │  │  (Cloud)    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Gemini    │  │    Groq     │  │  Together   │            │
│  │  (Cloud)    │  │  (Cloud)    │  │  (Cloud)    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐                                                 │
│  │  OpenRouter │                                                 │
│  │  (Aggregator)                                                │
│  └─────────────┘                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Core APIs
- **`/v1/chat`** — Universal chat completions (OpenAI-compatible)
- **`/v1/responses`** — OpenAI Responses API compatible
- **`/v1/embeddings`** — Vector embeddings generation
- **`/v1/models`** — List available models across all providers

### Advanced Capabilities
- **Streaming** — Server-Sent Events (SSE) + WebSocket support
- **RAG Pipeline** — Document ingestion, chunking, embedding, retrieval, reranking
- **Conversation Memory** — Redis (short-term) + Qdrant (long-term/semantic)
- **Tool Calling** — Function calling with builtin tools (calculator, HTTP, datetime)
- **Vision** — Image analysis with vision-capable models
- **Speech** — STT (Whisper) + TTS synthesis
- **Image Generation** — DALL-E 3 support

### Production Features
- **Provider Health Monitoring** — Real-time health checks with automatic failover
- **Intelligent Fallback** — Automatic provider fallback on failures
- **Cost Optimization** — Route to cheapest capable model
- **Caching** — Redis + in-memory with TTL
- **Guardrails** — Input/output validation, injection detection, PII blocking
- **JWT Authentication** — Secure API access with roles/scopes
- **Rate Limiting** — Per-user/IP with burst support
- **Admin APIs** — Provider management, routing rules, analytics

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (for Ollama local models)
- API keys for cloud providers (optional)

### 1. Clone & Configure
```bash
git clone https://github.com/agentforge/ai-services.git
cd ai-services
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 2. Start with Docker Compose
```bash
docker compose up -d
```

This starts:
- **PostgreSQL** (with pgvector) on port 5432
- **Redis** on port 6379
- **Qdrant** on port 6333
- **Ollama** on port 11434
- **AI Services API** on port 8000
- **Nginx** on ports 80/443

### 3. Verify Installation
```bash
# Health check
curl http://localhost:8000/v1/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, AgentForge!"}],
    "provider": "ollama",
    "temperature": 0.7
  }'
```

---

## 📖 API Documentation

### Interactive Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/health` | Service health check |
| `GET` | `/v1/models` | List all models |
| `POST` | `/v1/chat` | Chat completion |
| `POST` | `/v1/responses` | OpenAI Responses API |
| `POST` | `/v1/embeddings` | Generate embeddings |
| `POST` | `/v1/rag/query` | RAG query |
| `POST` | `/v1/rag/ingest` | Ingest documents |
| `POST` | `/v1/memory/conversations` | Create conversation |
| `POST` | `/v1/tools/execute` | Execute tools |
| `POST` | `/v1/speech/transcribe` | Speech-to-text |
| `POST` | `/v1/vision/analyze` | Image analysis |
| `POST` | `/v1/images/generate` | Image generation |

### Example: Chat with Streaming
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a haiku about AI"}],
    "stream": true,
    "temperature": 0.8
  }' \
  --no-buffer
```

### Example: RAG Query
```bash
# Ingest documents
curl -X POST http://localhost:8000/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "./docs", "collection": "knowledge-base"}'

# Query with context
curl -X POST http://localhost:8000/v1/rag/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "collection": "knowledge-base"
  }'
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment (development/staging/production) | `development` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `QDRANT_URL` | Qdrant connection string | `http://localhost:6333` |
| `OLLAMA_HOST` | Ollama host URL | `http://localhost:11434` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `GROQ_API_KEY` | Groq API key | - |
| `TOGETHER_API_KEY` | Together AI API key | - |
| `OPENROUTER_API_KEY` | OpenRouter API key | - |
| `JWT_SECRET_KEY` | JWT signing secret (32+ chars) | *required* |
| `AUTH_ENABLED` | Enable authentication | `true` |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | `true` |

See [`.env.example`](.env.example) for complete configuration.

### Provider Configuration

Each provider can be configured independently:

```env
# Ollama (Local)
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_VISION_MODEL=llava:7b

# OpenAI
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Routing
DEFAULT_TASK_TYPE=general
DEFAULT_MODEL_PROVIDER=ollama
ROUTER_COST_OPTIMIZATION=true
ROUTER_FALLBACK_CHAIN=ollama,openai,anthropic,groq,together,openrouter
```

---

## 🔧 Provider Management

### Adding Models (Ollama)
```bash
# Pull a model
curl -X POST http://localhost:8000/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model_name": "llama3.1:8b"}'

# List installed models
curl http://localhost:8000/v1/models?provider=ollama

# Delete a model
curl -X DELETE http://localhost:8000/v1/models/llama3.1:8b?provider=ollama
```

### GPU Health Check
```bash
curl http://localhost:8000/v1/health/ollama
```

---

## 🧠 Model Routing

The router automatically selects the best provider based on:

1. **Task Type** — chat, coding, reasoning, vision, embeddings, etc.
2. **Capabilities** — Required features (tools, vision, streaming)
3. **Cost** — Cheapest capable model (when enabled)
4. **Latency** — Fastest provider (when enabled)
5. **Health** — Only healthy providers
6. **Fallback Chain** — Automatic failover

### Routing Rules (Configurable)
```python
# Default routing preferences
TASK_ROUTES = {
    "chat": ["ollama", "openai", "groq"],
    "coding": ["openai", "anthropic", "together"],
    "reasoning": ["anthropic", "openai"],
    "vision": ["ollama", "openai", "anthropic", "gemini"],
    "embeddings": ["ollama", "openai", "gemini", "together"],
    "speech": ["openai"],
    "image_generation": ["openai"],
}
```

---

## 💾 Conversation Memory

### Short-term (Redis)
- Recent messages (configurable TTL, default 1 hour)
- Fast access for active conversations

### Long-term (Qdrant)
- Semantic search across conversation history
- Vector embeddings for similarity matching
- Configurable retention (default 30 days)

### Usage
```bash
# Create conversation
curl -X POST http://localhost:8000/v1/memory/conversations \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123"}'

# Add messages
curl -X POST http://localhost:8000/v1/memory/conversations/{id}/messages \
  -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "Hello!"}'

# Semantic search
curl -X POST http://localhost:8000/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "previous discussion about AI", "user_id": "user123"}'
```

---

## 🛡 Guardrails

Built-in safety checks:

- **Input Validation** — Length limits, injection detection
- **Output Validation** — Toxicity, PII, length limits
- **Configurable** — Enable/disable per environment

```env
ENABLE_INPUT_GUARDRAILS=true
ENABLE_OUTPUT_GUARDRAILS=true
GUARDRAILS_MAX_INPUT_LENGTH=10000
GUARDRAILS_BLOCK_INJECTION=true
GUARDRAILS_BLOCK_PII=true
```

---

## 📊 Monitoring & Analytics

### Health Checks
```bash
# Overall health
curl http://localhost:8000/v1/health

# Provider-specific
curl http://localhost:8000/v1/health/providers
```

### Prometheus Metrics
Available at `/metrics`:
- `http_requests_total` — Request count by method/endpoint/status
- `http_request_duration_seconds` — Request latency
- `provider_requests_total` — Provider request count
- `token_usage_total` — Token consumption
- `cost_usd_total` — Cost tracking

### Admin Analytics
```bash
# Usage analytics (admin only)
curl -H "X-Admin-Key: your-admin-key" \
  http://localhost:8000/v1/admin/analytics

# Cost breakdown
curl -H "X-Admin-Key: your-admin-key" \
  http://localhost:8000/v1/admin/costs
```

---

## 🔐 Authentication

### JWT Tokens
```bash
# Login
curl -X POST http://localhost:8000/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use token
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/v1/chat
```

### API Keys
```bash
# Create API key (admin)
curl -X POST http://localhost:8000/v1/admin/api-keys \
  -H "X-Admin-Key: your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app", "roles": ["user"]}'

# Use API key
curl -H "X-API-Key: afk_xxx..." \
  http://localhost:8000/v1/chat
```

---

## 🐳 Deployment

### Docker Compose (Recommended)
```bash
# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Development with hot reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Kubernetes
Helm chart available in [`deploy/helm`](deploy/helm).

### Manual Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start server
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=services --cov=rag --cov=libs

# Unit tests only
pytest services/llm-gateway/tests/unit

# Integration tests
pytest services/llm-gateway/tests/integration

# Specific provider tests
pytest services/llm-gateway/tests/unit/providers/test_ollama.py
```

---

## 📁 Project Structure

```
agentforge-ai-services/
├── docker-compose.yml          # Full stack deployment
├── Dockerfile                  # API service image
├── requirements.txt            # Python dependencies
├── .env.example               # Configuration template
├── nginx/
│   └── nginx.conf             # Reverse proxy config
├── docker/
│   └── init-postgres.sql      # Database initialization
├── libs/
│   ├── llm-common/            # Base provider interface
│   ├── embeddings-common/     # Embedding abstractions
│   └── schemas-common/        # Shared Pydantic models
├── rag/                       # RAG pipeline components
│   ├── ingestion/             # Document loading & chunking
│   ├── embeddings/            # Embedding services
│   ├── retrieval/             # Vector search & reranking
│   └── vectorstores/          # Qdrant, FAISS implementations
├── services/llm-gateway/
│   ├── src/
│   │   ├── api/               # FastAPI routes
│   │   ├── config/            # Settings management
│   │   ├── providers/         # LLM provider implementations
│   │   ├── router/            # Intelligent model router
│   │   ├── caching/           # Cache layer
│   │   ├── guardrails/        # Safety checks
│   │   ├── memory/            # Conversation memory
│   │   ├── tools/             # Function calling
│   │   ├── speech/            # STT/TTS
│   │   ├── vision/            # Image analysis
│   │   ├── image_gen/         # Image generation
│   │   ├── health/            # Health monitoring
│   │   ├── analytics/         # Usage tracking
│   │   ├── admin/             # Admin APIs
│   │   ├── auth/              # Authentication
│   │   ├── rate_limit/        # Rate limiting
│   │   ├── middleware/        # HTTP middleware
│   │   └── dependencies.py    # Service initialization
│   └── tests/                 # Unit & integration tests
└── docs/                      # Documentation
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### Development Guidelines
- Follow existing code style (type hints, async/await)
- Add tests for new features
- Update documentation
- Run linting: `ruff check .` and `mypy .`

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) — Modern web framework
- [Ollama](https://ollama.ai/) — Local LLM inference
- [Qdrant](https://qdrant.tech/) — Vector database
- [pgvector](https://github.com/pgvector/pgvector) — PostgreSQL vector extension
- All LLM providers for their APIs

---

**AgentForge AI Services** — Powering intelligent agents with unified AI capabilities.