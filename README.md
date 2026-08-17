```markdown
# AgentForge AI Services

AgentForge AI Services provides reusable AI capabilities to the rest of the AgentForge ecosystem. It acts as a centralised gateway for Large Language Models (LLMs), embeddings, Retrieval-Augmented Generation (RAG), prompt management, caching, and guardrails—all exposed through a clean, unified API.

## Architecture

The service is built around six core capabilities:

- **LLM Gateway** – Unified interface to multiple LLM providers (Ollama, OpenAI, Anthropic, etc.) with intelligent model routing.
- **Prompt Registry** – Versioned, centrally managed prompt templates.
- **Caching** – In‑memory (and optionally Redis) cache to reduce latency and cost.
- **Guardrails** – Input and output validation, injection detection, and safety checks.
- **Embeddings** – Common abstraction for embedding models, used for semantic search and RAG.
- **RAG Pipeline** – End‑to‑end ingestion, chunking, embedding, vector storage, retrieval, and reranking.

```
┌─────────────────────────────────────────────────────────────┐
│                       AI-SERVICES                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   LLM Gateway                       │    │
│  │  ┌────────┐  ┌──────────┐  ┌───────────────┐      │    │
│  │  │ Router │  │ Provider │  │ Caching       │      │    │
│  │  └────────┘  └──────────┘  └───────────────┘      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │    │
│  │  │Guardrails│  │  Prompt  │  │ Embeddings   │    │    │
│  │  │          │  │ Registry │  │              │    │    │
│  │  └──────────┘  └──────────┘  └──────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                      RAG                           │    │
│  │  ┌──────────┐  ┌────────────┐  ┌──────────────┐  │    │
│  │  │Ingestion │  │ Embedding  │  │  Vector Store│  │    │
│  │  │Chunking  │  │  Service   │  │ (FAISS)      │  │    │
│  │  └──────────┘  └────────────┘  └──────────────┘  │    │
│  │  ┌──────────┐  ┌────────────┐                     │    │
│  │  │Retrieval │  │ Reranking  │                     │    │
│  │  └──────────┘  └────────────┘                     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Unified LLM API** – Single `/chat` and `/generate` endpoints that work with any configured provider.
- **Model Routing** – Automatically select the best model based on task type, complexity, or explicit provider preference.
- **Prompt Templates** – Versioned YAML templates for consistent prompt engineering.
- **Response Caching** – Reduce costs and latency with TTL‑based caching (in‑memory or Redis).
- **Guardrails** – Input sanitisation, injection detection, and output safety checks.
- **Embeddings** – Generate vector embeddings for any text (supports Ollama, OpenAI, and others).
- **RAG Pipeline** – Ingest documents, chunk them, generate embeddings, store in a vector database, and retrieve relevant context for queries.
- **Health Checks** – Monitor provider availability and service status.

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) (or alternative LLM provider) running locally or accessible
- (Optional) Redis for persistent caching

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/agentforge-ai-services.git
   cd agentforge-ai-services
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file and adjust as needed:
   ```bash
   cp .env.example .env
   ```

### Configuration

Edit `.env` to set your LLM providers, embedding models, cache settings, etc.

```env
# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
OLLAMA_TIMEOUT=120
OLLAMA_KEEP_ALIVE=10m

# OpenAI (optional)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Router
DEFAULT_TASK_TYPE=general
DEFAULT_MODEL_PROVIDER=ollama

# Caching
CACHE_ENABLED=true
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=1000

# Guardrails
ENABLE_INPUT_GUARDRAILS=true
ENABLE_OUTPUT_GUARDRAILS=true

# Embeddings
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768

# RAG
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=50
RAG_TOP_K=5
VECTOR_STORE_TYPE=faiss
```

### Running the Service

Start the LLM Gateway with Uvicorn:

```bash
uvicorn services.llm-gateway.src.main:app --reload --host 0.0.0.0 --port 8000
```

Or using Docker (if you have a `Dockerfile`):

```bash
docker build -t agentforge-ai-services .
docker run -p 8000:8000 --env-file .env agentforge-ai-services
```

The API will be available at `http://localhost:8000/v1`.

## API Endpoints

All endpoints are prefixed with `/v1`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a chat completion request with optional routing and guardrails. |
| `POST` | `/generate` | Simple text generation from a prompt. |
| `POST` | `/embed` | Generate embeddings for a list of texts. |
| `POST` | `/rag/ingest` | Ingest a document (file) into the vector store. |
| `POST` | `/rag/query` | Query the RAG pipeline and retrieve relevant documents. |
| `GET` | `/models` | List available models from configured providers. |
| `GET` | `/health` | Service health check. |
| `GET` | `/health/ollama` | Check Ollama connectivity and list models. |

### Example Usage

**Chat completion:**
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "provider": "ollama",
    "temperature": 0.7
  }'
```

**RAG query:**
```bash
curl -X POST http://localhost:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

## Project Structure

```
agentforge-ai-services/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── libs/
│   └── embeddings-common/           # Shared embedding abstraction
│       ├── base.py
│       ├── models.py
│       └── client.py
├── rag/                             # RAG pipeline
│   ├── ingestion/
│   ├── embeddings/
│   ├── retrieval/
│   ├── vectorstores/
│   └── pipeline.py
└── services/
    └── llm-gateway/
        ├── src/
        │   ├── api/                 # FastAPI routes
        │   ├── config/              # Settings
        │   ├── providers/           # LLM provider implementations
        │   ├── router/              # Model routing logic
        │   ├── caching/             # Cache layer
        │   ├── guardrails/          # Input/output validation
        │   ├── prompt_registry/     # Prompt templates
        │   └── main.py              # Application entry point
        ├── tests/
        ├── Dockerfile
        └── docker-compose.yml
```

## Testing

Run unit and integration tests with:

```bash
pytest tests/
```

To include coverage:

```bash
pytest --cov=services --cov=rag tests/
```

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/amazing-feature`.
3. Commit your changes: `git commit -m 'Add amazing feature'`.
4. Push to the branch: `git push origin feature/amazing-feature`.
5. Open a Pull Request.

Please ensure your code passes all tests and follows the existing style conventions.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

**AgentForge AI Services** – Powering intelligent agents with reusable AI capabilities.
```