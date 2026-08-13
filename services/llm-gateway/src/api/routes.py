from fastapi import APIRouter

from providers.local.ollama_provider import OllamaProvider

router = APIRouter()

ollama = OllamaProvider()


@router.get("/health/ollama")
async def ollama_health():
    try:
        response = await ollama.client.list()

        return {
            "status": "healthy",
            "provider": "ollama",
            "model": ollama.model,
            "models": [
                model.model
                for model in response.models
            ],
        }

    except Exception as exc:
        return {
            "status": "unhealthy",
            "provider": "ollama",
            "error": str(exc),
        }