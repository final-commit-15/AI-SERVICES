from typing import Any

from ollama import AsyncClient

from config.settings import settings


class OllamaProvider:
    def __init__(self) -> None:
        self.client = AsyncClient(
            host=settings.ollama_host,
            timeout=settings.ollama_timeout,
        )

        self.model = settings.ollama_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: bool = False,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ):
        response = await self.client.chat(
            model=self.model,
            messages=messages,
            think=thinking,
            stream=False,
            format=response_format,
            options={
                "temperature": temperature,
            },
            keep_alive=settings.ollama_keep_alive,
        )

        return response