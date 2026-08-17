from typing import Any, Dict, Optional
from openai import AsyncOpenAI
from ..base import LLMProvider
from ...config.settings import settings


class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    @property
    def model_name(self) -> str:
        return self.model

    async def chat(self,
                   messages: list[dict[str, Any]],
                   *,
                   thinking: bool = False,
                   temperature: float = 0.0,
                   response_format: Optional[Dict[str, Any]] = None,
                   **kwargs) -> Dict[str, Any]:
        # If thinking is enabled, use reasoning model? For now ignore.
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,  # may need to map
            **kwargs
        )
        return {
            "message": {
                "role": "assistant",
                "content": resp.choices[0].message.content
            },
            "usage": resp.usage.model_dump() if resp.usage else {}
        }

    async def generate(self,
                       prompt: str,
                       *,
                       temperature: float = 0.0,
                       max_tokens: Optional[int] = None,
                       **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        resp = await self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        return resp["message"]["content"]