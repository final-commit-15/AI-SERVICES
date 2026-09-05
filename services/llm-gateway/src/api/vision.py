import structlog
import base64
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request, Depends, UploadFile, File, Form
from pydantic import BaseModel

from libs.schemas_common.chat import ChatMessage, ChatRequest, MessageRole
from ..router.router import ModelRouter
from ..dependencies import get_router
from ..config.settings import settings

logger = structlog.get_logger()

router = APIRouter()


class VisionAnalyzeRequest(BaseModel):
    prompt: str
    images: List[str]  # base64 encoded or URLs
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    detail: str = "auto"  # low, high, auto


class VisionChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None


@router.post("/vision/analyze")
async def analyze_image(
    request: VisionAnalyzeRequest,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Analyze images with vision model."""
    try:
        # Route to vision-capable provider
        provider_name = request.provider
        if provider_name:
            from libs.schemas_common.providers import ProviderName
            provider = router.route(provider=ProviderName(provider_name), model=request.model)
        else:
            provider = router.route(task_type="vision", model=request.model)

        if not provider.capabilities.supports_vision:
            raise HTTPException(status_code=501, detail="Provider does not support vision")

        # Build messages with images
        content = [{"type": "text", "text": request.prompt}]
        for img in request.images:
            if img.startswith("http"):
                content.append({"type": "image_url", "image_url": {"url": img, "detail": request.detail}})
            else:
                # Assume base64
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": request.detail}})

        chat_request = ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content=content)],
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        response = await provider.chat(chat_request)

        return {
            "response": response.choices[0].message.content if response.choices else "",
            "usage": response.usage.model_dump() if response.usage else {},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("vision_analyze_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vision/chat")
async def vision_chat(
    request: VisionChatRequest,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Chat with vision support."""
    try:
        provider_name = request.provider
        if provider_name:
            from libs.schemas_common.providers import ProviderName
            provider = router.route(provider=ProviderName(provider_name), model=request.model)
        else:
            provider = router.route(task_type="vision", model=request.model)

        if not provider.capabilities.supports_vision:
            raise HTTPException(status_code=501, detail="Provider does not support vision")

        chat_request = ChatRequest(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        response = await provider.chat(chat_request)

        return {
            "response": response.choices[0].message.content if response.choices else "",
            "usage": response.usage.model_dump() if response.usage else {},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("vision_chat_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vision/analyze/file")
async def analyze_image_file(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    provider: Optional[str] = Form(None),
    temperature: float = Form(0.7),
    max_tokens: Optional[int] = Form(None),
    detail: str = Form("auto"),
    http_request: Request = None,
    router: ModelRouter = Depends(get_router),
):
    """Analyze uploaded image file."""
    try:
        content = await file.read()
        import base64
        img_base64 = base64.b64encode(content).decode('utf-8')

        return await analyze_image(
            VisionAnalyzeRequest(
                prompt=prompt,
                images=[img_base64],
                model=model,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                detail=detail,
            ),
            http_request,
            router,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("vision_analyze_file_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))