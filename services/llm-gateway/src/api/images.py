import structlog
import base64
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends, Form, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO

from ..router.router import ModelRouter
from ..dependencies import get_router
from ..config.settings import settings

logger = structlog.get_logger()

router = APIRouter()


class ImageGenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    provider: Optional[str] = None
    n: int = 1
    size: str = "1024x1024"
    quality: str = "standard"
    style: str = "vivid"
    response_format: str = "url"  # url or b64_json


class ImageEditRequest(BaseModel):
    image: str  # base64
    prompt: str
    model: Optional[str] = None
    provider: Optional[str] = None
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "url"


class ImageVariationRequest(BaseModel):
    image: str  # base64
    model: Optional[str] = None
    provider: Optional[str] = None
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "url"


@router.post("/images/generate")
async def generate_image(
    request: ImageGenerateRequest,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Generate images."""
    try:
        provider_name = request.provider
        if provider_name:
            from libs.schemas_common.providers import ProviderName
            provider = router.route(provider=ProviderName(provider_name), model=request.model)
        else:
            provider = router.route(task_type="image_generation", model=request.model)

        if not provider.capabilities.supports_image_gen:
            raise HTTPException(status_code=501, detail="Provider does not support image generation")

        images = []
        for _ in range(request.n):
            image_data = await provider.generate_image(
                prompt=request.prompt,
                model=request.model or settings.image_gen_model,
                size=request.size,
                quality=request.quality,
                style=request.style,
            )

            if request.response_format == "b64_json":
                images.append({"b64_json": base64.b64encode(image_data).decode('utf-8')})
            else:
                # In production, upload to storage and return URL
                images.append({"url": f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"})

        return {"created": int(time.time()), "data": images}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("generate_image_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/edit")
async def edit_image(
    request: ImageEditRequest,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Edit an image."""
    try:
        import base64
        image_data = base64.b64decode(request.image)

        provider_name = request.provider
        if provider_name:
            from libs.schemas_common.providers import ProviderName
            provider = router.route(provider=ProviderName(provider_name), model=request.model)
        else:
            provider = router.route(task_type="image_generation", model=request.model)

        if not hasattr(provider, 'edit_image'):
            raise HTTPException(status_code=501, detail="Provider does not support image editing")

        edited_data = await provider.edit_image(
            image=image_data,
            prompt=request.prompt,
            model=request.model or settings.image_gen_model,
            size=request.size,
        )

        if request.response_format == "b64_json":
            return {"data": [{"b64_json": base64.b64encode(edited_data).decode('utf-8')}]}
        else:
            return {"data": [{"url": f"data:image/png;base64,{base64.b64encode(edited_data).decode('utf-8')}"}]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("edit_image_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/variations")
async def create_variation(
    request: ImageVariationRequest,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Create image variation."""
    try:
        import base64
        image_data = base64.b64decode(request.image)

        provider_name = request.provider
        if provider_name:
            from libs.schemas_common.providers import ProviderName
            provider = router.route(provider=ProviderName(provider_name), model=request.model)
        else:
            provider = router.route(task_type="image_generation", model=request.model)

        if not hasattr(provider, 'create_variation'):
            raise HTTPException(status_code=501, detail="Provider does not support image variation")

        variation_data = await provider.create_variation(
            image=image_data,
            model=request.model or settings.image_gen_model,
            size=request.size,
        )

        if request.response_format == "b64_json":
            return {"data": [{"b64_json": base64.b64encode(variation_data).decode('utf-8')}]}
        else:
            return {"data": [{"url": f"data:image/png;base64,{base64.b64encode(variation_data).decode('utf-8')}"}]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("create_variation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/generate/file")
async def generate_image_file(
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    provider: Optional[str] = Form(None),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    quality: str = Form("standard"),
    style: str = Form("vivid"),
    http_request: Request = None,
    router: ModelRouter = Depends(get_router),
):
    """Generate image and return as file."""
    try:
        provider_name = provider
        if provider_name:
            from libs.schemas_common.providers import ProviderName
            provider_obj = router.route(provider=ProviderName(provider_name), model=model)
        else:
            provider_obj = router.route(task_type="image_generation", model=model)

        if not provider_obj.capabilities.supports_image_gen:
            raise HTTPException(status_code=501, detail="Provider does not support image generation")

        image_data = await provider_obj.generate_image(
            prompt=prompt,
            model=model or settings.image_gen_model,
            size=size,
            quality=quality,
            style=style,
        )

        return StreamingResponse(
            BytesIO(image_data),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=generated.png"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("generate_image_file_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


import time