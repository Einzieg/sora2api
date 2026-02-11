"""API routes - OpenAI compatible endpoints"""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import base64
import json
import re
import time
from ..core.auth import verify_api_key_header
from ..core.models import ChatCompletionRequest, ImageGenerationRequest
from ..services.generation_handler import GenerationHandler, MODEL_CONFIG
from ..core.logger import debug_logger

router = APIRouter()

# Dependency injection will be set up in main.py
generation_handler: GenerationHandler = None
IMAGE_MODEL_ALIASES = {
    "gpt-image-1": "gpt-image",
    "dall-e-3": "gpt-image",
    "dall-e-2": "gpt-image"
}

def set_generation_handler(handler: GenerationHandler):
    """Set generation handler instance"""
    global generation_handler
    generation_handler = handler

def _build_error_response(message: str, error_type: str = "server_error", code: Optional[str] = None) -> dict:
    """Build OpenAI-compatible error response"""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code
        }
    }

def _resolve_image_model(model: Optional[str], size: Optional[str]) -> str:
    """Resolve image model from request model or size"""
    if model:
        mapped_model = IMAGE_MODEL_ALIASES.get(model, model)
        model_config = MODEL_CONFIG.get(mapped_model)
        if not model_config or model_config.get("type") != "image":
            raise ValueError(f"Invalid image model: {model}")
        return mapped_model

    if size:
        match = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", size.lower())
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            if width > height:
                return "gpt-image-landscape"
            if height > width:
                return "gpt-image-portrait"

    return "gpt-image"

def _extract_image_urls(markdown_content: str) -> List[str]:
    """Extract generated image URLs from markdown content"""
    if not markdown_content:
        return []

    urls = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", markdown_content)
    if not urls:
        urls = re.findall(r"https?://[^\s)]+", markdown_content)

    deduplicated = []
    for url in urls:
        if url not in deduplicated:
            deduplicated.append(url)
    return deduplicated

async def _generate_image_urls(model: str, prompt: str) -> List[str]:
    """Run image generation in streaming mode and collect final URLs"""
    content_chunks = []

    async for chunk in generation_handler.handle_generation_with_retry(
        model=model,
        prompt=prompt,
        image=None,
        video=None,
        remix_target_id=None,
        stream=True
    ):
        if not isinstance(chunk, str) or not chunk.startswith("data: "):
            continue

        payload = chunk[6:].strip()
        if payload == "[DONE]":
            break

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if isinstance(data, dict) and "error" in data:
            error_data = data.get("error", {})
            error_msg = error_data.get("message") or "Image generation failed"
            raise Exception(error_msg)

        for choice in data.get("choices", []):
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                content_chunks.append(content)

    image_urls = _extract_image_urls("".join(content_chunks))
    if not image_urls:
        raise Exception("Image generation completed but no image URL was returned")

    return image_urls

async def _image_url_to_base64(url: str) -> str:
    """Convert image URL to base64"""
    if "/tmp/" in url:
        filename = url.split("/tmp/", 1)[1].split("?", 1)[0].split("#", 1)[0]
        filename = Path(filename).name
        local_file = Path("tmp") / filename
        if local_file.exists():
            return base64.b64encode(local_file.read_bytes()).decode("utf-8")

    image_bytes = await generation_handler._download_file(url)
    return base64.b64encode(image_bytes).decode("utf-8")

def _extract_remix_id(text: str) -> str:
    """Extract remix ID from text

    Supports two formats:
    1. Full URL: https://sora.chatgpt.com/p/s_68e3a06dcd888191b150971da152c1f5
    2. Short ID: s_68e3a06dcd888191b150971da152c1f5

    Args:
        text: Text to search for remix ID

    Returns:
        Remix ID (s_[a-f0-9]{32}) or empty string if not found
    """
    if not text:
        return ""

    # Match Sora share link format: s_[a-f0-9]{32}
    match = re.search(r's_[a-f0-9]{32}', text)
    if match:
        return match.group(0)

    return ""

@router.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key_header)):
    """List available models"""
    models = []

    for model_id, config in MODEL_CONFIG.items():
        description = f"{config['type'].capitalize()} generation"
        if config['type'] == 'image':
            description += f" - {config['width']}x{config['height']}"
        elif config['type'] == 'video':
            description += f" - {config['orientation']}"
        elif config['type'] == 'prompt_enhance':
            description += f" - {config['expansion_level']} ({config['duration_s']}s)"

        models.append({
            "id": model_id,
            "object": "model",
            "owned_by": "sora2api",
            "description": description
        })

    return {
        "object": "list",
        "data": models
    }

@router.post("/v1/images/generations")
async def create_image_generation(
    request: ImageGenerationRequest,
    api_key: str = Depends(verify_api_key_header),
    http_request: Request = None
):
    """OpenAI-compatible image generation endpoint"""
    start_time = time.time()

    try:
        debug_logger.log_request(
            method="POST",
            url="/v1/images/generations",
            headers=dict(http_request.headers) if http_request else {},
            body=request.dict(),
            source="Client"
        )

        prompt = (request.prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        if request.n is not None and request.n < 1:
            raise ValueError("Parameter n must be greater than 0")

        # The upstream Sora image flow generates one task result per request.
        if request.n and request.n > 1:
            raise ValueError("This service currently supports n=1 only")

        response_format = (request.response_format or "url").lower()
        if response_format not in ("url", "b64_json"):
            raise ValueError(f"Unsupported response_format: {request.response_format}")

        model = _resolve_image_model(request.model, request.size)
        image_urls = await _generate_image_urls(model=model, prompt=prompt)

        if response_format == "b64_json":
            data = [{"b64_json": await _image_url_to_base64(url)} for url in image_urls]
        else:
            data = [{"url": url} for url in image_urls]

        response_data = {
            "created": int(datetime.now().timestamp()),
            "data": data
        }

        duration_ms = (time.time() - start_time) * 1000
        debug_logger.log_response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=response_data,
            duration_ms=duration_ms,
            source="Client"
        )
        return JSONResponse(content=response_data)

    except ValueError as e:
        duration_ms = (time.time() - start_time) * 1000
        error_response = _build_error_response(str(e), error_type="invalid_request_error")
        debug_logger.log_response(
            status_code=400,
            headers={"Content-Type": "application/json"},
            body=error_response,
            duration_ms=duration_ms,
            source="Client"
        )
        return JSONResponse(status_code=400, content=error_response)
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_response = _build_error_response(str(e))
        debug_logger.log_error(
            error_message=str(e),
            status_code=500,
            response_text=str(e),
            source="Client"
        )
        debug_logger.log_response(
            status_code=500,
            headers={"Content-Type": "application/json"},
            body=error_response,
            duration_ms=duration_ms,
            source="Client"
        )
        return JSONResponse(status_code=500, content=error_response)

@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key_header),
    http_request: Request = None
):
    """Create chat completion (unified endpoint for image and video generation)"""
    start_time = time.time()

    try:
        # Log client request
        debug_logger.log_request(
            method="POST",
            url="/v1/chat/completions",
            headers=dict(http_request.headers) if http_request else {},
            body=request.dict(),
            source="Client"
        )

        # Extract prompt from messages
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")

        last_message = request.messages[-1]
        content = last_message.content

        # Handle both string and array format (OpenAI multimodal)
        prompt = ""
        image_data = request.image  # Default to request.image if provided
        video_data = request.video  # Video parameter
        remix_target_id = request.remix_target_id  # Remix target ID

        if isinstance(content, str):
            # Simple string format
            prompt = content
            # Extract remix_target_id from prompt if not already provided
            if not remix_target_id:
                remix_target_id = _extract_remix_id(prompt)
        elif isinstance(content, list):
            # Array format (OpenAI multimodal)
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        prompt = item.get("text", "")
                        # Extract remix_target_id from prompt if not already provided
                        if not remix_target_id:
                            remix_target_id = _extract_remix_id(prompt)
                    elif item.get("type") == "image_url":
                        # Extract base64 image from data URI
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "")
                        if url.startswith("data:image"):
                            # Extract base64 data from data URI
                            if "base64," in url:
                                image_data = url.split("base64,", 1)[1]
                            else:
                                image_data = url
                    elif item.get("type") == "video_url":
                        # Extract video from video_url
                        video_url = item.get("video_url", {})
                        url = video_url.get("url", "")
                        if url.startswith("data:video") or url.startswith("data:application"):
                            # Extract base64 data from data URI
                            if "base64," in url:
                                video_data = url.split("base64,", 1)[1]
                            else:
                                video_data = url
                        else:
                            # It's a URL, pass it as-is (will be downloaded in generation_handler)
                            video_data = url
        else:
            raise HTTPException(status_code=400, detail="Invalid content format")

        # Validate model
        if request.model not in MODEL_CONFIG:
            raise HTTPException(status_code=400, detail=f"Invalid model: {request.model}")

        # Check if this is a video model
        model_config = MODEL_CONFIG[request.model]
        is_video_model = model_config["type"] == "video"

        # For video models with video parameter, we need streaming
        if is_video_model and (video_data or remix_target_id):
            if not request.stream:
                # Non-streaming mode: only check availability
                result = None
                async for chunk in generation_handler.handle_generation_with_retry(
                    model=request.model,
                    prompt=prompt,
                    image=image_data,
                    video=video_data,
                    remix_target_id=remix_target_id,
                    stream=False
                ):
                    result = chunk

                if result:
                    duration_ms = (time.time() - start_time) * 1000
                    response_data = json.loads(result)
                    debug_logger.log_response(
                        status_code=200,
                        headers={"Content-Type": "application/json"},
                        body=response_data,
                        duration_ms=duration_ms,
                        source="Client"
                    )
                    return JSONResponse(content=response_data)
                else:
                    duration_ms = (time.time() - start_time) * 1000
                    error_response = {
                        "error": {
                            "message": "Availability check failed",
                            "type": "server_error",
                            "param": None,
                            "code": None
                        }
                    }
                    debug_logger.log_response(
                        status_code=500,
                        headers={"Content-Type": "application/json"},
                        body=error_response,
                        duration_ms=duration_ms,
                        source="Client"
                    )
                    return JSONResponse(
                        status_code=500,
                        content=error_response
                    )

        # Handle streaming
        if request.stream:
            async def generate():
                try:
                    async for chunk in generation_handler.handle_generation_with_retry(
                        model=request.model,
                        prompt=prompt,
                        image=image_data,
                        video=video_data,
                        remix_target_id=remix_target_id,
                        stream=True
                    ):
                        yield chunk
                except Exception as e:
                    # Try to parse structured error (JSON format)
                    error_data = None
                    try:
                        error_data = json.loads(str(e))
                    except:
                        pass

                    # Return OpenAI-compatible error format
                    if error_data and isinstance(error_data, dict) and "error" in error_data:
                        # Structured error (e.g., unsupported_country_code)
                        error_response = error_data
                    else:
                        # Generic error
                        error_response = {
                            "error": {
                                "message": str(e),
                                "type": "server_error",
                                "param": None,
                                "code": None
                            }
                        }
                    error_chunk = f'data: {json.dumps(error_response)}\n\n'
                    yield error_chunk
                    yield 'data: [DONE]\n\n'

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response (availability check only)
            result = None
            async for chunk in generation_handler.handle_generation_with_retry(
                model=request.model,
                prompt=prompt,
                image=image_data,
                video=video_data,
                remix_target_id=remix_target_id,
                stream=False
            ):
                result = chunk

            if result:
                duration_ms = (time.time() - start_time) * 1000
                response_data = json.loads(result)
                debug_logger.log_response(
                    status_code=200,
                    headers={"Content-Type": "application/json"},
                    body=response_data,
                    duration_ms=duration_ms,
                    source="Client"
                )
                return JSONResponse(content=response_data)
            else:
                # Return OpenAI-compatible error format
                duration_ms = (time.time() - start_time) * 1000
                error_response = {
                    "error": {
                        "message": "Availability check failed",
                        "type": "server_error",
                        "param": None,
                        "code": None
                    }
                }
                debug_logger.log_response(
                    status_code=500,
                    headers={"Content-Type": "application/json"},
                    body=error_response,
                    duration_ms=duration_ms,
                    source="Client"
                )
                return JSONResponse(
                    status_code=500,
                    content=error_response
                )

    except Exception as e:
        # Return OpenAI-compatible error format
        duration_ms = (time.time() - start_time) * 1000
        error_response = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": None
            }
        }
        debug_logger.log_error(
            error_message=str(e),
            status_code=500,
            response_text=str(e),
            source="Client"
        )
        debug_logger.log_response(
            status_code=500,
            headers={"Content-Type": "application/json"},
            body=error_response,
            duration_ms=duration_ms,
            source="Client"
        )
        return JSONResponse(
            status_code=500,
            content=error_response
        )
