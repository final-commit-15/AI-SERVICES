from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal, Union
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class ChatMessage(BaseModel):
    role: MessageRole
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List["ToolCall"]] = None
    tool_call_id: Optional[str] = None
    images: Optional[List[str]] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    provider: Optional[str] = None
    task_type: Optional[str] = "general"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List["ToolDefinition"]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    provider: str
    choices: List["ChatChoice"]
    usage: "UsageInfo"
    system_fingerprint: Optional[str] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatStreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    provider: str
    choices: List["ChatStreamChoice"]


class ChatStreamChoice(BaseModel):
    index: int
    delta: "ChatDelta"
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatDelta(BaseModel):
    role: Optional[MessageRole] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Dict[str, int]] = None
    completion_tokens_details: Optional[Dict[str, int]] = None


class TaskType(str, Enum):
    GENERAL = "general"
    CHAT = "chat"
    CODING = "coding"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    VISION = "vision"
    EMBEDDING = "embedding"
    SPEECH = "speech"
    IMAGE_GEN = "image_generation"


ChatRequest.model_rebuild()
ChatResponse.model_rebuild()
ChatStreamChunk.model_rebuild()