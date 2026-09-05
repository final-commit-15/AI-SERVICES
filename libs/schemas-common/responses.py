from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class ResponsesRequest(BaseModel):
    model: str
    input: Union[str, List["ResponsesInputItem"]]
    instructions: Optional[str] = None
    tools: Optional[List["ToolDefinition"]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: bool = True
    truncation: str = "auto"
    max_output_tokens: Optional[int] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    store: bool = True
    metadata: Optional[Dict[str, Any]] = None
    user: Optional[str] = None


class ResponsesInputItem(BaseModel):
    type: str
    role: Optional[str] = None
    content: Optional[Union[str, List["ResponsesContent"]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    call_id: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None


class ResponsesContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional["ImageUrl"] = None
    file_id: Optional[str] = None


class ImageUrl(BaseModel):
    url: str
    detail: str = "auto"


class ResponsesResponse(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    status: str
    model: str
    output: List["ResponsesOutputItem"]
    usage: "ResponsesUsage"
    error: Optional["ResponsesError"] = None


class ResponsesOutputItem(BaseModel):
    id: str
    type: str
    role: Optional[str] = None
    content: Optional[List["ResponsesContent"]] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    call_id: Optional[str] = None
    output: Optional[str] = None
    status: Optional[str] = None


class ResponsesUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: Optional[Dict[str, int]] = None
    output_tokens_details: Optional[Dict[str, int]] = None


class ResponsesError(BaseModel):
    code: str
    message: str
    param: Optional[str] = None
    type: str


class ResponsesStreamEvent(BaseModel):
    type: str
    sequence_number: int
    item: Optional[ResponsesOutputItem] = None
    delta: Optional[Dict[str, Any]] = None
    snapshot: Optional[str] = None


from .tools import ToolDefinition