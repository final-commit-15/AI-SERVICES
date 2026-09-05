from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from enum import Enum


class ToolType(str, Enum):
    FUNCTION = "function"
    BUILTIN = "builtin"


class ToolDefinition(BaseModel):
    type: ToolType = ToolType.FUNCTION
    function: "FunctionDefinition"


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    strict: bool = False


class ToolCall(BaseModel):
    id: str
    type: ToolType = ToolType.FUNCTION
    function: "FunctionCall"


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolResult(BaseModel):
    tool_call_id: str
    role: str = "tool"
    content: str
    name: Optional[str] = None


class ToolExecutionRequest(BaseModel):
    tool_calls: List[ToolCall]
    conversation_id: Optional[str] = None


class ToolExecutionResponse(BaseModel):
    results: List[ToolResult]
    errors: List[str] = []


class BuiltinTool(str, Enum):
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    HTTP_REQUEST = "http_request"
    CALCULATOR = "calculator"
    DATE_TIME = "date_time"