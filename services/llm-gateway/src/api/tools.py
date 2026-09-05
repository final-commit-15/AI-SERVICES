import structlog
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel

from libs.schemas_common.tools import ToolDefinition, ToolCall, ToolExecutionRequest, ToolExecutionResponse, ToolResult, BuiltinTool
from ..tools.registry import ToolRegistry
from ..dependencies import get_tool_registry

logger = structlog.get_logger()

router = APIRouter()


class ExecuteToolRequest(BaseModel):
    tool_calls: List[ToolCall]
    conversation_id: Optional[str] = None


@router.post("/tools/execute", response_model=ToolExecutionResponse)
async def execute_tools(
    request: ExecuteToolRequest,
    http_request: Request,
    tool_registry: ToolRegistry = Depends(get_tool_registry),
):
    """Execute tool calls."""
    try:
        results = []
        errors = []

        for tool_call in request.tool_calls:
            try:
                result = await tool_registry.execute(
                    tool_name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                    conversation_id=request.conversation_id,
                )
                results.append(ToolResult(
                    tool_call_id=tool_call.id,
                    content=str(result),
                    name=tool_call.function.name,
                ))
            except Exception as e:
                errors.append(f"Tool {tool_call.function.name}: {str(e)}")
                results.append(ToolResult(
                    tool_call_id=tool_call.id,
                    content=f"Error: {str(e)}",
                    name=tool_call.function.name,
                ))

        return ToolExecutionResponse(results=results, errors=errors)
    except Exception as e:
        logger.error("execute_tools_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools", response_model=List[ToolDefinition])
async def list_tools(
    http_request: Request,
    tool_registry: ToolRegistry = Depends(get_tool_registry),
):
    """List available tools."""
    return tool_registry.list_tools()


@router.get("/tools/builtin", response_model=List[str])
async def list_builtin_tools():
    """List builtin tool names."""
    return [t.value for t in BuiltinTool]


@router.post("/tools/register")
async def register_tool(
    tool: ToolDefinition,
    http_request: Request,
    tool_registry: ToolRegistry = Depends(get_tool_registry),
):
    """Register a custom tool."""
    try:
        tool_registry.register_tool(tool)
        return {"status": "registered", "tool": tool.function.name}
    except Exception as e:
        logger.error("register_tool_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tools/{tool_name}")
async def unregister_tool(
    tool_name: str,
    http_request: Request,
    tool_registry: ToolRegistry = Depends(get_tool_registry),
):
    """Unregister a tool."""
    try:
        tool_registry.unregister_tool(tool_name)
        return {"status": "unregistered", "tool": tool_name}
    except Exception as e:
        logger.error("unregister_tool_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))