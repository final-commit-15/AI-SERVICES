import structlog
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable[..., Awaitable[Any]]
    strict: bool = False


class ToolRegistry:
    """Registry for managing tools/functions."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool_def) -> None:
        """Register a tool from definition."""
        from libs.schemas_common.tools import ToolDefinition, FunctionDefinition

        if isinstance(tool_def, ToolDefinition):
            func = tool_def.function
            # In production, would look up actual function implementation
            # For now, create a mock function
            async def mock_func(**kwargs):
                return f"Executed {func.name} with {kwargs}"

            tool = Tool(
                name=func.name,
                description=func.description,
                parameters=func.parameters,
                function=mock_func,
                strict=func.strict,
            )
            self._tools[func.name] = tool
            logger.info("tool_registered", name=func.name)
        else:
            raise ValueError("Invalid tool definition")

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            logger.info("tool_unregistered", name=name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }
            }
            for tool in self._tools.values()
        ]

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    async def execute(
        self,
        tool_name: str,
        arguments: str,
        conversation_id: Optional[str] = None,
    ) -> Any:
        """Execute a tool."""
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            result = await tool.function(**args)
            return result
        except Exception as e:
            logger.error("tool_execution_failed", tool=tool_name, error=str(e))
            raise

    def register_function(self, name: str, description: str, parameters: Dict[str, Any], function: Callable[..., Awaitable[Any]], strict: bool = False):
        """Register a tool directly with a function."""
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
            strict=strict,
        )
        self._tools[name] = tool
        logger.info("tool_registered", name=name)


# Builtin tools
async def register_builtin_tools(registry: ToolRegistry):
    """Register builtin tools."""

    # Calculator
    async def calculator(expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        try:
            # Only allow safe operations
            allowed_names = {
                k: v for k, v in __builtins__.items()
                if k in ['abs', 'min', 'max', 'pow', 'round', 'sum']
            }
            allowed_names.update({k: v for k, v in __import__('math').__dict__.items() if not k.startswith('_')})
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    registry.register_function(
        name="calculator",
        description="Evaluate a mathematical expression",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
            },
            "required": ["expression"],
        },
        function=calculator,
    )

    # Date/Time
    async def get_current_time(timezone: str = "UTC") -> str:
        """Get current date and time."""
        from datetime import datetime
        import pytz
        try:
            tz = pytz.timezone(timezone)
            now = datetime.now(tz)
        except:
            now = datetime.utcnow()
        return now.isoformat()

    registry.register_function(
        name="get_current_time",
        description="Get current date and time",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "Timezone (e.g., 'UTC', 'US/Eastern')"}
            },
        },
        function=get_current_time,
    )

    # HTTP Request
    async def http_request(
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request."""
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(method, url, headers=headers, content=body)
            return {
                "status": resp.status_code,
                "headers": dict(resp.headers),
                "body": resp.text,
            }

    registry.register_function(
        name="http_request",
        description="Make an HTTP request",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to request"},
                "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                "headers": {"type": "object", "description": "Request headers"},
                "body": {"type": "string", "description": "Request body"},
            },
            "required": ["url"],
        },
        function=http_request,
    )