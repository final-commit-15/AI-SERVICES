import pytest
from unittest.mock import AsyncMock, MagicMock

from services.llm_gateway.src.tools.registry import ToolRegistry, Tool, register_builtin_tools
from libs.schemas_common.tools import ToolDefinition, FunctionDefinition, BuiltinTool


class TestToolRegistry:
    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    def test_register_tool(self, registry):
        tool_def = ToolDefinition(
            function=FunctionDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        )
        
        registry.register_tool(tool_def)
        
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "test_tool"

    def test_unregister_tool(self, registry):
        tool_def = ToolDefinition(
            function=FunctionDefinition(
                name="test_tool",
                description="A test tool",
                parameters={},
            )
        )
        registry.register_tool(tool_def)
        
        registry.unregister_tool("test_tool")
        
        tools = registry.list_tools()
        assert len(tools) == 0

    def test_list_tools(self, registry):
        tool_def1 = ToolDefinition(
            function=FunctionDefinition(name="tool1", description="Tool 1", parameters={})
        )
        tool_def2 = ToolDefinition(
            function=FunctionDefinition(name="tool2", description="Tool 2", parameters={})
        )
        registry.register_tool(tool_def1)
        registry.register_tool(tool_def2)
        
        tools = registry.list_tools()
        
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"tool1", "tool2"}

    def test_get_tool(self, registry):
        tool_def = ToolDefinition(
            function=FunctionDefinition(name="test_tool", description="Test", parameters={})
        )
        registry.register_tool(tool_def)
        
        tool = registry.get_tool("test_tool")
        
        assert tool is not None
        assert tool.name == "test_tool"

    def test_get_tool_not_found(self, registry):
        tool = registry.get_tool("nonexistent")
        assert tool is None

    @pytest.mark.asyncio
    async def test_execute_tool(self, registry):
        async def mock_func(x: int, y: int) -> int:
            return x + y
        
        registry.register_function(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
            function=mock_func,
        )
        
        result = await registry.execute("add", '{"x": 2, "y": 3}')
        
        assert result == 5

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, registry):
        with pytest.raises(ValueError):
            await registry.execute("nonexistent", "{}")

    @pytest.mark.asyncio
    async def test_execute_tool_error(self, registry):
        async def failing_func():
            raise ValueError("Test error")
        
        registry.register_function(
            name="fail",
            description="Fails",
            parameters={},
            function=failing_func,
        )
        
        with pytest.raises(ValueError):
            await registry.execute("fail", "{}")


class TestBuiltinTools:
    @pytest.mark.asyncio
    async def test_register_builtin_tools(self):
        registry = ToolRegistry()
        await register_builtin_tools(registry)
        
        tools = registry.list_tools()
        tool_names = {t["function"]["name"] for t in tools}
        
        assert "calculator" in tool_names
        assert "get_current_time" in tool_names
        assert "http_request" in tool_names

    @pytest.mark.asyncio
    async def test_calculator_tool(self):
        registry = ToolRegistry()
        await register_builtin_tools(registry)
        
        result = await registry.execute("calculator", '{"expression": "2 + 2"}')
        assert "4" in result
        
        result = await registry.execute("calculator", '{"expression": "10 * 5"}')
        assert "50" in result

    @pytest.mark.asyncio
    async def test_get_current_time_tool(self):
        registry = ToolRegistry()
        await register_builtin_tools(registry)
        
        result = await registry.execute("get_current_time", '{"timezone": "UTC"}')
        assert "T" in result  # ISO format

    @pytest.mark.asyncio
    async def test_http_request_tool(self):
        registry = ToolRegistry()
        await register_builtin_tools(registry)
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.text = '{"result": "ok"}'
            
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(return_value=mock_response)
            
            result = await registry.execute("http_request", '{"url": "https://httpbin.org/get"}')
            
            assert "status" in result
            assert result["status"] == 200