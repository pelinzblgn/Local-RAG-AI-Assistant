import pytest

from src.agent_models import (
    ToolResult,
)
from src.agent_tools import (
    AgentTool,
    DuplicateToolError,
    InvalidToolError,
    ToolContext,
    ToolNotFoundError,
    ToolRegistry,
)


class EchoTool(AgentTool):
    """Simple deterministic tool used by registry tests."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Return the supplied user input."

    def execute(
        self,
        context: ToolContext,
    ) -> ToolResult:
        return ToolResult(
            success=True,
            content=context.user_input,
            data={
                "echoed": True,
            },
        )


class FailingTool(AgentTool):
    """Tool that raises an exception during execution."""

    @property
    def name(self) -> str:
        return "failing_tool"

    @property
    def description(self) -> str:
        return "Raise a controlled execution failure."

    def execute(
        self,
        context: ToolContext,
    ) -> ToolResult:
        del context

        raise RuntimeError(
            "Simulated tool failure."
        )


class InvalidResultTool(AgentTool):
    """Tool that violates the AgentTool result contract."""

    @property
    def name(self) -> str:
        return "invalid_result"

    @property
    def description(self) -> str:
        return "Return an invalid result for contract testing."

    def execute(
        self,
        context: ToolContext,
    ) -> ToolResult:
        del context

        return "invalid"  # type: ignore[return-value]


class AnotherEchoTool(AgentTool):
    """Tool with a duplicate normalized name."""

    @property
    def name(self) -> str:
        return " ECHO "

    @property
    def description(self) -> str:
        return "Duplicate normalized echo tool."

    def execute(
        self,
        context: ToolContext,
    ) -> ToolResult:
        return ToolResult(
            success=True,
            content=context.user_input,
        )


def test_tool_context_accepts_valid_input() -> None:
    context = ToolContext(
        user_input="STM32 nedir?",
        metadata={
            "session_id": "test-session",
        },
    )

    assert context.user_input == "STM32 nedir?"
    assert context.metadata[
        "session_id"
    ] == "test-session"


def test_tool_context_rejects_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        ToolContext(
            user_input="   ",
        )


def test_tool_context_rejects_non_string_input() -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        ToolContext(
            user_input=123,  # type: ignore[arg-type]
        )


def test_registry_starts_empty() -> None:
    registry = ToolRegistry()

    assert registry.size == 0
    assert registry.is_empty is True
    assert registry.list_tools() == []


def test_registry_can_be_initialized_with_tools() -> None:
    registry = ToolRegistry(
        tools=[
            EchoTool(),
        ]
    )

    assert registry.size == 1
    assert registry.is_empty is False
    assert registry.contains(
        "echo"
    ) is True


def test_registry_registers_tool() -> None:
    registry = ToolRegistry()

    tool = EchoTool()

    registry.register(
        tool
    )

    assert registry.size == 1
    assert registry.get(
        "echo"
    ) is tool


def test_registry_normalizes_tool_lookup() -> None:
    registry = ToolRegistry(
        tools=[
            EchoTool(),
        ]
    )

    assert registry.contains(
        " ECHO "
    ) is True

    tool = registry.get(
        "Echo"
    )

    assert tool.name == "echo"


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = ToolRegistry(
        tools=[
            EchoTool(),
        ]
    )

    with pytest.raises(
        DuplicateToolError,
        match="already registered",
    ):
        registry.register(
            AnotherEchoTool()
        )


def test_registry_rejects_non_tool_object() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        InvalidToolError,
        match="AgentTool",
    ):
        registry.register(
            object()  # type: ignore[arg-type]
        )


def test_registry_get_rejects_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolNotFoundError,
        match="not registered",
    ):
        registry.get(
            "missing"
        )


def test_registry_unregister_removes_tool() -> None:
    registry = ToolRegistry(
        tools=[
            EchoTool(),
        ]
    )

    removed = registry.unregister(
        "echo"
    )

    assert removed.name == "echo"
    assert registry.size == 0
    assert registry.is_empty is True


def test_registry_unregister_rejects_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolNotFoundError,
        match="not registered",
    ):
        registry.unregister(
            "missing"
        )


def test_registry_describes_tools() -> None:
    registry = ToolRegistry(
        tools=[
            EchoTool(),
        ]
    )

    descriptions = registry.describe_tools()

    assert descriptions == [
        {
            "name": "echo",
            "description": (
                "Return the supplied user input."
            ),
        }
    ]


def test_registry_list_tools_returns_copy() -> None:
    registry = ToolRegistry(
        tools=[
            EchoTool(),
        ]
    )

    tools = registry.list_tools()

    tools.clear()

    assert registry.size == 1
    assert registry.contains(
        "echo"
    ) is True


def test_registry_executes_registered_tool() -> None:
    registry = ToolRegistry(
        tools=[
            EchoTool(),
        ]
    )

    result = registry.execute(
        name="echo",
        context=ToolContext(
            user_input="PID nedir?"
        ),
    )

    assert result.success is True
    assert result.content == "PID nedir?"
    assert result.data == {
        "echoed": True,
    }


def test_registry_execute_rejects_invalid_context() -> None:
    registry = ToolRegistry(
        tools=[
            EchoTool(),
        ]
    )

    with pytest.raises(
        TypeError,
        match="ToolContext",
    ):
        registry.execute(
            name="echo",
            context="invalid",  # type: ignore[arg-type]
        )


def test_registry_execute_rejects_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolNotFoundError,
    ):
        registry.execute(
            name="missing",
            context=ToolContext(
                user_input="Test"
            ),
        )


def test_registry_converts_tool_exception_to_failed_result() -> None:
    registry = ToolRegistry(
        tools=[
            FailingTool(),
        ]
    )

    result = registry.execute(
        name="failing_tool",
        context=ToolContext(
            user_input="Run"
        ),
    )

    assert result.success is False

    assert result.content == (
        "Tool execution failed."
    )

    assert result.error is not None

    assert (
        "RuntimeError"
        in result.error
    )

    assert (
        "Simulated tool failure"
        in result.error
    )


def test_registry_converts_invalid_tool_result_to_failure() -> None:
    registry = ToolRegistry(
        tools=[
            InvalidResultTool(),
        ]
    )

    result = registry.execute(
        name="invalid_result",
        context=ToolContext(
            user_input="Run"
        ),
    )

    assert result.success is False

    assert result.content == (
        "Tool returned an invalid result."
    )

    assert result.error is not None

    assert (
        "Tool contract violation"
        in result.error
    )


def test_registry_rejects_empty_lookup_name() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        registry.contains(
            "   "
        )


def test_registry_rejects_non_string_lookup_name() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        registry.get(
            123  # type: ignore[arg-type]
        )