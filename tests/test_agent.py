from unittest.mock import MagicMock

import pytest

from src.agent import LocalAgent
from src.agent_decision import AgentDecisionEngine
from src.agent_models import (
    AgentDecision,
    AgentIntent,
    AgentStepStatus,
    ToolResult,
)
from src.agent_tools import (
    AgentTool,
    ToolContext,
    ToolRegistry,
)


class SuccessfulTool(AgentTool):
    """Deterministic successful test tool."""

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return "Return a successful knowledge result."

    def execute(
        self,
        context: ToolContext,
    ) -> ToolResult:
        return ToolResult(
            success=True,
            content=(
                f"Answer for: {context.user_input}"
            ),
            data={
                "sources": [
                    "test.txt",
                ],
            },
        )


class FailingTool(AgentTool):
    """Deterministic controlled-failure tool."""

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return "Return a controlled tool failure."

    def execute(
        self,
        context: ToolContext,
    ) -> ToolResult:
        del context

        return ToolResult(
            success=False,
            content=(
                "Knowledge search could not be completed."
            ),
            error=(
                "Controlled retrieval failure."
            ),
        )


def test_local_agent_requires_registry() -> None:
    with pytest.raises(
        TypeError,
        match="ToolRegistry",
    ):
        LocalAgent(
            registry=object(),  # type: ignore[arg-type]
        )


def test_local_agent_rejects_invalid_decision_engine() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        TypeError,
        match="AgentDecisionEngine",
    ):
        LocalAgent(
            registry=registry,
            decision_engine=object(),  # type: ignore[arg-type]
        )


def test_local_agent_creates_default_decision_engine() -> None:
    registry = ToolRegistry(
        tools=[
            SuccessfulTool(),
        ]
    )

    agent = LocalAgent(
        registry=registry
    )

    assert isinstance(
        agent.decision_engine,
        AgentDecisionEngine,
    )

    assert agent.registry is registry


def test_agent_rejects_empty_input() -> None:
    agent = LocalAgent(
        registry=ToolRegistry()
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        agent.run(
            "   "
        )


def test_agent_rejects_non_string_input() -> None:
    agent = LocalAgent(
        registry=ToolRegistry()
    )

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        agent.run(
            123  # type: ignore[arg-type]
        )


def test_agent_rejects_invalid_metadata() -> None:
    agent = LocalAgent(
        registry=ToolRegistry()
    )

    with pytest.raises(
        TypeError,
        match="dictionary",
    ):
        agent.run(
            "STM32 nedir?",
            metadata="invalid",  # type: ignore[arg-type]
        )


def test_agent_executes_selected_tool() -> None:
    registry = ToolRegistry(
        tools=[
            SuccessfulTool(),
        ]
    )

    agent = LocalAgent(
        registry=registry
    )

    result = agent.run(
        "STM32 nedir?"
    )

    assert result.succeeded is True

    assert result.answer == (
        "Answer for: STM32 nedir?"
    )

    assert (
        result.decision.intent
        == AgentIntent.KNOWLEDGE_QUERY
    )

    assert (
        result.decision.tool_name
        == "knowledge_search"
    )

    assert result.tool_result is not None
    assert result.tool_result.success is True

    assert result.tool_result.data[
        "sources"
    ] == [
        "test.txt",
    ]


def test_successful_agent_execution_has_complete_trace() -> None:
    agent = LocalAgent(
        registry=ToolRegistry(
            tools=[
                SuccessfulTool(),
            ]
        )
    )

    result = agent.run(
        "PID nedir?"
    )

    assert [
        step.name
        for step in result.steps
    ] == [
        "Intent analysis",
        "Tool selection",
        "Tool execution",
        "Response assembly",
    ]

    assert all(
        step.status
        == AgentStepStatus.COMPLETED
        for step in result.steps
    )


def test_agent_passes_clean_input_to_tool() -> None:
    tool = MagicMock(
        spec=AgentTool
    )

    type(tool).name = property(
        lambda self: "knowledge_search"
    )

    type(tool).description = property(
        lambda self: "Test knowledge tool."
    )

    tool.execute.return_value = ToolResult(
        success=True,
        content="Completed.",
    )

    registry = ToolRegistry(
        tools=[
            tool,
        ]
    )

    agent = LocalAgent(
        registry=registry
    )

    agent.run(
        "   STM32 nedir?   "
    )

    tool.execute.assert_called_once()

    context = (
        tool.execute.call_args.args[0]
    )

    assert isinstance(
        context,
        ToolContext,
    )

    assert (
        context.user_input
        == "STM32 nedir?"
    )


def test_agent_passes_request_metadata_to_tool() -> None:
    tool = MagicMock(
        spec=AgentTool
    )

    type(tool).name = property(
        lambda self: "knowledge_search"
    )

    type(tool).description = property(
        lambda self: "Test knowledge tool."
    )

    tool.execute.return_value = ToolResult(
        success=True,
        content="Completed.",
    )

    agent = LocalAgent(
        registry=ToolRegistry(
            tools=[
                tool,
            ]
        )
    )

    agent.run(
        "STM32 nedir?",
        metadata={
            "request_id": "abc-123",
        },
    )

    context = (
        tool.execute.call_args.args[0]
    )

    assert context.metadata == {
        "request_id": "abc-123",
    }


def test_agent_preserves_trusted_metadata() -> None:
    agent = LocalAgent(
        registry=ToolRegistry(
            tools=[
                SuccessfulTool(),
            ]
        )
    )

    result = agent.run(
        "STM32 nedir?",
        metadata={
            "local_only": False,
            "selected_tool": "malicious",
            "execution_ms": -1,
        },
    )

    assert (
        result.metadata[
            "local_only"
        ]
        is True
    )

    assert (
        result.metadata[
            "selected_tool"
        ]
        == "knowledge_search"
    )

    assert (
        result.metadata[
            "execution_ms"
        ]
        >= 0.0
    )

    assert result.metadata[
        "request_metadata"
    ] == {
        "local_only": False,
        "selected_tool": "malicious",
        "execution_ms": -1,
    }


def test_agent_returns_unknown_result_without_tools() -> None:
    agent = LocalAgent(
        registry=ToolRegistry()
    )

    result = agent.run(
        "STM32 nedir?"
    )

    assert result.succeeded is True

    assert (
        result.decision.intent
        == AgentIntent.UNKNOWN
    )

    assert result.decision.tool_name is None

    assert result.tool_result is None

    assert len(
        result.steps
    ) == 2

    assert (
        result.steps[1].status
        == AgentStepStatus.SKIPPED
    )

    assert (
        result.metadata[
            "selected_tool"
        ]
        is None
    )


def test_agent_handles_controlled_tool_failure() -> None:
    agent = LocalAgent(
        registry=ToolRegistry(
            tools=[
                FailingTool(),
            ]
        )
    )

    result = agent.run(
        "STM32 nedir?"
    )

    assert result.succeeded is False

    assert result.tool_result is not None

    assert (
        result.tool_result.success
        is False
    )

    assert result.answer == (
        "Knowledge search could not be completed."
    )

    assert (
        result.steps[-1].status
        == AgentStepStatus.FAILED
    )

    assert (
        result.steps[-1].name
        == "Tool execution"
    )


def test_agent_handles_tool_exception_through_registry() -> None:
    class ExplodingTool(AgentTool):
        @property
        def name(self) -> str:
            return "knowledge_search"

        @property
        def description(self) -> str:
            return "Raise an unexpected exception."

        def execute(
            self,
            context: ToolContext,
        ) -> ToolResult:
            del context

            raise RuntimeError(
                "Unexpected failure."
            )

    agent = LocalAgent(
        registry=ToolRegistry(
            tools=[
                ExplodingTool(),
            ]
        )
    )

    result = agent.run(
        "STM32 nedir?"
    )

    assert result.succeeded is False

    assert result.tool_result is not None
    assert result.tool_result.error is not None

    assert (
        "RuntimeError"
        in result.tool_result.error
    )


def test_agent_handles_selected_tool_disappearing_from_registry() -> None:
    registry = ToolRegistry(
        tools=[
            SuccessfulTool(),
        ]
    )

    engine = AgentDecisionEngine(
        registry
    )

    agent = LocalAgent(
        registry=registry,
        decision_engine=engine,
    )

    registry.unregister(
        "knowledge_search"
    )

    decision = AgentDecision(
        intent=AgentIntent.KNOWLEDGE_QUERY,
        tool_name="knowledge_search",
        reason="Forced test decision.",
        confidence=1.0,
    )

    engine.decide = MagicMock(
        return_value=decision
    )

    result = agent.run(
        "STM32 nedir?"
    )

    assert result.succeeded is False

    assert result.tool_result is not None

    assert (
        result.steps[-1].status
        == AgentStepStatus.FAILED
    )

    assert (
        result.metadata[
            "selected_tool"
        ]
        == "knowledge_search"
    )


def test_agent_uses_injected_decision_engine() -> None:
    registry = ToolRegistry(
        tools=[
            SuccessfulTool(),
        ]
    )

    engine = AgentDecisionEngine(
        registry
    )

    engine.decide = MagicMock(
        return_value=AgentDecision(
            intent=AgentIntent.KNOWLEDGE_QUERY,
            tool_name="knowledge_search",
            reason="Injected decision.",
            confidence=0.97,
        )
    )

    agent = LocalAgent(
        registry=registry,
        decision_engine=engine,
    )

    result = agent.run(
        "Custom request"
    )

    engine.decide.assert_called_once_with(
        "Custom request"
    )

    assert (
        result.decision.confidence
        == 0.97
    )


def test_agent_records_execution_time() -> None:
    agent = LocalAgent(
        registry=ToolRegistry(
            tools=[
                SuccessfulTool(),
            ]
        )
    )

    result = agent.run(
        "STM32 nedir?"
    )

    execution_ms = result.metadata[
        "execution_ms"
    ]

    assert isinstance(
        execution_ms,
        float,
    )

    assert execution_ms >= 0.0


def test_agent_result_keeps_decision_explanation() -> None:
    agent = LocalAgent(
        registry=ToolRegistry(
            tools=[
                SuccessfulTool(),
            ]
        )
    )

    result = agent.run(
        "STM32 nedir?"
    )

    assert result.decision.reason.strip()

    assert (
        result.decision.confidence
        > 0.0
    )