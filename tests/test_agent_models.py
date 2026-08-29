import pytest

from src.agent_models import (
    AgentDecision,
    AgentIntent,
    AgentResult,
    AgentStep,
    AgentStepStatus,
    ToolResult,
)


def test_agent_decision_accepts_valid_values() -> None:
    decision = AgentDecision(
        intent=AgentIntent.KNOWLEDGE_QUERY,
        tool_name="knowledge_search",
        reason="The user is asking about local knowledge.",
        confidence=0.95,
    )

    assert decision.intent == AgentIntent.KNOWLEDGE_QUERY
    assert decision.tool_name == "knowledge_search"
    assert decision.confidence == 0.95


@pytest.mark.parametrize(
    "confidence",
    [
        -0.01,
        1.01,
        -10.0,
        10.0,
    ],
)
def test_agent_decision_rejects_invalid_confidence(
    confidence: float,
) -> None:
    with pytest.raises(ValueError):
        AgentDecision(
            intent=AgentIntent.KNOWLEDGE_QUERY,
            tool_name="knowledge_search",
            reason="Valid reason.",
            confidence=confidence,
        )


def test_agent_decision_rejects_empty_reason() -> None:
    with pytest.raises(ValueError):
        AgentDecision(
            intent=AgentIntent.UNKNOWN,
            tool_name=None,
            reason="   ",
            confidence=0.2,
        )


def test_agent_decision_rejects_empty_tool_name() -> None:
    with pytest.raises(ValueError):
        AgentDecision(
            intent=AgentIntent.KNOWLEDGE_QUERY,
            tool_name="   ",
            reason="Knowledge lookup required.",
            confidence=0.9,
        )


def test_agent_step_accepts_valid_values() -> None:
    step = AgentStep(
        name="Knowledge retrieval",
        status=AgentStepStatus.COMPLETED,
        detail="Relevant local documents were retrieved.",
        tool_name="knowledge_search",
    )

    assert step.status == AgentStepStatus.COMPLETED
    assert step.tool_name == "knowledge_search"


def test_agent_step_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        AgentStep(
            name=" ",
            status=AgentStepStatus.PENDING,
            detail="Waiting for execution.",
        )


def test_agent_step_rejects_empty_detail() -> None:
    with pytest.raises(ValueError):
        AgentStep(
            name="Intent analysis",
            status=AgentStepStatus.COMPLETED,
            detail=" ",
        )


def test_successful_tool_result() -> None:
    result = ToolResult(
        success=True,
        content="Knowledge retrieved.",
        data={
            "sources": [
                "stm32_notes.txt",
            ],
        },
    )

    assert result.success is True
    assert result.error is None
    assert result.data["sources"] == [
        "stm32_notes.txt",
    ]


def test_successful_tool_result_rejects_error() -> None:
    with pytest.raises(ValueError):
        ToolResult(
            success=True,
            content="Completed.",
            error="Unexpected error.",
        )


def test_failed_tool_result_requires_error() -> None:
    with pytest.raises(ValueError):
        ToolResult(
            success=False,
            content="Execution failed.",
        )


def test_tool_result_content_must_be_string() -> None:
    with pytest.raises(TypeError):
        ToolResult(
            success=True,
            content=123,  # type: ignore[arg-type]
        )


def test_successful_agent_result() -> None:
    decision = AgentDecision(
        intent=AgentIntent.KNOWLEDGE_QUERY,
        tool_name="knowledge_search",
        reason="Knowledge retrieval is required.",
        confidence=0.98,
    )

    tool_result = ToolResult(
        success=True,
        content="STM32 is a microcontroller family.",
    )

    result = AgentResult(
        answer="STM32 is a microcontroller family.",
        decision=decision,
        steps=[
            AgentStep(
                name="Intent analysis",
                status=AgentStepStatus.COMPLETED,
                detail="Knowledge query detected.",
            ),
            AgentStep(
                name="Knowledge retrieval",
                status=AgentStepStatus.COMPLETED,
                detail="Relevant evidence retrieved.",
                tool_name="knowledge_search",
            ),
        ],
        tool_result=tool_result,
    )

    assert result.succeeded is True
    assert result.decision is decision
    assert len(result.steps) == 2


def test_agent_result_fails_when_step_failed() -> None:
    decision = AgentDecision(
        intent=AgentIntent.KNOWLEDGE_QUERY,
        tool_name="knowledge_search",
        reason="Knowledge retrieval is required.",
        confidence=0.9,
    )

    result = AgentResult(
        answer="The operation could not be completed.",
        decision=decision,
        steps=[
            AgentStep(
                name="Knowledge retrieval",
                status=AgentStepStatus.FAILED,
                detail="Retrieval failed.",
                tool_name="knowledge_search",
            ),
        ],
    )

    assert result.succeeded is False


def test_agent_result_fails_when_tool_failed() -> None:
    decision = AgentDecision(
        intent=AgentIntent.KNOWLEDGE_SYNC,
        tool_name="knowledge_sync",
        reason="The knowledge base must be synchronized.",
        confidence=1.0,
    )

    result = AgentResult(
        answer="Knowledge base synchronization failed.",
        decision=decision,
        steps=[
            AgentStep(
                name="Knowledge synchronization",
                status=AgentStepStatus.FAILED,
                detail="Synchronization tool failed.",
                tool_name="knowledge_sync",
            ),
        ],
        tool_result=ToolResult(
            success=False,
            content="Synchronization failed.",
            error="Sync operation failed.",
        ),
    )

    assert result.succeeded is False


def test_agent_result_rejects_empty_answer() -> None:
    decision = AgentDecision(
        intent=AgentIntent.UNKNOWN,
        tool_name=None,
        reason="No supported intent detected.",
        confidence=0.1,
    )

    with pytest.raises(ValueError):
        AgentResult(
            answer="   ",
            decision=decision,
            steps=[],
        )


def test_agent_result_preserves_metadata() -> None:
    decision = AgentDecision(
        intent=AgentIntent.KNOWLEDGE_QUERY,
        tool_name="knowledge_search",
        reason="Knowledge retrieval is required.",
        confidence=0.95,
    )

    result = AgentResult(
        answer="Completed.",
        decision=decision,
        steps=[],
        metadata={
            "execution_ms": 42.5,
            "local_only": True,
        },
    )

    assert result.metadata["execution_ms"] == 42.5
    assert result.metadata["local_only"] is True