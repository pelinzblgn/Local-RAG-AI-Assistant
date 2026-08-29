from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentIntent(str, Enum):
    """
    High-level actions that the local agent
    is allowed to perform.
    """

    KNOWLEDGE_QUERY = "knowledge_query"
    KNOWLEDGE_SYNC = "knowledge_sync"
    KNOWLEDGE_STATUS = "knowledge_status"
    CONVERSATION_CLEAR = "conversation_clear"
    UNKNOWN = "unknown"


class AgentStepStatus(str, Enum):
    """Execution state of one agent step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class AgentDecision:
    """
    Result produced by the agent decision layer.

    The decision describes what the agent intends
    to do before any tool is executed.
    """

    intent: AgentIntent
    tool_name: str | None
    reason: str
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Agent decision confidence must "
                "be between 0.0 and 1.0."
            )

        if not self.reason.strip():
            raise ValueError(
                "Agent decision reason cannot be empty."
            )

        if self.tool_name is not None:
            clean_tool_name = self.tool_name.strip()

            if not clean_tool_name:
                raise ValueError(
                    "Agent tool name cannot be empty."
                )


@dataclass(frozen=True)
class AgentStep:
    """
    One observable step performed during
    agent execution.
    """

    name: str
    status: AgentStepStatus
    detail: str
    tool_name: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "Agent step name cannot be empty."
            )

        if not self.detail.strip():
            raise ValueError(
                "Agent step detail cannot be empty."
            )


@dataclass
class ToolResult:
    """
    Normalized result returned by an agent tool.
    """

    success: bool
    content: str
    data: dict[str, Any] = field(
        default_factory=dict
    )
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.content,
            str,
        ):
            raise TypeError(
                "Tool result content must be a string."
            )

        if self.success and self.error is not None:
            raise ValueError(
                "Successful tool result cannot "
                "contain an error."
            )

        if not self.success and not self.error:
            raise ValueError(
                "Failed tool result must contain "
                "an error message."
            )


@dataclass
class AgentResult:
    """
    Final result returned by the agent layer.

    It contains both the user-facing answer and
    an execution trace for observability.
    """

    answer: str
    decision: AgentDecision
    steps: list[AgentStep]
    tool_result: ToolResult | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.answer,
            str,
        ):
            raise TypeError(
                "Agent answer must be a string."
            )

        if not self.answer.strip():
            raise ValueError(
                "Agent answer cannot be empty."
            )

    @property
    def succeeded(self) -> bool:
        """
        Return whether agent execution completed
        without a failed step or tool result.
        """

        if any(
            step.status == AgentStepStatus.FAILED
            for step in self.steps
        ):
            return False

        if (
            self.tool_result is not None
            and not self.tool_result.success
        ):
            return False

        return True