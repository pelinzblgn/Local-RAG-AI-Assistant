from __future__ import annotations

from time import perf_counter
from typing import Any

from src.agent_decision import AgentDecisionEngine
from src.agent_models import (
    AgentIntent,
    AgentResult,
    AgentStep,
    AgentStepStatus,
    ToolResult,
)
from src.agent_tools import (
    ToolContext,
    ToolNotFoundError,
    ToolRegistry,
)


class LocalAgent:
    """
    Orchestrates local agent execution.

    The agent itself does not implement retrieval,
    synchronization or database operations. It decides
    which registered capability should handle the request
    and coordinates its execution.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        decision_engine: AgentDecisionEngine | None = None,
    ) -> None:
        if not isinstance(
            registry,
            ToolRegistry,
        ):
            raise TypeError(
                "registry must be a ToolRegistry."
            )

        if (
            decision_engine is not None
            and not isinstance(
                decision_engine,
                AgentDecisionEngine,
            )
        ):
            raise TypeError(
                "decision_engine must be an "
                "AgentDecisionEngine."
            )

        self._registry = registry

        self._decision_engine = (
            decision_engine
            if decision_engine is not None
            else AgentDecisionEngine(
                registry
            )
        )

    @property
    def registry(self) -> ToolRegistry:
        """Return the active tool registry."""

        return self._registry

    @property
    def decision_engine(
        self,
    ) -> AgentDecisionEngine:
        """Return the active decision engine."""

        return self._decision_engine

    def run(
        self,
        user_input: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Process one user request.

        Execution stages:
        1. Analyze intent.
        2. Select an allowed tool.
        3. Execute the tool.
        4. Produce an observable AgentResult.
        """

        if not isinstance(
            user_input,
            str,
        ):
            raise TypeError(
                "Agent input must be a string."
            )

        clean_input = user_input.strip()

        if not clean_input:
            raise ValueError(
                "Agent input cannot be empty."
            )

        if (
            metadata is not None
            and not isinstance(
                metadata,
                dict,
            )
        ):
            raise TypeError(
                "Agent metadata must be a dictionary."
            )

        started_at = perf_counter()

        steps: list[AgentStep] = []

        decision = self._decision_engine.decide(
            clean_input
        )

        steps.append(
            AgentStep(
                name="Intent analysis",
                status=AgentStepStatus.COMPLETED,
                detail=(
                    f"Detected intent: "
                    f"{decision.intent.value}. "
                    f"{decision.reason}"
                ),
                tool_name=decision.tool_name,
            )
        )

        if (
            decision.intent == AgentIntent.UNKNOWN
            or decision.tool_name is None
        ):
            steps.append(
                AgentStep(
                    name="Tool selection",
                    status=AgentStepStatus.SKIPPED,
                    detail=(
                        "No supported registered tool "
                        "was selected."
                    ),
                )
            )

            return AgentResult(
                answer=(
                    "Bu istek için kullanılabilecek "
                    "desteklenen bir yerel araç bulunamadı."
                ),
                decision=decision,
                steps=steps,
                tool_result=None,
                metadata=self._build_metadata(
                    started_at=started_at,
                    request_metadata=metadata,
                    tool_name=None,
                ),
            )

        steps.append(
            AgentStep(
                name="Tool selection",
                status=AgentStepStatus.COMPLETED,
                detail=(
                    f"Selected registered tool: "
                    f"{decision.tool_name}."
                ),
                tool_name=decision.tool_name,
            )
        )

        context = ToolContext(
            user_input=clean_input,
            metadata=dict(
                metadata or {}
            ),
        )

        try:
            tool_result = self._registry.execute(
                name=decision.tool_name,
                context=context,
            )

        except ToolNotFoundError:
            failed_result = ToolResult(
                success=False,
                content=(
                    "The selected capability "
                    "is unavailable."
                ),
                error=(
                    "The decision engine selected "
                    "a tool that is not registered."
                ),
            )

            steps.append(
                AgentStep(
                    name="Tool execution",
                    status=AgentStepStatus.FAILED,
                    detail=(
                        "The selected tool was not "
                        "available in the registry."
                    ),
                    tool_name=decision.tool_name,
                )
            )

            return AgentResult(
                answer=(
                    "Seçilen yerel yetenek şu anda "
                    "kullanılamıyor."
                ),
                decision=decision,
                steps=steps,
                tool_result=failed_result,
                metadata=self._build_metadata(
                    started_at=started_at,
                    request_metadata=metadata,
                    tool_name=decision.tool_name,
                ),
            )

        if not tool_result.success:
            steps.append(
                AgentStep(
                    name="Tool execution",
                    status=AgentStepStatus.FAILED,
                    detail=(
                        "The selected tool returned "
                        "a controlled failure."
                    ),
                    tool_name=decision.tool_name,
                )
            )

            return AgentResult(
                answer=tool_result.content,
                decision=decision,
                steps=steps,
                tool_result=tool_result,
                metadata=self._build_metadata(
                    started_at=started_at,
                    request_metadata=metadata,
                    tool_name=decision.tool_name,
                ),
            )

        steps.append(
            AgentStep(
                name="Tool execution",
                status=AgentStepStatus.COMPLETED,
                detail=(
                    "The selected tool completed "
                    "successfully."
                ),
                tool_name=decision.tool_name,
            )
        )

        steps.append(
            AgentStep(
                name="Response assembly",
                status=AgentStepStatus.COMPLETED,
                detail=(
                    "The tool result was converted "
                    "into the final agent response."
                ),
                tool_name=decision.tool_name,
            )
        )

        return AgentResult(
            answer=tool_result.content,
            decision=decision,
            steps=steps,
            tool_result=tool_result,
            metadata=self._build_metadata(
                started_at=started_at,
                request_metadata=metadata,
                tool_name=decision.tool_name,
            ),
        )

    @staticmethod
    def _build_metadata(
        started_at: float,
        request_metadata: dict[str, Any] | None,
        tool_name: str | None,
    ) -> dict[str, Any]:
        """
        Build safe execution metadata for diagnostics.

        Request metadata is nested rather than merged into
        system metadata so callers cannot overwrite trusted
        execution fields.
        """

        elapsed_ms = (
            perf_counter() - started_at
        ) * 1000.0

        return {
            "execution_ms": elapsed_ms,
            "local_only": True,
            "selected_tool": tool_name,
            "request_metadata": dict(
                request_metadata or {}
            ),
        }