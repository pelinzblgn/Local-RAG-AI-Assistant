from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from src.agent_models import ToolResult


@dataclass(frozen=True)
class ToolContext:
    """
    Runtime information supplied to an agent tool.

    Context is deliberately small so tools do not
    receive unrestricted application access.
    """

    user_input: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.user_input,
            str,
        ):
            raise TypeError(
                "Tool context user_input must be a string."
            )

        if not self.user_input.strip():
            raise ValueError(
                "Tool context user_input cannot be empty."
            )


class AgentTool(ABC):
    """
    Base contract for every tool available
    to the local agent.

    Agent tools must expose a stable name,
    description and execute method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique tool identifier."""

        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable tool description."""

        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool."""

        raise NotImplementedError


class ToolNotFoundError(LookupError):
    """Raised when an unknown tool is requested."""


class DuplicateToolError(ValueError):
    """Raised when a tool name is registered twice."""


class InvalidToolError(ValueError):
    """Raised when a tool violates the tool contract."""


class ToolRegistry:
    """
    Explicit allow-list of tools available to the agent.

    A tool must be registered here before the agent can
    discover or execute it.
    """

    def __init__(
        self,
        tools: Iterable[AgentTool] | None = None,
    ) -> None:
        self._tools: dict[str, AgentTool] = {}

        if tools is not None:
            for tool in tools:
                self.register(tool)

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:
        if not isinstance(
            name,
            str,
        ):
            raise TypeError(
                "Tool name must be a string."
            )

        normalized = name.strip().lower()

        if not normalized:
            raise ValueError(
                "Tool name cannot be empty."
            )

        return normalized

    @staticmethod
    def _validate_tool(
        tool: AgentTool,
    ) -> tuple[str, str]:
        if not isinstance(
            tool,
            AgentTool,
        ):
            raise InvalidToolError(
                "Registered object must implement AgentTool."
            )

        try:
            raw_name = tool.name
            raw_description = tool.description

        except Exception as error:
            raise InvalidToolError(
                "Tool metadata could not be read."
            ) from error

        if not isinstance(
            raw_name,
            str,
        ):
            raise InvalidToolError(
                "Tool name must be a string."
            )

        if not isinstance(
            raw_description,
            str,
        ):
            raise InvalidToolError(
                "Tool description must be a string."
            )

        name = raw_name.strip()
        description = raw_description.strip()

        if not name:
            raise InvalidToolError(
                "Tool name cannot be empty."
            )

        if not description:
            raise InvalidToolError(
                "Tool description cannot be empty."
            )

        return name, description

    @property
    def size(self) -> int:
        """Return number of registered tools."""

        return len(
            self._tools
        )

    @property
    def is_empty(self) -> bool:
        """Return whether no tools are registered."""

        return not self._tools

    def register(
        self,
        tool: AgentTool,
    ) -> None:
        """Register one tool."""

        name, _ = self._validate_tool(
            tool
        )

        normalized_name = self._normalize_name(
            name
        )

        if normalized_name in self._tools:
            raise DuplicateToolError(
                f"Tool '{name}' is already registered."
            )

        self._tools[
            normalized_name
        ] = tool

    def unregister(
        self,
        name: str,
    ) -> AgentTool:
        """Remove and return one registered tool."""

        normalized_name = self._normalize_name(
            name
        )

        try:
            return self._tools.pop(
                normalized_name
            )

        except KeyError as error:
            raise ToolNotFoundError(
                f"Tool '{name}' is not registered."
            ) from error

    def get(
        self,
        name: str,
    ) -> AgentTool:
        """Return one registered tool."""

        normalized_name = self._normalize_name(
            name
        )

        try:
            return self._tools[
                normalized_name
            ]

        except KeyError as error:
            raise ToolNotFoundError(
                f"Tool '{name}' is not registered."
            ) from error

    def contains(
        self,
        name: str,
    ) -> bool:
        """Return whether a tool is registered."""

        normalized_name = self._normalize_name(
            name
        )

        return (
            normalized_name
            in self._tools
        )

    def list_tools(
        self,
    ) -> list[AgentTool]:
        """
        Return registered tools without exposing
        the registry's internal dictionary.
        """

        return list(
            self._tools.values()
        )

    def describe_tools(
        self,
    ) -> list[dict[str, str]]:
        """
        Return safe tool metadata for decision
        engines, APIs and diagnostics.
        """

        descriptions: list[
            dict[str, str]
        ] = []

        for tool in self._tools.values():
            name, description = (
                self._validate_tool(
                    tool
                )
            )

            descriptions.append(
                {
                    "name": name,
                    "description": description,
                }
            )

        return descriptions

    def execute(
        self,
        name: str,
        context: ToolContext,
    ) -> ToolResult:
        """
        Execute a registered tool.

        Unexpected tool exceptions are converted into
        normalized failed ToolResult objects so agent
        orchestration does not crash.
        """

        if not isinstance(
            context,
            ToolContext,
        ):
            raise TypeError(
                "context must be a ToolContext."
            )

        tool = self.get(
            name
        )

        try:
            result = tool.execute(
                context
            )

        except Exception as error:
            return ToolResult(
                success=False,
                content=(
                    "Tool execution failed."
                ),
                error=(
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            )

        if not isinstance(
            result,
            ToolResult,
        ):
            return ToolResult(
                success=False,
                content=(
                    "Tool returned an invalid result."
                ),
                error=(
                    "Tool contract violation: execute() "
                    "must return ToolResult."
                ),
            )

        return result