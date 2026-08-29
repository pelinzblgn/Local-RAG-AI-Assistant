import pytest

from src.agent_decision import AgentDecisionEngine
from src.agent_models import (
    AgentIntent,
    ToolResult,
)
from src.agent_tools import (
    AgentTool,
    ToolContext,
    ToolRegistry,
)


class DummyTool(AgentTool):
    """
    Minimal deterministic tool used only for
    decision-engine tests.
    """

    def __init__(
        self,
        tool_name: str,
    ) -> None:
        self._tool_name = tool_name

    @property
    def name(self) -> str:
        return self._tool_name

    @property
    def description(self) -> str:
        return (
            f"Test tool for {self._tool_name}."
        )

    def execute(
        self,
        context: ToolContext,
    ) -> ToolResult:
        return ToolResult(
            success=True,
            content=context.user_input,
        )


def create_registry(
    *tool_names: str,
) -> ToolRegistry:
    """
    Create an isolated registry containing only
    the requested test tools.
    """

    return ToolRegistry(
        tools=[
            DummyTool(name)
            for name in tool_names
        ]
    )


def test_decision_engine_requires_registry() -> None:
    with pytest.raises(
        TypeError,
        match="ToolRegistry",
    ):
        AgentDecisionEngine(
            object()  # type: ignore[arg-type]
        )


def test_empty_input_is_rejected() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search"
        )
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        engine.decide(
            "   "
        )


def test_non_string_input_is_rejected() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search"
        )
    )

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        engine.decide(
            123  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "user_input",
    [
        "Bilgi tabanını senkronize et.",
        "Bilgi tabanını güncelle.",
        "Knowledge base sync",
        "Sync yap.",
        "Bilgi tabanını yenile.",
    ],
)
def test_sync_requests_route_to_sync_tool(
    user_input: str,
) -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search",
            "knowledge_status",
            "knowledge_sync",
        )
    )

    decision = engine.decide(
        user_input
    )

    assert (
        decision.intent
        == AgentIntent.KNOWLEDGE_SYNC
    )

    assert (
        decision.tool_name
        == "knowledge_sync"
    )

    assert decision.confidence == 0.99


@pytest.mark.parametrize(
    "user_input",
    [
        "Bilgi tabanında kaç belge var?",
        "Bilgi tabanında kaç kaynak var?",
        "Bilgi tabanında ne var?",
        "Bilgi tabanı durumu nedir?",
        "Knowledge base status",
        "Index status",
    ],
)
def test_status_requests_route_to_status_tool(
    user_input: str,
) -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search",
            "knowledge_status",
            "knowledge_sync",
        )
    )

    decision = engine.decide(
        user_input
    )

    assert (
        decision.intent
        == AgentIntent.KNOWLEDGE_STATUS
    )

    assert (
        decision.tool_name
        == "knowledge_status"
    )

    assert decision.confidence == 0.98


@pytest.mark.parametrize(
    "user_input",
    [
        "STM32 nedir?",
        "PID kontrolcüsü nasıl çalışır?",
        "RAG sistemi nasıl çalışır?",
        "SQLite neden kullanılır?",
        "PWM ne işe yarar?",
    ],
)
def test_normal_questions_route_to_knowledge_search(
    user_input: str,
) -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search",
            "knowledge_status",
            "knowledge_sync",
        )
    )

    decision = engine.decide(
        user_input
    )

    assert (
        decision.intent
        == AgentIntent.KNOWLEDGE_QUERY
    )

    assert (
        decision.tool_name
        == "knowledge_search"
    )

    assert decision.confidence == 0.85


def test_turkish_characters_are_normalized() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search",
            "knowledge_sync",
        )
    )

    decision = engine.decide(
        "BİLGİ TABANINI GÜNCELLE!"
    )

    assert (
        decision.intent
        == AgentIntent.KNOWLEDGE_SYNC
    )

    assert (
        decision.tool_name
        == "knowledge_sync"
    )


def test_punctuation_does_not_break_routing() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search",
            "knowledge_status",
        )
    )

    decision = engine.decide(
        "Bilgi tabanında... kaç belge var???"
    )

    assert (
        decision.intent
        == AgentIntent.KNOWLEDGE_STATUS
    )

    assert (
        decision.tool_name
        == "knowledge_status"
    )


def test_unavailable_sync_tool_is_not_selected() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search"
        )
    )

    decision = engine.decide(
        "Bilgi tabanını senkronize et."
    )

    assert (
        decision.intent
        == AgentIntent.UNKNOWN
    )

    assert decision.tool_name is None
    assert decision.confidence == 1.0

    assert (
        "not registered"
        in decision.reason.lower()
    )


def test_unavailable_status_tool_is_not_selected() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search"
        )
    )

    decision = engine.decide(
        "Bilgi tabanında kaç belge var?"
    )

    assert (
        decision.intent
        == AgentIntent.UNKNOWN
    )

    assert decision.tool_name is None
    assert decision.confidence == 1.0

    assert (
        "not registered"
        in decision.reason.lower()
    )


def test_clear_history_routes_when_tool_is_registered() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search",
            "conversation_clear",
        )
    )

    decision = engine.decide(
        "Konuşma geçmişini temizle."
    )

    assert (
        decision.intent
        == AgentIntent.CONVERSATION_CLEAR
    )

    assert (
        decision.tool_name
        == "conversation_clear"
    )

    assert decision.confidence == 0.99


def test_clear_history_does_not_select_missing_tool() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search"
        )
    )

    decision = engine.decide(
        "Konuşma geçmişini temizle."
    )

    assert (
        decision.intent
        == AgentIntent.UNKNOWN
    )

    assert decision.tool_name is None
    assert decision.confidence == 1.0

    assert (
        "conversation_clear"
        in decision.reason
    )


def test_empty_registry_returns_unknown() -> None:
    engine = AgentDecisionEngine(
        ToolRegistry()
    )

    decision = engine.decide(
        "STM32 nedir?"
    )

    assert (
        decision.intent
        == AgentIntent.UNKNOWN
    )

    assert decision.tool_name is None
    assert decision.confidence == 0.0


def test_registry_without_matching_or_search_tool_returns_unknown() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_status"
        )
    )

    decision = engine.decide(
        "STM32 nedir?"
    )

    assert (
        decision.intent
        == AgentIntent.UNKNOWN
    )

    assert decision.tool_name is None
    assert decision.confidence == 0.0


def test_operational_command_has_priority_over_search() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search",
            "knowledge_sync",
        )
    )

    decision = engine.decide(
        "Bilgi tabanını güncelle."
    )

    assert (
        decision.tool_name
        == "knowledge_sync"
    )

    assert (
        decision.intent
        == AgentIntent.KNOWLEDGE_SYNC
    )


def test_status_command_has_priority_over_search() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search",
            "knowledge_status",
        )
    )

    decision = engine.decide(
        "Bilgi tabanında kaç kaynak var?"
    )

    assert (
        decision.tool_name
        == "knowledge_status"
    )

    assert (
        decision.intent
        == AgentIntent.KNOWLEDGE_STATUS
    )


def test_decision_contains_explanation() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search"
        )
    )

    decision = engine.decide(
        "STM32 nedir?"
    )

    assert decision.reason.strip()

    assert (
        "knowledge"
        in decision.reason.lower()
    )


def test_same_input_produces_same_decision() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search",
            "knowledge_status",
            "knowledge_sync",
        )
    )

    first = engine.decide(
        "Bilgi tabanında kaç belge var?"
    )

    second = engine.decide(
        "Bilgi tabanında kaç belge var?"
    )

    assert first == second


def test_missing_operational_capability_never_falls_back_to_search() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search"
        )
    )

    sync_decision = engine.decide(
        "Bilgi tabanını güncelle."
    )

    status_decision = engine.decide(
        "Bilgi tabanında kaç kaynak var?"
    )

    clear_decision = engine.decide(
        "Geçmişi temizle."
    )

    for decision in (
        sync_decision,
        status_decision,
        clear_decision,
    ):
        assert (
            decision.intent
            == AgentIntent.UNKNOWN
        )

        assert decision.tool_name is None

        assert (
            decision.tool_name
            != "knowledge_search"
        )


def test_regular_question_still_uses_search_after_hardening() -> None:
    engine = AgentDecisionEngine(
        create_registry(
            "knowledge_search"
        )
    )

    decision = engine.decide(
        "PWM ne işe yarar?"
    )

    assert (
        decision.intent
        == AgentIntent.KNOWLEDGE_QUERY
    )

    assert (
        decision.tool_name
        == "knowledge_search"
    )