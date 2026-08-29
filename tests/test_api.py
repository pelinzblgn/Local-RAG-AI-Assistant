from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import (
    app,
    set_assistant_for_testing,
)


@pytest.fixture
def mock_assistant():
    """
    Provide an isolated assistant mock.

    Real Foundry Local models must never be loaded
    during API unit tests.
    """

    assistant = MagicMock()

    assistant.answer.return_value = {
        "answer": (
            "STM32, STMicroelectronics tarafından "
            "geliştirilen ARM tabanlı bir "
            "mikrodenetleyici ailesidir."
        ),
        "sources": [
            "stm32_notes.txt",
        ],
        "confidence": {
            "level": "high",
            "is_confident": True,
            "top_score": 0.91,
            "second_score": 0.42,
            "score_gap": 0.49,
            "evidence_coverage": 1.0,
            "selected_count": 1,
            "total_count": 3,
            "filtered_count": 2,
            "reason": (
                "Strong semantic match."
            ),
        },
        "query_rewrite": {
            "original_query": (
                "STM32 nedir?"
            ),
            "retrieval_query": (
                "STM32 nedir?"
            ),
            "was_rewritten": False,
            "reason": (
                "The query is self-contained."
            ),
        },
        "retrieved_documents": [
            {
                "id": 1,
                "content": (
                    "STM32, STMicroelectronics "
                    "tarafından geliştirilen ARM "
                    "tabanlı mikrodenetleyici "
                    "ailesidir."
                ),
                "source": (
                    "stm32_notes.txt"
                ),
                "score": 0.91,
            },
        ],
    }

    assistant.get_conversation_history.return_value = (
        ""
    )

    set_assistant_for_testing(
        assistant
    )

    yield assistant

    set_assistant_for_testing(
        None
    )


@pytest.fixture
def client():
    """
    Create a TestClient without entering the
    application lifespan.

    The assistant state is injected explicitly
    by individual tests.
    """

    return TestClient(
        app
    )


def test_health_returns_ok_when_assistant_ready(
    client,
    mock_assistant,
) -> None:
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload == {
        "status": "ok",
        "assistant_ready": True,
        "agent_ready": False,
        "local_only": True,
    }
    
    assert payload == {
    "status": "ok",
    "assistant_ready": True,
    "agent_ready": False,
    "local_only": True,
}


def test_health_reports_not_ready_without_assistant(
    client,
) -> None:
    set_assistant_for_testing(
        None
    )

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "assistant_ready"
    ] is False

    assert payload[
        "local_only"
    ] is True


def test_chat_returns_complete_rag_response(
    client,
    mock_assistant,
) -> None:
    response = client.post(
        "/chat",
        json={
            "question": (
                "STM32 nedir?"
            ),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "answer"
    ].startswith(
        "STM32"
    )

    assert payload[
        "sources"
    ] == [
        "stm32_notes.txt",
    ]

    assert payload[
        "confidence"
    ][
        "level"
    ] == "high"

    assert payload[
        "query_rewrite"
    ][
        "was_rewritten"
    ] is False

    assert len(
        payload[
            "retrieved_documents"
        ]
    ) == 1

    assert payload[
        "retrieved_documents"
    ][0][
        "source"
    ] == "stm32_notes.txt"

    mock_assistant.answer.assert_called_once_with(
        "STM32 nedir?"
    )


def test_chat_strips_question_whitespace(
    client,
    mock_assistant,
) -> None:
    response = client.post(
        "/chat",
        json={
            "question": (
                "   STM32 nedir?   "
            ),
        },
    )

    assert response.status_code == 200

    mock_assistant.answer.assert_called_once_with(
        "STM32 nedir?"
    )


def test_chat_rejects_empty_question(
    client,
    mock_assistant,
) -> None:
    response = client.post(
        "/chat",
        json={
            "question": "   ",
        },
    )

    assert response.status_code == 422

    mock_assistant.answer.assert_not_called()


def test_chat_rejects_missing_question(
    client,
    mock_assistant,
) -> None:
    response = client.post(
        "/chat",
        json={},
    )

    assert response.status_code == 422

    mock_assistant.answer.assert_not_called()


def test_chat_rejects_question_over_limit(
    client,
    mock_assistant,
) -> None:
    response = client.post(
        "/chat",
        json={
            "question": (
                "a" * 2001
            ),
        },
    )

    assert response.status_code == 422

    mock_assistant.answer.assert_not_called()


def test_chat_returns_503_when_assistant_not_ready(
    client,
) -> None:
    set_assistant_for_testing(
        None
    )

    response = client.post(
        "/chat",
        json={
            "question": (
                "STM32 nedir?"
            ),
        },
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "Local RAG assistant is not ready."
        )
    }


def test_chat_maps_validation_error_to_422(
    client,
    mock_assistant,
) -> None:
    mock_assistant.answer.side_effect = (
        ValueError(
            "Invalid question."
        )
    )

    response = client.post(
        "/chat",
        json={
            "question": (
                "STM32 nedir?"
            ),
        },
    )

    assert response.status_code == 422

    assert response.json() == {
        "detail": (
            "Invalid question."
        )
    }


def test_chat_hides_internal_server_error(
    client,
    mock_assistant,
) -> None:
    mock_assistant.answer.side_effect = (
        RuntimeError(
            "Sensitive internal error."
        )
    )

    response = client.post(
        "/chat",
        json={
            "question": (
                "STM32 nedir?"
            ),
        },
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": (
            "The local RAG assistant "
            "could not process the request."
        )
    }

    assert (
        "Sensitive internal error."
        not in response.text
    )


def test_history_returns_empty_history(
    client,
    mock_assistant,
) -> None:
    mock_assistant.get_conversation_history.return_value = (
        ""
    )

    response = client.get(
        "/history"
    )

    assert response.status_code == 200

    assert response.json() == {
        "history": "",
        "is_empty": True,
    }


def test_history_returns_conversation(
    client,
    mock_assistant,
) -> None:
    mock_assistant.get_conversation_history.return_value = (
        "[Konuşma 1]\n"
        "Kullanıcı: STM32 nedir?\n"
        "Asistan: STM32 bir "
        "mikrodenetleyici ailesidir."
    )

    response = client.get(
        "/history"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "is_empty"
    ] is False

    assert (
        "STM32 nedir?"
        in payload[
            "history"
        ]
    )


def test_history_returns_503_when_assistant_not_ready(
    client,
) -> None:
    set_assistant_for_testing(
        None
    )

    response = client.get(
        "/history"
    )

    assert response.status_code == 503


def test_clear_history_delegates_to_assistant(
    client,
    mock_assistant,
) -> None:
    response = client.delete(
        "/history"
    )

    assert response.status_code == 200

    assert response.json() == {
        "message": (
            "Conversation history cleared."
        )
    }

    mock_assistant.clear_memory.assert_called_once_with()


def test_clear_history_returns_503_when_not_ready(
    client,
) -> None:
    set_assistant_for_testing(
        None
    )

    response = client.delete(
        "/history"
    )

    assert response.status_code == 503


def test_sync_returns_sync_result(
    client,
) -> None:
    sync_result = SimpleNamespace(
        new_files=[
            "new.txt",
        ],
        modified_files=[
            "modified.txt",
        ],
        deleted_files=[
            "deleted.txt",
        ],
        unchanged_files=[
            "stable.txt",
        ],
        inserted_chunks=3,
        deleted_chunks=2,
        has_changes=True,
    )

    with patch(
        "src.api.synchronize_knowledge_base",
        return_value=sync_result,
    ):
        response = client.post(
            "/sync"
        )

    assert response.status_code == 200

    assert response.json() == {
        "new_files": [
            "new.txt",
        ],
        "modified_files": [
            "modified.txt",
        ],
        "deleted_files": [
            "deleted.txt",
        ],
        "unchanged_files": [
            "stable.txt",
        ],
        "inserted_chunks": 3,
        "deleted_chunks": 2,
        "has_changes": True,
    }


def test_sync_hides_internal_error(
    client,
) -> None:
    with patch(
        "src.api.synchronize_knowledge_base",
        side_effect=RuntimeError(
            "Database details."
        ),
    ):
        response = client.post(
            "/sync"
        )

    assert response.status_code == 500

    assert response.json() == {
        "detail": (
            "Knowledge base "
            "synchronization failed."
        )
    }

    assert (
        "Database details."
        not in response.text
    )


def test_web_interface_returns_index(
    client,
) -> None:
    response = client.get(
        "/"
    )

    assert response.status_code == 200

    content_type = response.headers.get(
        "content-type",
        "",
    )

    assert (
        "text/html"
        in content_type
    )

    assert (
        "Local RAG"
        in response.text
    )


def test_static_javascript_is_available(
    client,
) -> None:
    response = client.get(
        "/static/app.js"
    )

    assert response.status_code == 200

    content_type = response.headers.get(
        "content-type",
        "",
    )

    assert (
        "javascript"
        in content_type
        or
        "text/plain"
        in content_type
    )

    assert (
        "initializeApplication"
        in response.text
    )


def test_static_stylesheet_is_available(
    client,
) -> None:
    response = client.get(
        "/static/styles.css"
    )

    assert response.status_code == 200

    content_type = response.headers.get(
        "content-type",
        "",
    )

    assert (
        "text/css"
        in content_type
    )

    assert (
        ".app-shell"
        in response.text
    )
    
def test_agent_returns_503_when_agent_not_ready(
    client,
) -> None:
    from src.api import set_agent_for_testing

    set_agent_for_testing(
        None
    )

    response = client.post(
        "/agent",
        json={
            "message": "STM32 nedir?",
        },
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": "Local agent is not ready."
    }


def test_agent_endpoint_returns_successful_result(
    client,
) -> None:
    from unittest.mock import MagicMock

    from src.agent_models import (
        AgentDecision,
        AgentIntent,
        AgentResult,
        AgentStep,
        AgentStepStatus,
        ToolResult,
    )
    from src.api import set_agent_for_testing

    mock_agent = MagicMock()

    mock_agent.run.return_value = AgentResult(
        answer="STM32 bir mikrodenetleyici ailesidir.",
        decision=AgentDecision(
            intent=AgentIntent.KNOWLEDGE_QUERY,
            tool_name="knowledge_search",
            reason="Knowledge query detected.",
            confidence=0.85,
        ),
        steps=[
            AgentStep(
                name="Intent analysis",
                status=AgentStepStatus.COMPLETED,
                detail="Knowledge query detected.",
                tool_name="knowledge_search",
            ),
            AgentStep(
                name="Tool selection",
                status=AgentStepStatus.COMPLETED,
                detail="knowledge_search selected.",
                tool_name="knowledge_search",
            ),
            AgentStep(
                name="Tool execution",
                status=AgentStepStatus.COMPLETED,
                detail="Tool completed.",
                tool_name="knowledge_search",
            ),
            AgentStep(
                name="Response assembly",
                status=AgentStepStatus.COMPLETED,
                detail="Response assembled.",
                tool_name="knowledge_search",
            ),
        ],
        tool_result=ToolResult(
            success=True,
            content=(
                "STM32 bir mikrodenetleyici ailesidir."
            ),
            data={
                "sources": [
                    "stm32_notes.txt",
                ],
            },
        ),
        metadata={
            "execution_ms": 12.5,
            "local_only": True,
            "selected_tool": "knowledge_search",
            "request_metadata": {},
        },
    )

    set_agent_for_testing(
        mock_agent
    )

    try:
        response = client.post(
            "/agent",
            json={
                "message": "STM32 nedir?",
            },
        )

        assert response.status_code == 200

        payload = response.json()

        assert payload["answer"] == (
            "STM32 bir mikrodenetleyici ailesidir."
        )

        assert payload["succeeded"] is True

        assert payload["decision"] == {
            "intent": "knowledge_query",
            "tool_name": "knowledge_search",
            "reason": "Knowledge query detected.",
            "confidence": 0.85,
        }

        assert len(
            payload["steps"]
        ) == 4

        assert payload[
            "tool_result"
        ][
            "success"
        ] is True

        assert payload[
            "tool_result"
        ][
            "data"
        ][
            "sources"
        ] == [
            "stm32_notes.txt",
        ]

        assert payload[
            "metadata"
        ][
            "local_only"
        ] is True

        mock_agent.run.assert_called_once_with(
            user_input="STM32 nedir?",
            metadata={},
        )

    finally:
        set_agent_for_testing(
            None
        )


def test_agent_endpoint_passes_metadata(
    client,
) -> None:
    from unittest.mock import MagicMock

    from src.agent_models import (
        AgentDecision,
        AgentIntent,
        AgentResult,
        AgentStep,
        AgentStepStatus,
        ToolResult,
    )
    from src.api import set_agent_for_testing

    mock_agent = MagicMock()

    mock_agent.run.return_value = AgentResult(
        answer="Completed.",
        decision=AgentDecision(
            intent=AgentIntent.KNOWLEDGE_QUERY,
            tool_name="knowledge_search",
            reason="Knowledge query.",
            confidence=0.85,
        ),
        steps=[
            AgentStep(
                name="Intent analysis",
                status=AgentStepStatus.COMPLETED,
                detail="Completed.",
                tool_name="knowledge_search",
            ),
        ],
        tool_result=ToolResult(
            success=True,
            content="Completed.",
        ),
        metadata={
            "execution_ms": 1.0,
            "local_only": True,
            "selected_tool": "knowledge_search",
            "request_metadata": {
                "request_id": "api-test",
            },
        },
    )

    set_agent_for_testing(
        mock_agent
    )

    try:
        response = client.post(
            "/agent",
            json={
                "message": "Test request",
                "metadata": {
                    "request_id": "api-test",
                },
            },
        )

        assert response.status_code == 200

        mock_agent.run.assert_called_once_with(
            user_input="Test request",
            metadata={
                "request_id": "api-test",
            },
        )

    finally:
        set_agent_for_testing(
            None
        )


def test_agent_endpoint_rejects_empty_message(
    client,
) -> None:
    response = client.post(
        "/agent",
        json={
            "message": "   ",
        },
    )

    assert response.status_code == 422


def test_agent_endpoint_rejects_missing_message(
    client,
) -> None:
    response = client.post(
        "/agent",
        json={},
    )

    assert response.status_code == 422


def test_agent_endpoint_rejects_oversized_message(
    client,
) -> None:
    response = client.post(
        "/agent",
        json={
            "message": "x" * 2001,
        },
    )

    assert response.status_code == 422


def test_agent_endpoint_converts_validation_error(
    client,
) -> None:
    from unittest.mock import MagicMock

    from src.api import set_agent_for_testing

    mock_agent = MagicMock()

    mock_agent.run.side_effect = ValueError(
        "Invalid agent request."
    )

    set_agent_for_testing(
        mock_agent
    )

    try:
        response = client.post(
            "/agent",
            json={
                "message": "Test request",
            },
        )

        assert response.status_code == 422

        assert response.json() == {
            "detail": "Invalid agent request."
        }

    finally:
        set_agent_for_testing(
            None
        )


def test_agent_endpoint_hides_internal_exception(
    client,
) -> None:
    from unittest.mock import MagicMock

    from src.api import set_agent_for_testing

    mock_agent = MagicMock()

    mock_agent.run.side_effect = RuntimeError(
        "Sensitive internal failure."
    )

    set_agent_for_testing(
        mock_agent
    )

    try:
        response = client.post(
            "/agent",
            json={
                "message": "Test request",
            },
        )

        assert response.status_code == 500

        assert response.json() == {
            "detail": (
                "The local agent could not "
                "process the request."
            )
        }

        assert (
            "Sensitive internal failure"
            not in response.text
        )

    finally:
        set_agent_for_testing(
            None
        )