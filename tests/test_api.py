from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import (
    app,
    set_agent_for_testing,
    set_assistant_for_testing,
)


@pytest.fixture
def mock_assistant():
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

    assert response.json() == {
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
    ] == (
        "stm32_notes.txt"
    )

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

    assert (
        "Sensitive internal error."
        not in response.text
    )


def test_history_returns_empty_history(
    client,
    mock_assistant,
) -> None:
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

    assert response.json()[
        "is_empty"
    ] is False


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

    assert response.json()[
        "inserted_chunks"
    ] == 3


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

    assert (
        "LocalMind"
        in response.text
    )

    assert (
        "Private Knowledge Workspace"
        in response.text
    )


def test_static_javascript_is_available(
    client,
) -> None:
    response = client.get(
        "/static/app.js"
    )

    assert response.status_code == 200

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

    assert (
        ".app-shell"
        in response.text
    )


def test_agent_returns_503_when_agent_not_ready(
    client,
) -> None:
    set_agent_for_testing(
        None
    )

    response = client.post(
        "/agent",
        json={
            "message": (
                "STM32 nedir?"
            ),
        },
    )

    assert response.status_code == 503


def test_agent_endpoint_returns_successful_result(
    client,
) -> None:
    from src.agent_models import (
        AgentDecision,
        AgentIntent,
        AgentResult,
        AgentStep,
        AgentStepStatus,
        ToolResult,
    )

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
                status=(
                    AgentStepStatus.COMPLETED
                ),
                detail="Completed.",
                tool_name=(
                    "knowledge_search"
                ),
            ),
        ],
        tool_result=ToolResult(
            success=True,
            content="Completed.",
        ),
        metadata={
            "execution_ms": 1.0,
            "local_only": True,
            "selected_tool": (
                "knowledge_search"
            ),
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
                "message": (
                    "STM32 nedir?"
                ),
            },
        )

        assert response.status_code == 200

    finally:
        set_agent_for_testing(
            None
        )


def test_agent_endpoint_passes_metadata(
    client,
) -> None:
    from src.agent_models import (
        AgentDecision,
        AgentIntent,
        AgentResult,
        AgentStep,
        AgentStepStatus,
        ToolResult,
    )

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
                status=(
                    AgentStepStatus.COMPLETED
                ),
                detail="Completed.",
                tool_name=(
                    "knowledge_search"
                ),
            ),
        ],
        tool_result=ToolResult(
            success=True,
            content="Completed.",
        ),
        metadata={
            "execution_ms": 1.0,
            "local_only": True,
            "selected_tool": (
                "knowledge_search"
            ),
            "request_metadata": {
                "request_id": (
                    "api-test"
                ),
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
                "message": (
                    "Test request"
                ),
                "metadata": {
                    "request_id": (
                        "api-test"
                    ),
                },
            },
        )

        assert response.status_code == 200

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
            "message": (
                "x" * 2001
            ),
        },
    )

    assert response.status_code == 422


def test_agent_endpoint_converts_validation_error(
    client,
) -> None:
    mock_agent = MagicMock()

    mock_agent.run.side_effect = (
        ValueError(
            "Invalid agent request."
        )
    )

    set_agent_for_testing(
        mock_agent
    )

    try:
        response = client.post(
            "/agent",
            json={
                "message": (
                    "Test request"
                ),
            },
        )

        assert response.status_code == 422

    finally:
        set_agent_for_testing(
            None
        )


def test_agent_endpoint_hides_internal_exception(
    client,
) -> None:
    mock_agent = MagicMock()

    mock_agent.run.side_effect = (
        RuntimeError(
            "Sensitive internal failure."
        )
    )

    set_agent_for_testing(
        mock_agent
    )

    try:
        response = client.post(
            "/agent",
            json={
                "message": (
                    "Test request"
                ),
            },
        )

        assert response.status_code == 500

        assert (
            "Sensitive internal failure"
            not in response.text
        )

    finally:
        set_agent_for_testing(
            None
        )


# ==========================================================
# KNOWLEDGE BASE FILE UPLOAD
# ==========================================================


def test_knowledge_file_upload_accepts_valid_txt(
    client,
) -> None:
    ingestion_result = {
        "source_path": (
            "temporary.txt"
        ),
        "file_count": 1,
        "inserted_chunks": 2,
        "sources": [
            "external/stm32_notes.txt",
        ],
        "recursive": False,
    }

    with patch(
        "src.api.ingest_selected_source",
        return_value=ingestion_result,
    ) as mock_ingest:
        response = client.post(
            "/knowledge/files",
            files={
                "file": (
                    "stm32_notes.txt",
                    (
                        "STM32 is an ARM-based "
                        "microcontroller family."
                    ),
                    "text/plain",
                ),
            },
        )

    assert response.status_code == 201

    assert response.json()[
        "sources"
    ] == [
        "external/stm32_notes.txt",
    ]

    call_kwargs = (
        mock_ingest.call_args.kwargs
    )

    assert (
        call_kwargs[
            "external_source_name"
        ]
        == "stm32_notes.txt"
    )

    temporary_path = (
        call_kwargs[
            "source_path"
        ]
    )

    assert (
        temporary_path.suffix
        == ".txt"
    )

    assert (
        temporary_path.exists()
        is False
    )


def test_knowledge_file_upload_accepts_pdf(
    client,
) -> None:
    with patch(
        "src.api.ingest_selected_source",
        return_value={
            "source_path": "temporary.pdf",
            "file_count": 1,
            "inserted_chunks": 2,
            "sources": [
                "external/document.pdf",
            ],
            "recursive": False,
        },
    ) as mock_ingest:
        response = client.post(
            "/knowledge/files",
            files={
                "file": (
                    "document.pdf",
                    b"%PDF-1.4 test content",
                    "application/pdf",
                ),
            },
        )

    assert response.status_code == 201

    payload = response.json()

    assert payload[
        "file_count"
    ] == 1

    assert payload[
        "inserted_chunks"
    ] == 2

    assert payload[
        "sources"
    ] == [
        "external/document.pdf"
    ]

    assert payload[
        "message"
    ] == (
        "document.pdf was added to "
        "the local knowledge base."
    )

    mock_ingest.assert_called_once()

    call_kwargs = (
        mock_ingest.call_args.kwargs
    )

    assert (
        call_kwargs[
            "external_source_name"
        ]
        == "document.pdf"
    )

    temporary_path = (
        call_kwargs[
            "source_path"
        ]
    )

    assert (
        temporary_path.suffix
        == ".pdf"
    )

    assert (
        temporary_path.exists()
        is False
    )


def test_knowledge_file_upload_accepts_docx(
    client,
) -> None:
    with patch(
        "src.api.ingest_selected_source",
        return_value={
            "source_path": "temporary.docx",
            "file_count": 1,
            "inserted_chunks": 1,
            "sources": [
                "external/report.docx",
            ],
            "recursive": False,
        },
    ) as mock_ingest:
        response = client.post(
            "/knowledge/files",
            files={
                "file": (
                    "report.docx",
                    b"DOCX binary test content",
                    (
                        "application/"
                        "vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                ),
            },
        )

    assert response.status_code == 201

    payload = response.json()

    assert payload[
        "file_count"
    ] == 1

    assert payload[
        "inserted_chunks"
    ] == 1

    assert payload[
        "sources"
    ] == [
        "external/report.docx"
    ]

    assert payload[
        "message"
    ] == (
        "report.docx was added to "
        "the local knowledge base."
    )

    mock_ingest.assert_called_once()

    call_kwargs = (
        mock_ingest.call_args.kwargs
    )

    assert (
        call_kwargs[
            "external_source_name"
        ]
        == "report.docx"
    )

    temporary_path = (
        call_kwargs[
            "source_path"
        ]
    )

    assert (
        temporary_path.suffix
        == ".docx"
    )

    assert (
        temporary_path.exists()
        is False
    )


def test_knowledge_file_upload_rejects_unsupported_extension(
    client,
) -> None:
    with patch(
        "src.api.ingest_selected_source",
    ) as mock_ingest:
        response = client.post(
            "/knowledge/files",
            files={
                "file": (
                    "spreadsheet.xlsx",
                    b"spreadsheet content",
                    (
                        "application/"
                        "vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                ),
            },
        )

    assert response.status_code == 415

    assert response.json() == {
        "detail": (
            "Supported document types are "
            "TXT, PDF, and DOCX."
        )
    }

    mock_ingest.assert_not_called()


def test_knowledge_file_upload_does_not_utf8_decode_pdf(
    client,
) -> None:
    binary_content = (
        b"\xff\xfe\x00\x81"
        b"%PDF-test"
    )

    with patch(
        "src.api.ingest_selected_source",
        return_value={
            "source_path": "temporary.pdf",
            "file_count": 1,
            "inserted_chunks": 1,
            "sources": [
                "external/binary.pdf",
            ],
            "recursive": False,
        },
    ):
        response = client.post(
            "/knowledge/files",
            files={
                "file": (
                    "binary.pdf",
                    binary_content,
                    "application/pdf",
                ),
            },
        )

    assert response.status_code == 201


def test_knowledge_file_upload_does_not_utf8_decode_docx(
    client,
) -> None:
    binary_content = (
        b"\xff\xfe\x00\x81"
        b"DOCX-test"
    )

    with patch(
        "src.api.ingest_selected_source",
        return_value={
            "source_path": "temporary.docx",
            "file_count": 1,
            "inserted_chunks": 1,
            "sources": [
                "external/binary.docx",
            ],
            "recursive": False,
        },
    ):
        response = client.post(
            "/knowledge/files",
            files={
                "file": (
                    "binary.docx",
                    binary_content,
                    (
                        "application/"
                        "vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                ),
            },
        )

    assert response.status_code == 201


def test_knowledge_file_upload_rejects_empty_txt(
    client,
) -> None:
    with patch(
        "src.api.ingest_selected_source",
    ) as mock_ingest:
        response = client.post(
            "/knowledge/files",
            files={
                "file": (
                    "empty.txt",
                    b"",
                    "text/plain",
                ),
            },
        )

    assert response.status_code == 422

    mock_ingest.assert_not_called()


def test_knowledge_file_upload_rejects_invalid_utf8(
    client,
) -> None:
    invalid_utf8 = bytes(
        [
            0xFF,
            0xFE,
            0xFA,
        ]
    )

    with patch(
        "src.api.ingest_selected_source",
    ) as mock_ingest:
        response = client.post(
            "/knowledge/files",
            files={
                "file": (
                    "invalid.txt",
                    invalid_utf8,
                    "text/plain",
                ),
            },
        )

    assert response.status_code == 422

    mock_ingest.assert_not_called()


def test_knowledge_file_upload_rejects_oversized_file(
    client,
) -> None:
    oversized_content = (
        b"a"
        * (
            5 * 1024 * 1024
            + 1
        )
    )

    with patch(
        "src.api.ingest_selected_source",
    ) as mock_ingest:
        response = client.post(
            "/knowledge/files",
            files={
                "file": (
                    "large.txt",
                    oversized_content,
                    "text/plain",
                ),
            },
        )

    assert response.status_code == 413

    mock_ingest.assert_not_called()


def test_knowledge_file_upload_sanitizes_filename(
    client,
) -> None:
    ingestion_result = {
        "source_path": (
            "temporary.txt"
        ),
        "file_count": 1,
        "inserted_chunks": 1,
        "sources": [
            "external/secret.txt",
        ],
        "recursive": False,
    }

    with patch(
        "src.api.ingest_selected_source",
        return_value=ingestion_result,
    ) as mock_ingest:
        response = client.post(
            "/knowledge/files",
            files={
                "file": (
                    "../../secret.txt",
                    b"Safe local document.",
                    "text/plain",
                ),
            },
        )

    assert response.status_code == 201

    assert (
        mock_ingest
        .call_args
        .kwargs[
            "external_source_name"
        ]
        == "secret.txt"
    )


def test_knowledge_file_upload_maps_ingestion_error(
    client,
) -> None:
    with patch(
        "src.api.ingest_selected_source",
        side_effect=ValueError(
            "Invalid selected document."
        ),
    ):
        response = client.post(
            "/knowledge/files",
            files={
                "file": (
                    "notes.txt",
                    b"Document content",
                    "text/plain",
                ),
            },
        )

    assert response.status_code == 422

def test_knowledge_file_upload_hides_temporary_path_in_validation_error(
    client,
) -> None:
    sensitive_error = (
        "Could not read PDF document: "
        "C:\\Users\\pelin\\AppData\\Local\\Temp\\tmp12345.pdf"
    )

    with patch(
        "src.api.ingest_selected_source",
        side_effect=RuntimeError(
            sensitive_error
        ),
    ):
        response = client.post(
            "/knowledge/files",
            files={
                "file": (
                    "document.pdf",
                    b"%PDF-1.4 invalid test content",
                    "application/pdf",
                ),
            },
        )

    assert response.status_code == 422

    assert (
        "C:\\Users\\pelin"
        not in response.text
    )

    assert (
        "tmp12345.pdf"
        not in response.text
    )

    assert response.json() == {
        "detail": (
            "The PDF document could not be read "
            "or contains no extractable text."
        )
    }

def test_knowledge_file_upload_hides_internal_error(
    client,
) -> None:
    with patch(
        "src.api.ingest_selected_source",
        side_effect=OSError(
            "Sensitive operating-system path."
        ),
    ):
        response = client.post(
            "/knowledge/files",
            files={
                "file": (
                    "notes.txt",
                    b"Document content",
                    "text/plain",
                ),
            },
        )

    assert response.status_code == 500

    assert (
        "Sensitive operating-system path"
        not in response.text
    )


# ==========================================================
# KNOWLEDGE BASE SOURCES
# ==========================================================


def test_knowledge_sources_returns_indexed_sources(
    client,
) -> None:
    indexed_sources = [
        "external/ui_upload_test.txt",
        "foundry_local_notes.txt",
        "pid_notes.txt",
        "rag_notes.txt",
        "sqlite_notes.txt",
        "stm32_notes.txt",
    ]

    with patch(
        "src.api.get_unique_document_sources",
        return_value=indexed_sources,
    ) as mock_get_sources:
        response = client.get(
            "/knowledge/sources"
        )

    assert response.status_code == 200

    assert response.json() == {
        "source_count": 6,
        "sources": indexed_sources,
    }

    mock_get_sources.assert_called_once_with()


def test_knowledge_sources_returns_empty_library(
    client,
) -> None:
    with patch(
        "src.api.get_unique_document_sources",
        return_value=[],
    ):
        response = client.get(
            "/knowledge/sources"
        )

    assert response.status_code == 200

    assert response.json() == {
        "source_count": 0,
        "sources": [],
    }


def test_knowledge_sources_hides_internal_error(
    client,
) -> None:
    with patch(
        "src.api.get_unique_document_sources",
        side_effect=RuntimeError(
            "Sensitive database details."
        ),
    ):
        response = client.get(
            "/knowledge/sources"
        )

    assert response.status_code == 500

    assert response.json() == {
        "detail": (
            "Knowledge-base sources "
            "could not be retrieved."
        ),
    }

    assert (
        "Sensitive database details."
        not in response.text
    )