from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agent_tools import (
    ToolContext,
    ToolRegistry,
)
from src.assistant import RAGAssistant
from src.knowledge_tools import (
    KnowledgeSearchTool,
    KnowledgeStatusTool,
    KnowledgeSyncTool,
)


def create_assistant() -> RAGAssistant:
    """
    Create a RAGAssistant whose local LLM is mocked.

    No Foundry Local model is loaded by these tests.
    """

    mock_llm = MagicMock()

    return RAGAssistant(
        local_llm=mock_llm,
    )


def test_knowledge_search_tool_metadata() -> None:
    assistant = create_assistant()

    tool = KnowledgeSearchTool(
        assistant
    )

    assert tool.name == "knowledge_search"

    assert (
        "local knowledge base"
        in tool.description.lower()
    )


def test_knowledge_search_tool_rejects_invalid_assistant() -> None:
    with pytest.raises(
        TypeError,
        match="RAGAssistant",
    ):
        KnowledgeSearchTool(
            object()  # type: ignore[arg-type]
        )


def test_knowledge_search_delegates_to_rag_assistant(
    monkeypatch,
) -> None:
    assistant = create_assistant()

    expected_response = {
        "answer": (
            "STM32 bir mikrodenetleyici ailesidir."
        ),
        "sources": [
            "stm32_notes.txt",
        ],
        "confidence": {
            "level": "high",
            "top_score": 0.91,
        },
        "query_rewrite": {
            "original_query": "STM32 nedir?",
            "retrieval_query": "STM32 nedir?",
            "was_rewritten": False,
        },
        "retrieved_documents": [
            {
                "id": 1,
                "source": "stm32_notes.txt",
                "content": "STM32 content",
                "score": 0.91,
            }
        ],
    }

    answer_mock = MagicMock(
        return_value=expected_response
    )

    monkeypatch.setattr(
        assistant,
        "answer",
        answer_mock,
    )

    tool = KnowledgeSearchTool(
        assistant
    )

    result = tool.execute(
        ToolContext(
            user_input="STM32 nedir?"
        )
    )

    assert result.success is True

    assert result.content == (
        "STM32 bir mikrodenetleyici ailesidir."
    )

    answer_mock.assert_called_once_with(
        "STM32 nedir?"
    )


def test_knowledge_search_preserves_rag_metadata(
    monkeypatch,
) -> None:
    assistant = create_assistant()

    monkeypatch.setattr(
        assistant,
        "answer",
        MagicMock(
            return_value={
                "answer": "PWM motor hızını kontrol eder.",
                "sources": [
                    "stm32_notes.txt",
                ],
                "confidence": {
                    "level": "high",
                    "top_score": 0.88,
                },
                "query_rewrite": {
                    "original_query": (
                        "Peki PWM ne işe yarar?"
                    ),
                    "retrieval_query": (
                        "STM32 nedir? "
                        "Peki PWM ne işe yarar?"
                    ),
                    "was_rewritten": True,
                },
                "retrieved_documents": [
                    {
                        "id": 5,
                        "source": (
                            "stm32_notes.txt"
                        ),
                        "content": (
                            "PWM sinyali motor "
                            "hızını kontrol eder."
                        ),
                        "score": 0.88,
                    }
                ],
            }
        ),
    )

    tool = KnowledgeSearchTool(
        assistant
    )

    result = tool.execute(
        ToolContext(
            user_input=(
                "Peki PWM ne işe yarar?"
            )
        )
    )

    assert result.success is True

    assert result.data[
        "sources"
    ] == [
        "stm32_notes.txt",
    ]

    assert result.data[
        "confidence"
    ][
        "level"
    ] == "high"

    assert result.data[
        "query_rewrite"
    ][
        "was_rewritten"
    ] is True

    assert result.data[
        "retrieved_documents"
    ][0][
        "source"
    ] == "stm32_notes.txt"


def test_knowledge_search_normalizes_missing_metadata(
    monkeypatch,
) -> None:
    assistant = create_assistant()

    monkeypatch.setattr(
        assistant,
        "answer",
        MagicMock(
            return_value={
                "answer": "Valid answer.",
            }
        ),
    )

    tool = KnowledgeSearchTool(
        assistant
    )

    result = tool.execute(
        ToolContext(
            user_input="Question"
        )
    )

    assert result.success is True

    assert result.data == {
        "sources": [],
        "confidence": {},
        "query_rewrite": {},
        "retrieved_documents": [],
    }


def test_knowledge_search_normalizes_invalid_metadata(
    monkeypatch,
) -> None:
    assistant = create_assistant()

    monkeypatch.setattr(
        assistant,
        "answer",
        MagicMock(
            return_value={
                "answer": "Valid answer.",
                "sources": "invalid",
                "confidence": "invalid",
                "query_rewrite": None,
                "retrieved_documents": 123,
            }
        ),
    )

    tool = KnowledgeSearchTool(
        assistant
    )

    result = tool.execute(
        ToolContext(
            user_input="Question"
        )
    )

    assert result.success is True

    assert result.data == {
        "sources": [],
        "confidence": {},
        "query_rewrite": {},
        "retrieved_documents": [],
    }


def test_knowledge_search_rejects_missing_answer(
    monkeypatch,
) -> None:
    assistant = create_assistant()

    monkeypatch.setattr(
        assistant,
        "answer",
        MagicMock(
            return_value={
                "sources": [],
            }
        ),
    )

    tool = KnowledgeSearchTool(
        assistant
    )

    result = tool.execute(
        ToolContext(
            user_input="Question"
        )
    )

    assert result.success is False

    assert result.error is not None

    assert (
        "valid answer"
        in result.error.lower()
    )


def test_knowledge_search_rejects_empty_answer(
    monkeypatch,
) -> None:
    assistant = create_assistant()

    monkeypatch.setattr(
        assistant,
        "answer",
        MagicMock(
            return_value={
                "answer": "   ",
            }
        ),
    )

    tool = KnowledgeSearchTool(
        assistant
    )

    result = tool.execute(
        ToolContext(
            user_input="Question"
        )
    )

    assert result.success is False

    assert result.error is not None

    assert (
        "empty"
        in result.error.lower()
    )


def test_knowledge_search_can_be_executed_through_registry(
    monkeypatch,
) -> None:
    from src.agent_tools import ToolRegistry

    assistant = create_assistant()

    monkeypatch.setattr(
        assistant,
        "answer",
        MagicMock(
            return_value={
                "answer": (
                    "RAG retrieves relevant "
                    "local evidence."
                ),
                "sources": [
                    "rag_notes.txt",
                ],
                "confidence": {
                    "level": "high",
                },
                "query_rewrite": {},
                "retrieved_documents": [],
            }
        ),
    )

    registry = ToolRegistry(
        tools=[
            KnowledgeSearchTool(
                assistant
            ),
        ]
    )

    result = registry.execute(
        name="knowledge_search",
        context=ToolContext(
            user_input=(
                "RAG nasıl çalışır?"
            )
        ),
    )

    assert result.success is True

    assert result.data[
        "sources"
    ] == [
        "rag_notes.txt",
    ]


def test_knowledge_status_tool_metadata() -> None:
    tool = KnowledgeStatusTool()

    assert tool.name == "knowledge_status"

    assert (
        "knowledge base"
        in tool.description.lower()
    )

def test_knowledge_status_reports_database_state() -> None:
    documents = [
        {
            "id": 1,
            "content": "STM32 content",
            "source": "stm32_notes.txt",
        },
        {
            "id": 2,
            "content": "PWM content",
            "source": "stm32_notes.txt",
        },
        {
            "id": 3,
            "content": "PID content",
            "source": "pid_notes.txt",
        },
    ]

    with patch(
        "src.knowledge_tools.get_all_documents",
        return_value=documents,
    ):
        result = KnowledgeStatusTool().execute(
            ToolContext(
                user_input=(
                    "Bilgi tabanında ne var?"
                )
            )
        )

    assert result.success is True

    assert result.data[
        "document_count"
    ] == 3

    assert result.data[
        "source_count"
    ] == 2

    assert result.data[
        "sources"
    ] == [
        "pid_notes.txt",
        "stm32_notes.txt",
    ]

    assert result.data[
        "local_only"
    ] is True


def test_knowledge_status_handles_empty_database() -> None:
    with patch(
        "src.knowledge_tools.get_all_documents",
        return_value=[],
    ):
        result = KnowledgeStatusTool().execute(
            ToolContext(
                user_input=(
                    "Knowledge base status"
                )
            )
        )

    assert result.success is True

    assert result.data[
        "document_count"
    ] == 0

    assert result.data[
        "source_count"
    ] == 0

    assert result.data[
        "sources"
    ] == []

    assert (
        "empty"
        in result.content.lower()
    )


def test_knowledge_sync_tool_metadata() -> None:
    tool = KnowledgeSyncTool()

    assert tool.name == "knowledge_sync"

    assert (
        "synchronize"
        in tool.description.lower()
    )


def test_knowledge_sync_reports_changes() -> None:
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
        inserted_chunks=4,
        deleted_chunks=2,
        has_changes=True,
    )

    with patch(
        "src.knowledge_tools.synchronize_knowledge_base",
        return_value=sync_result,
    ):
        result = KnowledgeSyncTool().execute(
            ToolContext(
                user_input=(
                    "Bilgi tabanını güncelle."
                )
            )
        )

    assert result.success is True

    assert result.data[
        "new_files"
    ] == [
        "new.txt",
    ]

    assert result.data[
        "modified_files"
    ] == [
        "modified.txt",
    ]

    assert result.data[
        "deleted_files"
    ] == [
        "deleted.txt",
    ]

    assert result.data[
        "inserted_chunks"
    ] == 4

    assert result.data[
        "deleted_chunks"
    ] == 2

    assert result.data[
        "has_changes"
    ] is True


def test_knowledge_sync_reports_no_changes() -> None:
    sync_result = SimpleNamespace(
        new_files=[],
        modified_files=[],
        deleted_files=[],
        unchanged_files=[
            "stm32_notes.txt",
            "pid_notes.txt",
        ],
        inserted_chunks=0,
        deleted_chunks=0,
        has_changes=False,
    )

    with patch(
        "src.knowledge_tools.synchronize_knowledge_base",
        return_value=sync_result,
    ):
        result = KnowledgeSyncTool().execute(
            ToolContext(
                user_input=(
                    "Knowledge base'i senkronize et."
                )
            )
        )

    assert result.success is True

    assert result.data[
        "has_changes"
    ] is False

    assert result.data[
        "inserted_chunks"
    ] == 0

    assert (
        "already up to date"
        in result.content.lower()
    )


def test_all_knowledge_tools_can_share_registry() -> None:
    assistant = create_assistant()

    registry = ToolRegistry(
        tools=[
            KnowledgeSearchTool(
                assistant
            ),
            KnowledgeStatusTool(),
            KnowledgeSyncTool(),
        ]
    )

    assert registry.size == 3

    assert registry.contains(
        "knowledge_search"
    )

    assert registry.contains(
        "knowledge_status"
    )

    assert registry.contains(
        "knowledge_sync"
    )

    assert registry.describe_tools() == [
        {
            "name": "knowledge_search",
            "description": (
                "Search the indexed local knowledge base "
                "and generate an evidence-grounded answer."
            ),
        },
        {
            "name": "knowledge_status",
            "description": (
                "Inspect the indexed local knowledge base "
                "and report document and source statistics."
            ),
        },
        {
            "name": "knowledge_sync",
            "description": (
                "Synchronize the configured local knowledge "
                "base with its source folder."
            ),
        },
    ]