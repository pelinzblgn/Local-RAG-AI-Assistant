from unittest.mock import MagicMock, patch

from src.cli import (
    _format_confidence,
    _format_query_rewrite,
    _format_retrieved_documents,
    format_sync_result,
    run_chat_session,
)
from src.file_sync import SyncResult


def _default_query_rewrite(
    query: str = "STM32 nedir?",
) -> dict:
    """Return reusable non-rewritten query metadata."""

    return {
        "original_query": query,
        "retrieval_query": query,
        "was_rewritten": False,
        "reason": "The query appears to be self-contained.",
    }


def test_chat_session_answers_question_and_exits() -> None:
    assistant = MagicMock()

    assistant.answer.return_value = {
        "answer": "STM32 bir mikrodenetleyicidir.",
        "sources": ["stm32.txt"],
        "retrieved_documents": [
            {
                "id": 1,
                "content": "STM32 bir mikrodenetleyicidir.",
                "source": "stm32.txt",
                "score": 0.82,
            }
        ],
        "confidence": {},
        "query_rewrite": _default_query_rewrite(),
    }

    inputs = iter(
        [
            "STM32 nedir?",
            "exit",
        ]
    )

    outputs: list[str] = []

    run_chat_session(
        assistant=assistant,
        input_function=lambda _: next(inputs),
        output_function=outputs.append,
    )

    assistant.answer.assert_called_once_with(
        "STM32 nedir?"
    )

    combined_output = "\n".join(
        outputs
    )

    assert (
        "STM32 bir mikrodenetleyicidir."
        in combined_output
    )

    assert "stm32.txt" in combined_output
    assert "0.8200" in combined_output

    assert (
        outputs[-1]
        == "Sohbet sonlandırıldı."
    )


def test_empty_question_is_rejected() -> None:
    assistant = MagicMock()

    inputs = iter(
        [
            "   ",
            "exit",
        ]
    )

    outputs: list[str] = []

    run_chat_session(
        assistant=assistant,
        input_function=lambda _: next(inputs),
        output_function=outputs.append,
    )

    assistant.answer.assert_not_called()

    assert (
        "Lütfen boş olmayan bir soru gir."
        in outputs
    )


def test_turkish_exit_command_stops_session() -> None:
    assistant = MagicMock()

    outputs: list[str] = []

    run_chat_session(
        assistant=assistant,
        input_function=lambda _: "çıkış",
        output_function=outputs.append,
    )

    assistant.answer.assert_not_called()

    assert (
        outputs[-1]
        == "Sohbet sonlandırıldı."
    )


def test_assistant_error_does_not_close_session() -> None:
    assistant = MagicMock()

    assistant.answer.side_effect = [
        RuntimeError(
            "Test error"
        ),
        {
            "answer": "Başarılı cevap",
            "sources": [],
            "retrieved_documents": [],
            "confidence": {},
            "query_rewrite": {},
        },
    ]

    inputs = iter(
        [
            "İlk soru",
            "İkinci soru",
            "exit",
        ]
    )

    outputs: list[str] = []

    run_chat_session(
        assistant=assistant,
        input_function=lambda _: next(inputs),
        output_function=outputs.append,
    )

    assert (
        assistant.answer.call_count
        == 2
    )

    assert (
        "Bir hata oluştu: Test error"
        in outputs
    )


def test_clear_command_clears_memory() -> None:
    assistant = MagicMock()

    inputs = iter(
        [
            "/clear",
            "exit",
        ]
    )

    outputs: list[str] = []

    run_chat_session(
        assistant=assistant,
        input_function=lambda _: next(inputs),
        output_function=outputs.append,
    )

    assistant.clear_memory.assert_called_once()

    assistant.answer.assert_not_called()


def test_history_command_shows_history() -> None:
    assistant = MagicMock()

    assistant.get_conversation_history.return_value = (
        "Kullanıcı: STM32 nedir?\n"
        "Asistan: STM32 bir mikrodenetleyicidir."
    )

    inputs = iter(
        [
            "/history",
            "exit",
        ]
    )

    outputs: list[str] = []

    run_chat_session(
        assistant=assistant,
        input_function=lambda _: next(inputs),
        output_function=outputs.append,
    )

    combined_output = "\n".join(
        outputs
    )

    assert (
        "KONUŞMA GEÇMİŞİ"
        in combined_output
    )

    assert (
        "STM32 nedir?"
        in combined_output
    )


def test_history_command_handles_empty_history() -> None:
    assistant = MagicMock()

    assistant.get_conversation_history.return_value = ""

    inputs = iter(
        [
            "/history",
            "exit",
        ]
    )

    outputs: list[str] = []

    run_chat_session(
        assistant=assistant,
        input_function=lambda _: next(inputs),
        output_function=outputs.append,
    )

    assert (
        "Konuşma geçmişi boş."
        in outputs
    )


def test_cli_commands_do_not_call_answer() -> None:
    assistant = MagicMock()

    assistant.get_conversation_history.return_value = ""

    inputs = iter(
        [
            "/history",
            "/clear",
            "exit",
        ]
    )

    outputs: list[str] = []

    run_chat_session(
        assistant=assistant,
        input_function=lambda _: next(inputs),
        output_function=outputs.append,
    )

    assistant.answer.assert_not_called()


def test_format_retrieved_documents_shows_sources_and_scores() -> None:
    documents = [
        {
            "source": "stm32.txt",
            "score": 0.81234,
        },
        {
            "source": "pid.txt",
            "score": 0.71234,
        },
    ]

    output = "\n".join(
        _format_retrieved_documents(
            documents
        )
    )

    assert (
        "RETRIEVED DOCUMENTS"
        in output
    )

    assert "stm32.txt" in output
    assert "pid.txt" in output

    assert "0.8123" in output
    assert "0.7123" in output


def test_format_retrieved_documents_handles_empty_list() -> None:
    lines = _format_retrieved_documents(
        []
    )

    assert lines == [
        "Retrieved document bulunamadı."
    ]


def test_chat_session_displays_retrieval_metadata() -> None:
    assistant = MagicMock()

    assistant.answer.return_value = {
        "answer": "STM32 cevabı",
        "sources": ["stm32.txt"],
        "retrieved_documents": [
            {
                "id": 1,
                "content": "STM32 content",
                "source": "stm32.txt",
                "score": 0.82,
            }
        ],
        "confidence": {},
        "query_rewrite": _default_query_rewrite(),
    }

    inputs = iter(
        [
            "STM32 nedir?",
            "exit",
        ]
    )

    outputs: list[str] = []

    run_chat_session(
        assistant=assistant,
        input_function=lambda _: next(inputs),
        output_function=outputs.append,
    )

    combined_output = "\n".join(
        outputs
    )

    assert (
        "RETRIEVED DOCUMENTS"
        in combined_output
    )

    assert "stm32.txt" in combined_output
    assert "0.8200" in combined_output


def test_format_sync_result_shows_file_changes() -> None:
    result = SyncResult(
        new_files=("new.txt",),
        modified_files=("modified.txt",),
        deleted_files=("deleted.txt",),
        unchanged_files=("unchanged.txt",),
        inserted_chunks=2,
        deleted_chunks=1,
    )

    output = "\n".join(
        format_sync_result(
            result
        )
    )

    assert "new.txt" in output
    assert "modified.txt" in output
    assert "deleted.txt" in output
    assert "unchanged.txt" in output

    assert (
        "Inserted chunks: 2"
        in output
    )

    assert (
        "Deleted chunks : 1"
        in output
    )


def test_format_sync_result_handles_no_changes() -> None:
    result = SyncResult(
        new_files=(),
        modified_files=(),
        deleted_files=(),
        unchanged_files=("stm32.txt",),
        inserted_chunks=0,
        deleted_chunks=0,
    )

    output = "\n".join(
        format_sync_result(
            result
        )
    )

    assert (
        "already up to date"
        in output
    )


def test_sync_command_runs_synchronization() -> None:
    assistant = MagicMock()

    result = SyncResult(
        new_files=("new.txt",),
        modified_files=(),
        deleted_files=(),
        unchanged_files=(),
        inserted_chunks=1,
        deleted_chunks=0,
    )

    inputs = iter(
        [
            "/sync",
            "exit",
        ]
    )

    outputs: list[str] = []

    with patch(
        "src.cli.synchronize_knowledge_base",
        return_value=result,
    ) as mocked_sync:
        run_chat_session(
            assistant=assistant,
            input_function=lambda _: next(inputs),
            output_function=outputs.append,
        )

    mocked_sync.assert_called_once()

    assistant.answer.assert_not_called()


def test_sync_command_handles_no_changes() -> None:
    assistant = MagicMock()

    result = SyncResult(
        new_files=(),
        modified_files=(),
        deleted_files=(),
        unchanged_files=("stm32.txt",),
        inserted_chunks=0,
        deleted_chunks=0,
    )

    inputs = iter(
        [
            "/sync",
            "exit",
        ]
    )

    outputs: list[str] = []

    with patch(
        "src.cli.synchronize_knowledge_base",
        return_value=result,
    ):
        run_chat_session(
            assistant=assistant,
            input_function=lambda _: next(inputs),
            output_function=outputs.append,
        )

    assert (
        "already up to date"
        in "\n".join(outputs)
    )


def test_sync_command_handles_error() -> None:
    assistant = MagicMock()

    inputs = iter(
        [
            "/sync",
            "exit",
        ]
    )

    outputs: list[str] = []

    with patch(
        "src.cli.synchronize_knowledge_base",
        side_effect=RuntimeError(
            "Sync failed"
        ),
    ):
        run_chat_session(
            assistant=assistant,
            input_function=lambda _: next(inputs),
            output_function=outputs.append,
        )

    assert (
        "Sync failed"
        in "\n".join(outputs)
    )


def test_sync_command_does_not_clear_memory() -> None:
    assistant = MagicMock()

    result = SyncResult(
        new_files=(),
        modified_files=(),
        deleted_files=(),
        unchanged_files=(),
        inserted_chunks=0,
        deleted_chunks=0,
    )

    inputs = iter(
        [
            "/sync",
            "exit",
        ]
    )

    outputs: list[str] = []

    with patch(
        "src.cli.synchronize_knowledge_base",
        return_value=result,
    ):
        run_chat_session(
            assistant=assistant,
            input_function=lambda _: next(inputs),
            output_function=outputs.append,
        )

    assistant.clear_memory.assert_not_called()


def test_format_confidence_displays_high_confidence() -> None:
    confidence = {
        "level": "high",
        "is_confident": True,
        "top_score": 0.81234,
        "second_score": 0.51234,
        "score_gap": 0.30,
        "evidence_coverage": 0.75,
        "selected_count": 1,
        "total_count": 3,
        "filtered_count": 2,
        "reason": "Strong retrieval evidence.",
    }

    output = "\n".join(
        _format_confidence(
            confidence
        )
    )

    assert (
        "Level            : HIGH"
        in output
    )

    assert (
        "Top Score        : 0.8123"
        in output
    )

    assert (
        "Second Score     : 0.5123"
        in output
    )

    assert (
        "Score Gap        : 0.3000"
        in output
    )

    assert (
        "Evidence Coverage: 75.00%"
        in output
    )

    assert (
        "Selected Context  : 1/3"
        in output
    )

    assert (
        "Filtered Noise    : 2"
        in output
    )


def test_format_confidence_handles_missing_second_result() -> None:
    confidence = {
        "level": "high",
        "is_confident": True,
        "top_score": 0.90,
        "second_score": None,
        "score_gap": None,
        "evidence_coverage": 1.0,
        "selected_count": 1,
        "total_count": 1,
        "filtered_count": 0,
        "reason": "Single strong result.",
    }

    output = "\n".join(
        _format_confidence(
            confidence
        )
    )

    assert (
        "Second Score     : N/A"
        in output
    )

    assert (
        "Score Gap        : N/A"
        in output
    )

    assert (
        "Evidence Coverage: 100.00%"
        in output
    )


def test_chat_session_displays_confidence_metadata() -> None:
    assistant = MagicMock()

    assistant.answer.return_value = {
        "answer": "STM32 cevabı",
        "sources": ["stm32.txt"],
        "retrieved_documents": [
            {
                "id": 1,
                "content": "STM32 content",
                "source": "stm32.txt",
                "score": 0.82,
            }
        ],
        "confidence": {
            "level": "high",
            "is_confident": True,
            "top_score": 0.82,
            "second_score": None,
            "score_gap": None,
            "evidence_coverage": 0.80,
            "selected_count": 1,
            "total_count": 1,
            "filtered_count": 0,
            "reason": "Strong retrieval evidence.",
        },
        "query_rewrite": _default_query_rewrite(),
    }

    inputs = iter(
        [
            "STM32 nedir?",
            "exit",
        ]
    )

    outputs: list[str] = []

    run_chat_session(
        assistant=assistant,
        input_function=lambda _: next(inputs),
        output_function=outputs.append,
    )

    combined_output = "\n".join(
        outputs
    )

    assert (
        "RETRIEVAL CONFIDENCE"
        in combined_output
    )

    assert (
        "Level            : HIGH"
        in combined_output
    )

    assert (
        "Top Score        : 0.8200"
        in combined_output
    )

    assert (
        "Evidence Coverage: 80.00%"
        in combined_output
    )


def test_chat_session_displays_trusted_sources() -> None:
    assistant = MagicMock()

    assistant.answer.return_value = {
        "answer": (
            "STM32 PWM motor hızını kontrol eder."
        ),
        "sources": [
            "stm32_notes.txt",
        ],
        "retrieved_documents": [],
        "confidence": {},
        "query_rewrite": {},
    }

    inputs = iter(
        [
            "PWM ne işe yarar?",
            "exit",
        ]
    )

    outputs: list[str] = []

    run_chat_session(
        assistant=assistant,
        input_function=lambda _: next(inputs),
        output_function=outputs.append,
    )

    combined_output = "\n".join(
        outputs
    )

    assert (
        "Kaynaklar:"
        in combined_output
    )

    assert (
        "- stm32_notes.txt"
        in combined_output
    )


def test_format_query_rewrite_displays_rewritten_query() -> None:
    metadata = {
        "original_query": (
            "Peki PWM ne işe yarar?"
        ),
        "retrieval_query": (
            "STM32 nedir? "
            "Peki PWM ne işe yarar?"
        ),
        "was_rewritten": True,
        "reason": (
            "The current query appears to be a follow-up."
        ),
    }

    output = "\n".join(
        _format_query_rewrite(
            metadata
        )
    )

    assert (
        "QUERY REWRITE"
        in output
    )

    assert (
        "Rewritten       : YES"
        in output
    )

    assert (
        "Original Query  : Peki PWM ne işe yarar?"
        in output
    )

    assert (
        "STM32 nedir?"
        in output
    )


def test_format_query_rewrite_displays_unchanged_query() -> None:
    metadata = {
        "original_query": "SQLite nedir?",
        "retrieval_query": "SQLite nedir?",
        "was_rewritten": False,
        "reason": (
            "The query appears to be self-contained."
        ),
    }

    output = "\n".join(
        _format_query_rewrite(
            metadata
        )
    )

    assert (
        "Rewritten       : NO"
        in output
    )

    assert (
        "Original Query  : SQLite nedir?"
        in output
    )

    assert (
        "Retrieval Query : SQLite nedir?"
        in output
    )


def test_format_query_rewrite_handles_empty_metadata() -> None:
    assert (
        _format_query_rewrite({})
        == []
    )


def test_chat_session_displays_query_rewrite_metadata() -> None:
    assistant = MagicMock()

    assistant.answer.return_value = {
        "answer": (
            "PWM motor hızını kontrol eder."
        ),
        "sources": [
            "stm32.txt",
        ],
        "retrieved_documents": [],
        "confidence": {},
        "query_rewrite": {
            "original_query": (
                "Peki PWM ne işe yarar?"
            ),
            "retrieval_query": (
                "STM32 nedir? "
                "Peki PWM ne işe yarar?"
            ),
            "was_rewritten": True,
            "reason": (
                "The current query appears to be a follow-up."
            ),
        },
    }

    inputs = iter(
        [
            "Peki PWM ne işe yarar?",
            "exit",
        ]
    )

    outputs: list[str] = []

    run_chat_session(
        assistant=assistant,
        input_function=lambda _: next(inputs),
        output_function=outputs.append,
    )

    combined_output = "\n".join(
        outputs
    )

    assert (
        "QUERY REWRITE"
        in combined_output
    )

    assert (
        "Rewritten       : YES"
        in combined_output
    )

    assert (
        "Original Query  : Peki PWM ne işe yarar?"
        in combined_output
    )

    assert (
        "STM32 nedir? Peki PWM ne işe yarar?"
        in combined_output
    )