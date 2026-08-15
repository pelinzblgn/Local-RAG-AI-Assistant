from unittest.mock import MagicMock

from src.cli import (
    _format_retrieved_documents,
    run_chat_session,
)


def test_chat_session_answers_question_and_exits() -> None:
    assistant = MagicMock()

    assistant.answer.return_value = {
        "answer": "PWM motor hızını kontrol eder.",
        "sources": ["stm32.txt"],
        "retrieved_documents": [],
    }

    inputs = iter(
        [
            "PWM nedir?",
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
        "PWM nedir?"
    )

    assert any(
        "PWM motor hızını kontrol eder." in output
        for output in outputs
    )

    assert outputs[-1] == "Sohbet sonlandırıldı."


def test_empty_question_is_rejected() -> None:
    assistant = MagicMock()

    inputs = iter(
        [
            "   ",
            "q",
        ]
    )

    outputs: list[str] = []

    run_chat_session(
        assistant=assistant,
        input_function=lambda _: next(inputs),
        output_function=outputs.append,
    )

    assistant.answer.assert_not_called()

    assert any(
        "boş olmayan bir soru" in output
        for output in outputs
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

    assert outputs[-1] == "Sohbet sonlandırıldı."


def test_assistant_error_does_not_close_session() -> None:
    assistant = MagicMock()

    assistant.answer.side_effect = [
        RuntimeError("Test hatası"),
        {
            "answer": "İkinci soru başarılı.",
            "sources": [],
            "retrieved_documents": [],
        },
    ]

    inputs = iter(
        [
            "Birinci soru",
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

    assert assistant.answer.call_count == 2

    assert any(
        "Test hatası" in output
        for output in outputs
    )

    assert any(
        "İkinci soru başarılı." in output
        for output in outputs
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

    assert any(
        "hafızası temizlendi" in output
        for output in outputs
    )


def test_history_command_shows_history() -> None:
    assistant = MagicMock()

    assistant.get_conversation_history.return_value = (
        "[Konuşma 1]\n"
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

    assistant.get_conversation_history.assert_called_once()

    assert any(
        "STM32 nedir?" in output
        for output in outputs
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

    assert any(
        "Konuşma geçmişi boş" in output
        for output in outputs
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
            "source": "stm32_notes.txt",
            "score": 0.98765,
        },
        {
            "source": "pid_notes.txt",
            "score": 0.87654,
        },
    ]

    lines = _format_retrieved_documents(
        documents
    )

    output = "\n".join(lines)

    assert "RETRIEVED DOCUMENTS" in output
    assert "stm32_notes.txt" in output
    assert "pid_notes.txt" in output
    assert "0.9877" in output
    assert "0.8765" in output


def test_format_retrieved_documents_handles_empty_list() -> None:
    lines = _format_retrieved_documents([])

    assert lines == [
        "Retrieved document bulunamadı."
    ]


def test_chat_session_displays_retrieval_metadata() -> None:
    assistant = MagicMock()

    assistant.answer.return_value = {
        "answer": "Test cevabı",
        "sources": ["stm32_notes.txt"],
        "retrieved_documents": [
            {
                "id": 1,
                "content": "STM32 içeriği",
                "source": "stm32_notes.txt",
                "score": 0.91234,
            }
        ],
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

    combined_output = "\n".join(outputs)

    assert "RETRIEVED DOCUMENTS" in combined_output
    assert "stm32_notes.txt" in combined_output
    assert "0.9123" in combined_output