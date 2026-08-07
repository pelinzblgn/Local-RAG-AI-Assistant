from unittest.mock import MagicMock

from src.cli import run_chat_session


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
        "PWM motor hızını kontrol eder."
        in output
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
        "boş olmayan bir soru"
        in output
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
        "İkinci soru başarılı."
        in output
        for output in outputs
    )