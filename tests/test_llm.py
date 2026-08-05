from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.llm import (
    DEFAULT_SYSTEM_PROMPT,
    LocalLLM,
    _clean_prompt,
)


def test_clean_prompt_strips_whitespace() -> None:
    result = _clean_prompt(
        "   Merhaba   ",
        "Prompt",
    )

    assert result == "Merhaba"


def test_clean_prompt_rejects_empty_text() -> None:
    try:
        _clean_prompt(
            "   ",
            "Prompt",
        )
    except ValueError as error:
        assert "cannot be empty" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for empty prompt."
    )


def test_clean_prompt_rejects_non_string() -> None:
    try:
        _clean_prompt(
            123,  # type: ignore[arg-type]
            "Prompt",
        )
    except TypeError as error:
        assert "must be a string" in str(error)
        return

    raise AssertionError(
        "Expected TypeError for non-string prompt."
    )


def test_empty_model_alias_raises_error() -> None:
    try:
        LocalLLM(model_alias="   ")
    except ValueError as error:
        assert "Model alias cannot be empty" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for empty model alias."
    )


def test_generate_returns_clean_answer() -> None:
    llm = LocalLLM()

    mock_client = MagicMock()

    mock_client.complete_chat.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="  Test cevabı  "
                )
            )
        ]
    )

    with patch.object(
        llm,
        "_ensure_client",
        return_value=mock_client,
    ):
        answer = llm.generate(
            prompt="Test sorusu",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )

    assert answer == "Test cevabı"


def test_generate_sends_system_and_user_messages() -> None:
    llm = LocalLLM()

    mock_client = MagicMock()

    mock_client.complete_chat.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Cevap"
                )
            )
        ]
    )

    with patch.object(
        llm,
        "_ensure_client",
        return_value=mock_client,
    ):
        llm.generate(
            prompt="Kullanıcı sorusu",
            system_prompt="Sistem talimatı",
        )

    mock_client.complete_chat.assert_called_once_with(
        [
            {
                "role": "system",
                "content": "Sistem talimatı",
            },
            {
                "role": "user",
                "content": "Kullanıcı sorusu",
            },
        ]
    )


def test_no_choices_raises_error() -> None:
    llm = LocalLLM()

    mock_client = MagicMock()
    mock_client.complete_chat.return_value = SimpleNamespace(
        choices=[]
    )

    with patch.object(
        llm,
        "_ensure_client",
        return_value=mock_client,
    ):
        try:
            llm.generate(
                prompt="Soru",
            )
        except RuntimeError as error:
            assert "no response choices" in str(error)
            return

    raise AssertionError(
        "Expected RuntimeError for missing choices."
    )


def test_empty_answer_raises_error() -> None:
    llm = LocalLLM()

    mock_client = MagicMock()

    mock_client.complete_chat.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="   "
                )
            )
        ]
    )

    with patch.object(
        llm,
        "_ensure_client",
        return_value=mock_client,
    ):
        try:
            llm.generate(
                prompt="Soru",
            )
        except RuntimeError as error:
            assert "empty response" in str(error)
            return

    raise AssertionError(
        "Expected RuntimeError for empty model response."
    )


def run_tests() -> None:
    tests = [
        test_clean_prompt_strips_whitespace,
        test_clean_prompt_rejects_empty_text,
        test_clean_prompt_rejects_non_string,
        test_empty_model_alias_raises_error,
        test_generate_returns_clean_answer,
        test_generate_sends_system_and_user_messages,
        test_no_choices_raises_error,
        test_empty_answer_raises_error,
    ]

    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print(f"PASS: {test.__name__}")
        except Exception as error:
            print(f"FAIL: {test.__name__}")
            print(f"      {error}")

    print("-" * 50)
    print(
        f"Sonuç: {passed}/{len(tests)} "
        "test başarılı."
    )


if __name__ == "__main__":
    run_tests()