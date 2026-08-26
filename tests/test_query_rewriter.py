import pytest

from src.query_rewriter import (
    QueryRewriteResult,
    rewrite_query,
)


def test_standalone_query_without_history_is_unchanged() -> None:
    result = rewrite_query(
        "STM32 nedir?"
    )

    assert isinstance(
        result,
        QueryRewriteResult,
    )

    assert (
        result.original_query
        == "STM32 nedir?"
    )

    assert (
        result.rewritten_query
        == "STM32 nedir?"
    )

    assert result.was_rewritten is False


def test_follow_up_query_uses_previous_question() -> None:
    history = (
        "Kullanıcı: STM32 nedir?"
    )

    result = rewrite_query(
        query="Peki PWM ne işe yarar?",
        conversation_history=history,
    )

    assert result.was_rewritten is True

    assert (
        result.rewritten_query
        == (
            "STM32 nedir? "
            "Peki PWM ne işe yarar?"
        )
    )


def test_reference_based_follow_up_is_rewritten() -> None:
    history = (
        "Kullanıcı: PID kontrol nedir?"
    )

    result = rewrite_query(
        query="Bunun avantajı nedir?",
        conversation_history=history,
    )

    assert result.was_rewritten is True

    assert (
        result.rewritten_query
        == (
            "PID kontrol nedir? "
            "Bunun avantajı nedir?"
        )
    )


def test_short_generic_follow_up_is_rewritten() -> None:
    history = (
        "Kullanıcı: SQLite nedir?"
    )

    result = rewrite_query(
        query="Avantajları neler?",
        conversation_history=history,
    )

    assert result.was_rewritten is True

    assert (
        "SQLite nedir?"
        in result.rewritten_query
    )


def test_generic_how_question_is_rewritten() -> None:
    history = (
        "Kullanıcı: PID kontrol nedir?"
    )

    result = rewrite_query(
        query="Nasıl çalışır?",
        conversation_history=history,
    )

    assert result.was_rewritten is True


def test_short_stm32_question_is_self_contained() -> None:
    history = (
        "Kullanıcı: Bunun avantajı nedir?"
    )

    result = rewrite_query(
        query="STM32 nedir?",
        conversation_history=history,
    )

    assert result.was_rewritten is False

    assert (
        result.rewritten_query
        == "STM32 nedir?"
    )


def test_short_pwm_question_is_self_contained() -> None:
    history = (
        "Kullanıcı: STM32 nedir?"
    )

    result = rewrite_query(
        query="PWM nedir?",
        conversation_history=history,
    )

    assert result.was_rewritten is False

    assert (
        result.rewritten_query
        == "PWM nedir?"
    )


def test_pid_control_question_is_self_contained() -> None:
    history = (
        "Kullanıcı: SQLite nedir?"
    )

    result = rewrite_query(
        query="PID kontrol nedir?",
        conversation_history=history,
    )

    assert result.was_rewritten is False


def test_sqlite_question_is_self_contained() -> None:
    history = (
        "Kullanıcı: Bunun avantajı nedir?"
    )

    query = (
        "SQLite neden yerel veri "
        "depolamada kullanılabilir?"
    )

    result = rewrite_query(
        query=query,
        conversation_history=history,
    )

    assert result.was_rewritten is False

    assert (
        result.rewritten_query
        == query
    )


def test_explicit_reference_to_stm32_is_rewritten() -> None:
    history = (
        "Kullanıcı: STM32 nedir?"
    )

    result = rewrite_query(
        query=(
            "Bunun PWM özelliği "
            "ne işe yarar?"
        ),
        conversation_history=history,
    )

    assert result.was_rewritten is True

    assert (
        result.rewritten_query
        == (
            "STM32 nedir? "
            "Bunun PWM özelliği ne işe yarar?"
        )
    )


def test_latest_user_question_is_used() -> None:
    history = (
        "Kullanıcı: STM32 nedir?\n"
        "Kullanıcı: PID nedir?"
    )

    result = rewrite_query(
        query="Peki bileşenleri neler?",
        conversation_history=history,
    )

    assert result.was_rewritten is True

    assert (
        "PID nedir?"
        in result.rewritten_query
    )

    assert (
        "STM32 nedir?"
        not in result.rewritten_query
    )


def test_duplicate_question_is_not_rewritten() -> None:
    history = (
        "Kullanıcı: PWM nedir?"
    )

    result = rewrite_query(
        query="PWM nedir?",
        conversation_history=history,
    )

    assert result.was_rewritten is False

    assert (
        result.rewritten_query
        == "PWM nedir?"
    )


def test_query_whitespace_is_normalized() -> None:
    result = rewrite_query(
        "   STM32     nedir?   "
    )

    assert (
        result.original_query
        == "STM32 nedir?"
    )

    assert (
        result.rewritten_query
        == "STM32 nedir?"
    )


def test_empty_query_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        rewrite_query(
            "   "
        )


def test_non_string_query_raises_error() -> None:
    with pytest.raises(
        TypeError,
        match="Query must be a string",
    ):
        rewrite_query(
            123  # type: ignore[arg-type]
        )


def test_non_string_history_raises_error() -> None:
    with pytest.raises(
        TypeError,
        match="Conversation history must be a string",
    ):
        rewrite_query(
            query="PWM nedir?",
            conversation_history=123,  # type: ignore[arg-type]
        )


def test_history_without_user_question_does_not_rewrite() -> None:
    history = (
        "Asistan: STM32 bir mikrodenetleyicidir."
    )

    result = rewrite_query(
        query="Peki PWM?",
        conversation_history=history,
    )

    assert result.was_rewritten is False


def test_english_follow_up_is_rewritten() -> None:
    history = (
        "User: What is STM32?"
    )

    result = rewrite_query(
        query="What about PWM?",
        conversation_history=history,
    )

    assert result.was_rewritten is True

    assert (
        "What is STM32?"
        in result.rewritten_query
    )


def test_english_standalone_query_is_not_rewritten() -> None:
    history = (
        "User: What is STM32?"
    )

    result = rewrite_query(
        query="What is SQLite?",
        conversation_history=history,
    )

    assert result.was_rewritten is False