import pytest

from src.memory import (
    ConversationMemory,
)


def test_memory_starts_empty() -> None:
    memory = ConversationMemory()

    assert memory.size == 0
    assert memory.is_empty is True

    assert (
        memory.get_turns()
        == []
    )

    assert (
        memory.build_history_text()
        == ""
    )

    assert memory.last_user_query is None

    assert (
        memory.build_rewrite_context()
        == ""
    )


def test_add_turn_stores_conversation() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        question="STM32 nedir?",
        answer=(
            "STM32 bir mikrodenetleyici ailesidir."
        ),
    )

    assert memory.size == 1
    assert memory.is_empty is False

    turns = memory.get_turns()

    assert turns == [
        {
            "question": "STM32 nedir?",
            "answer": (
                "STM32 bir mikrodenetleyici ailesidir."
            ),
        }
    ]


def test_add_turn_strips_whitespace() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        question="   STM32 nedir?   ",
        answer="   Test cevabı   ",
    )

    turns = memory.get_turns()

    assert (
        turns[0]["question"]
        == "STM32 nedir?"
    )

    assert (
        turns[0]["answer"]
        == "Test cevabı"
    )


def test_add_turn_updates_last_user_query() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        question="STM32 nedir?",
        answer="STM32 cevabı",
    )

    assert (
        memory.last_user_query
        == "STM32 nedir?"
    )


def test_memory_respects_max_turns() -> None:
    memory = ConversationMemory(
        max_turns=2,
        retrieval_turns=1,
    )

    memory.add_turn(
        "Birinci soru",
        "Birinci cevap",
    )

    memory.add_turn(
        "İkinci soru",
        "İkinci cevap",
    )

    memory.add_turn(
        "Üçüncü soru",
        "Üçüncü cevap",
    )

    turns = memory.get_turns()

    assert len(turns) == 2

    assert (
        turns[0]["question"]
        == "İkinci soru"
    )

    assert (
        turns[1]["question"]
        == "Üçüncü soru"
    )


def test_get_turns_returns_copy() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        "STM32 nedir?",
        "STM32 cevabı",
    )

    copied_turns = (
        memory.get_turns()
    )

    copied_turns[0][
        "question"
    ] = "Değiştirildi"

    assert (
        memory.get_turns()[0]["question"]
        == "STM32 nedir?"
    )


def test_history_text_contains_turns() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        "STM32 nedir?",
        "STM32 cevabı",
    )

    memory.add_turn(
        "PWM nedir?",
        "PWM cevabı",
    )

    history = (
        memory.build_history_text()
    )

    assert (
        "[Konuşma 1]"
        in history
    )

    assert (
        "Kullanıcı: STM32 nedir?"
        in history
    )

    assert (
        "Asistan: STM32 cevabı"
        in history
    )

    assert (
        "[Konuşma 2]"
        in history
    )

    assert (
        "Kullanıcı: PWM nedir?"
        in history
    )


def test_record_user_query_tracks_latest_question() -> None:
    memory = ConversationMemory()

    memory.record_user_query(
        "SQLite nedir?"
    )

    assert (
        memory.last_user_query
        == "SQLite nedir?"
    )


def test_record_user_query_strips_whitespace() -> None:
    memory = ConversationMemory()

    memory.record_user_query(
        "   SQLite nedir?   "
    )

    assert (
        memory.last_user_query
        == "SQLite nedir?"
    )


def test_rewrite_context_uses_latest_user_query() -> None:
    memory = ConversationMemory()

    memory.record_user_query(
        (
            "SQLite neden yerel veri "
            "depolamada kullanılabilir?"
        )
    )

    assert (
        memory.build_rewrite_context()
        == (
            "Kullanıcı: "
            "SQLite neden yerel veri "
            "depolamada kullanılabilir?"
        )
    )


def test_new_user_query_replaces_previous_query_context() -> None:
    memory = ConversationMemory()

    memory.record_user_query(
        "STM32 nedir?"
    )

    memory.record_user_query(
        "SQLite nedir?"
    )

    assert (
        memory.last_user_query
        == "SQLite nedir?"
    )

    assert (
        memory.build_rewrite_context()
        == "Kullanıcı: SQLite nedir?"
    )


def test_latest_query_does_not_create_successful_turn() -> None:
    memory = ConversationMemory()

    memory.record_user_query(
        "Belgelerde olmayan soru"
    )

    assert memory.size == 0
    assert memory.is_empty is True

    assert (
        memory.build_history_text()
        == ""
    )

    assert (
        memory.last_user_query
        == "Belgelerde olmayan soru"
    )


def test_failed_query_context_is_separate_from_history() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        "STM32 nedir?",
        "STM32 cevabı",
    )

    memory.record_user_query(
        "Fransa'nın başkenti nedir?"
    )

    history = (
        memory.build_history_text()
    )

    assert (
        "STM32 nedir?"
        in history
    )

    assert (
        "Fransa'nın başkenti nedir?"
        not in history
    )

    assert (
        memory.last_user_query
        == "Fransa'nın başkenti nedir?"
    )


def test_build_retrieval_query_without_memory_returns_question() -> None:
    memory = ConversationMemory()

    result = (
        memory.build_retrieval_query(
            "PWM nedir?"
        )
    )

    assert (
        result
        == "PWM nedir?"
    )


def test_build_retrieval_query_uses_recent_questions() -> None:
    memory = ConversationMemory(
        max_turns=3,
        retrieval_turns=2,
    )

    memory.add_turn(
        "STM32 nedir?",
        "STM32 cevabı",
    )

    memory.add_turn(
        "PWM nedir?",
        "PWM cevabı",
    )

    result = (
        memory.build_retrieval_query(
            "Avantajı nedir?"
        )
    )

    assert (
        "STM32 nedir?"
        in result
    )

    assert (
        "PWM nedir?"
        in result
    )

    assert (
        "Avantajı nedir?"
        in result
    )


def test_retrieval_query_respects_retrieval_turn_limit() -> None:
    memory = ConversationMemory(
        max_turns=3,
        retrieval_turns=1,
    )

    memory.add_turn(
        "STM32 nedir?",
        "STM32 cevabı",
    )

    memory.add_turn(
        "SQLite nedir?",
        "SQLite cevabı",
    )

    result = (
        memory.build_retrieval_query(
            "Avantajı nedir?"
        )
    )

    assert (
        "SQLite nedir?"
        in result
    )

    assert (
        "STM32 nedir?"
        not in result
    )


def test_zero_retrieval_turns_returns_current_question() -> None:
    memory = ConversationMemory(
        max_turns=3,
        retrieval_turns=0,
    )

    memory.add_turn(
        "STM32 nedir?",
        "STM32 cevabı",
    )

    result = (
        memory.build_retrieval_query(
            "PWM nedir?"
        )
    )

    assert (
        result
        == "PWM nedir?"
    )


def test_clear_removes_all_memory_state() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        "STM32 nedir?",
        "STM32 cevabı",
    )

    memory.record_user_query(
        "SQLite nedir?"
    )

    memory.clear()

    assert memory.size == 0
    assert memory.is_empty is True
    assert memory.last_user_query is None

    assert (
        memory.get_turns()
        == []
    )

    assert (
        memory.build_history_text()
        == ""
    )

    assert (
        memory.build_rewrite_context()
        == ""
    )


def test_invalid_max_turns_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="max_turns",
    ):
        ConversationMemory(
            max_turns=0,
        )


def test_negative_retrieval_turns_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="retrieval_turns",
    ):
        ConversationMemory(
            max_turns=3,
            retrieval_turns=-1,
        )


def test_retrieval_turns_cannot_exceed_max_turns() -> None:
    with pytest.raises(
        ValueError,
        match="retrieval_turns",
    ):
        ConversationMemory(
            max_turns=2,
            retrieval_turns=3,
        )


def test_non_string_turn_question_raises_error() -> None:
    memory = ConversationMemory()

    with pytest.raises(
        TypeError,
        match="Question must be a string",
    ):
        memory.add_turn(
            123,  # type: ignore[arg-type]
            "Cevap",
        )


def test_non_string_turn_answer_raises_error() -> None:
    memory = ConversationMemory()

    with pytest.raises(
        TypeError,
        match="Answer must be a string",
    ):
        memory.add_turn(
            "Soru",
            123,  # type: ignore[arg-type]
        )


def test_empty_turn_question_raises_error() -> None:
    memory = ConversationMemory()

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        memory.add_turn(
            "   ",
            "Cevap",
        )


def test_empty_turn_answer_raises_error() -> None:
    memory = ConversationMemory()

    with pytest.raises(
        ValueError,
        match="Answer cannot be empty",
    ):
        memory.add_turn(
            "Soru",
            "   ",
        )


def test_non_string_recorded_query_raises_error() -> None:
    memory = ConversationMemory()

    with pytest.raises(
        TypeError,
        match="Question must be a string",
    ):
        memory.record_user_query(
            123  # type: ignore[arg-type]
        )


def test_empty_recorded_query_raises_error() -> None:
    memory = ConversationMemory()

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        memory.record_user_query(
            "   "
        )


def test_non_string_retrieval_question_raises_error() -> None:
    memory = ConversationMemory()

    with pytest.raises(
        TypeError,
        match="Question must be a string",
    ):
        memory.build_retrieval_query(
            123  # type: ignore[arg-type]
        )


def test_empty_retrieval_question_raises_error() -> None:
    memory = ConversationMemory()

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        memory.build_retrieval_query(
            "   "
        )