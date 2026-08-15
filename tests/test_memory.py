from src.memory import ConversationMemory


def test_memory_starts_empty() -> None:
    memory = ConversationMemory()

    assert memory.size == 0
    assert memory.is_empty


def test_add_turn_stores_conversation() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        question="STM32 nedir?",
        answer="STM32 bir mikrodenetleyici ailesidir.",
    )

    turns = memory.get_turns()

    assert memory.size == 1
    assert turns[0]["question"] == "STM32 nedir?"
    assert (
        turns[0]["answer"]
        == "STM32 bir mikrodenetleyici ailesidir."
    )


def test_memory_removes_oldest_turn() -> None:
    memory = ConversationMemory(
        max_turns=2,
        retrieval_turns=1,
    )

    memory.add_turn("Soru 1", "Cevap 1")
    memory.add_turn("Soru 2", "Cevap 2")
    memory.add_turn("Soru 3", "Cevap 3")

    turns = memory.get_turns()

    assert len(turns) == 2
    assert turns[0]["question"] == "Soru 2"
    assert turns[1]["question"] == "Soru 3"


def test_history_text_contains_turns() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        "PID nedir?",
        "PID bir kontrol algoritmasıdır.",
    )

    history = memory.build_history_text()

    assert "PID nedir?" in history
    assert "PID bir kontrol algoritmasıdır." in history
    assert "Kullanıcı:" in history
    assert "Asistan:" in history


def test_retrieval_query_without_history_is_question() -> None:
    memory = ConversationMemory()

    result = memory.build_retrieval_query(
        "PWM nedir?"
    )

    assert result == "PWM nedir?"


def test_retrieval_query_contains_recent_context() -> None:
    memory = ConversationMemory(
        max_turns=5,
        retrieval_turns=2,
    )

    memory.add_turn(
        "STM32 nedir?",
        "Bir mikrodenetleyici ailesidir.",
    )

    memory.add_turn(
        "PWM ne işe yarar?",
        "Motor hızını kontrol edebilir.",
    )

    result = memory.build_retrieval_query(
        "Peki bunun avantajı nedir?"
    )

    assert "STM32 nedir?" in result
    assert "PWM ne işe yarar?" in result
    assert "Peki bunun avantajı nedir?" in result


def test_clear_removes_history() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        "PID nedir?",
        "Bir kontrol algoritmasıdır.",
    )

    memory.clear()

    assert memory.size == 0
    assert memory.is_empty


def test_invalid_max_turns_raises_error() -> None:
    try:
        ConversationMemory(max_turns=0)
    except ValueError as error:
        assert "max_turns" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for invalid max_turns."
    )


def test_retrieval_turns_cannot_exceed_max_turns() -> None:
    try:
        ConversationMemory(
            max_turns=2,
            retrieval_turns=3,
        )
    except ValueError as error:
        assert "retrieval_turns" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for retrieval_turns."
    )