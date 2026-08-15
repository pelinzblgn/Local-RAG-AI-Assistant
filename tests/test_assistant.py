from unittest.mock import MagicMock, patch

from src.assistant import RAGAssistant
from src.prompts import DEFAULT_FALLBACK_MESSAGE


SAMPLE_DOCUMENTS = [
    {
        "id": 1,
        "content": "PWM motor hızını kontrol eder.",
        "source": "stm32.txt",
        "score": 0.91,
    },
    {
        "id": 2,
        "content": "PID hata değerine göre düzeltme yapar.",
        "source": "pid.txt",
        "score": 0.83,
    },
]

def test_warm_up_delegates_to_local_llm() -> None:
    mock_llm = MagicMock()

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    assistant.warm_up()

    mock_llm.warm_up.assert_called_once()

def test_answer_returns_generated_response_and_sources() -> None:
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Test cevabı"

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=SAMPLE_DOCUMENTS,
    ):
        response = assistant.answer(
            "Motor hızı nasıl kontrol edilir?"
        )

    assert response["answer"] == "Test cevabı"
    assert response["sources"] == [
        "stm32.txt",
        "pid.txt",
    ]
    assert response["retrieved_documents"] == SAMPLE_DOCUMENTS

    mock_llm.generate.assert_called_once()


def test_empty_retrieval_returns_fallback_without_llm() -> None:
    mock_llm = MagicMock()

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=[],
    ):
        response = assistant.answer(
            "Belgede olmayan soru"
        )

    assert response["answer"] == DEFAULT_FALLBACK_MESSAGE
    assert response["sources"] == []
    assert response["retrieved_documents"] == []

    mock_llm.generate.assert_not_called()


def test_duplicate_sources_are_removed() -> None:
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Cevap"

    documents = [
        {
            "id": 1,
            "content": "Birinci chunk",
            "source": "same.txt",
            "score": 0.90,
        },
        {
            "id": 2,
            "content": "İkinci chunk",
            "source": "same.txt",
            "score": 0.80,
        },
    ]

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=documents,
    ):
        response = assistant.answer("Test")

    assert response["sources"] == ["same.txt"]


def test_question_whitespace_is_removed() -> None:
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Cevap"

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=SAMPLE_DOCUMENTS,
    ) as mocked_retrieval:
        assistant.answer(
            "   PWM nedir?   "
        )

    mocked_retrieval.assert_called_once_with(
        query="PWM nedir?",
        top_k=3,
        minimum_score=0.20,
    )


def test_empty_question_raises_error() -> None:
    assistant = RAGAssistant(
        local_llm=MagicMock(),
    )

    try:
        assistant.answer("   ")
    except ValueError as error:
        assert "Question cannot be empty" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for empty question."
    )


def test_invalid_top_k_raises_error() -> None:
    try:
        RAGAssistant(
            local_llm=MagicMock(),
            top_k=0,
        )
    except ValueError as error:
        assert "top_k" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for invalid top_k."
    )


def test_close_unloads_llm() -> None:
    mock_llm = MagicMock()

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    assistant.close()

    mock_llm.unload.assert_called_once()

def test_successful_answer_is_saved_to_memory() -> None:
    mock_llm = MagicMock()
    mock_llm.generate.return_value = (
        "STM32 bir mikrodenetleyici ailesidir."
    )

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=SAMPLE_DOCUMENTS,
    ):
        assistant.answer(
            "STM32 nedir?"
        )

    assert assistant.memory_size == 1

    history = assistant.get_conversation_history()

    assert "STM32 nedir?" in history
    assert (
        "STM32 bir mikrodenetleyici ailesidir."
        in history
    )


def test_previous_question_is_used_for_followup_retrieval() -> None:
    mock_llm = MagicMock()

    mock_llm.generate.side_effect = [
        "STM32 bir mikrodenetleyicidir.",
        "PWM motor kontrolünde kullanılabilir.",
    ]

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=SAMPLE_DOCUMENTS,
    ) as mocked_retrieval:
        assistant.answer(
            "STM32 nedir?"
        )

        assistant.answer(
            "Peki PWM ne işe yarar?"
        )

    second_query = (
        mocked_retrieval.call_args_list[1]
        .kwargs["query"]
    )

    assert "STM32 nedir?" in second_query
    assert "Peki PWM ne işe yarar?" in second_query


def test_clear_memory_removes_conversation_history() -> None:
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Cevap"

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=SAMPLE_DOCUMENTS,
    ):
        assistant.answer(
            "PID nedir?"
        )

    assert assistant.memory_size == 1

    assistant.clear_memory()

    assert assistant.memory_size == 0
    assert assistant.get_conversation_history() == ""


def test_fallback_response_is_not_saved_to_memory() -> None:
    mock_llm = MagicMock()

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=[],
    ):
        response = assistant.answer(
            "Belgelerde olmayan bilgi"
        )

    assert response["answer"] == DEFAULT_FALLBACK_MESSAGE
    assert assistant.memory_size == 0

    mock_llm.generate.assert_not_called()
def run_tests() -> None:
    tests = [
        test_answer_returns_generated_response_and_sources,
        test_empty_retrieval_returns_fallback_without_llm,
        test_duplicate_sources_are_removed,
        test_question_whitespace_is_removed,
        test_empty_question_raises_error,
        test_invalid_top_k_raises_error,
        test_close_unloads_llm,
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