from unittest.mock import MagicMock, patch

import pytest

from src.assistant import (
    RAGAssistant,
    _is_fallback_answer,
)
from src.confidence import (
    ConfidenceLevel,
    ConfidencePolicy,
)
from src.prompts import DEFAULT_FALLBACK_MESSAGE


SAMPLE_DOCUMENTS = [
    {
        "id": 1,
        "content": (
            "STM32 bir mikrodenetleyici ailesidir. "
            "PWM motor hızını kontrol etmek için kullanılabilir."
        ),
        "source": "stm32.txt",
        "score": 0.91,
    },
    {
        "id": 2,
        "content": (
            "PID kontrol sistemi hata değerine göre "
            "motor çıkışına düzeltme uygular."
        ),
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
    mock_llm.generate.return_value = (
        "Motor hızı PWM ile kontrol edilir."
    )

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

    assert (
        response["answer"]
        == "Motor hızı PWM ile kontrol edilir."
    )

    assert response["sources"] == [
        "stm32.txt",
        "pid.txt",
    ]

    assert (
        response["retrieved_documents"]
        == SAMPLE_DOCUMENTS
    )

    assert (
        response["confidence"]["is_confident"]
        is True
    )

    assert (
        response["query_rewrite"]["was_rewritten"]
        is False
    )

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

    assert (
        response["answer"]
        == DEFAULT_FALLBACK_MESSAGE
    )

    assert response["sources"] == []

    assert (
        response["confidence"]["level"]
        == ConfidenceLevel.LOW.value
    )

    assert assistant.memory_size == 0

    mock_llm.generate.assert_not_called()


def test_low_confidence_retrieval_returns_fallback_without_llm() -> None:
    mock_llm = MagicMock()

    documents = [
        {
            "id": 1,
            "content": "STM32 teknik içeriği",
            "source": "weak.txt",
            "score": 0.24,
        },
    ]

    assistant = RAGAssistant(
        local_llm=mock_llm,
        minimum_score=0.0,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=documents,
    ):
        response = assistant.answer(
            "Fransa'nın başkenti nedir?"
        )

    assert (
        response["confidence"]["level"]
        == ConfidenceLevel.LOW.value
    )

    assert (
        response["answer"]
        == DEFAULT_FALLBACK_MESSAGE
    )

    assert response["sources"] == []

    assert assistant.memory_size == 0

    mock_llm.generate.assert_not_called()


def test_adaptive_context_filters_noisy_documents() -> None:
    mock_llm = MagicMock()

    mock_llm.generate.return_value = (
        "STM32 motor kontrolü yapabilir."
    )

    prompt_builder = MagicMock()

    prompt_builder.build.return_value = (
        "system prompt",
        "user prompt",
    )

    documents = [
        {
            "id": 1,
            "content": "STM32 PWM motor kontrolü",
            "source": "stm32.txt",
            "score": 0.90,
        },
        {
            "id": 2,
            "content": "STM32 timer ve motor kontrolü",
            "source": "timer.txt",
            "score": 0.78,
        },
        {
            "id": 3,
            "content": "RAG retrieval bilgisi",
            "source": "rag.txt",
            "score": 0.31,
        },
    ]

    assistant = RAGAssistant(
        local_llm=mock_llm,
        prompt_builder=prompt_builder,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=documents,
    ):
        response = assistant.answer(
            "STM32 motor kontrolü nasıl yapılır?"
        )

    selected_documents = (
        prompt_builder
        .build
        .call_args
        .kwargs["retrieved_documents"]
    )

    assert len(
        selected_documents
    ) == 2

    assert (
        response["confidence"]["filtered_count"]
        == 1
    )


def test_duplicate_sources_are_removed() -> None:
    mock_llm = MagicMock()

    mock_llm.generate.return_value = (
        "PWM motor hızını kontrol eder."
    )

    documents = [
        {
            "id": 1,
            "content": (
                "PWM motor hızını duty cycle "
                "değiştirerek kontrol eder."
            ),
            "source": "same.txt",
            "score": 0.90,
        },
        {
            "id": 2,
            "content": (
                "PWM sinyali motor kontrolünde "
                "kullanılabilir."
            ),
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
        response = assistant.answer(
            "PWM motor hızını nasıl kontrol eder?"
        )

    assert response["sources"] == [
        "same.txt",
    ]

    mock_llm.generate.assert_called_once()


def test_question_whitespace_is_removed() -> None:
    mock_llm = MagicMock()

    mock_llm.generate.return_value = (
        "PWM test cevabı"
    )

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


def test_non_string_question_raises_error() -> None:
    assistant = RAGAssistant(
        local_llm=MagicMock(),
    )

    with pytest.raises(
        TypeError,
        match="Question must be a string",
    ):
        assistant.answer(
            123  # type: ignore[arg-type]
        )


def test_empty_question_raises_error() -> None:
    assistant = RAGAssistant(
        local_llm=MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        assistant.answer(
            "   "
        )


def test_invalid_top_k_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        RAGAssistant(
            local_llm=MagicMock(),
            top_k=0,
        )


def test_invalid_minimum_score_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_score",
    ):
        RAGAssistant(
            local_llm=MagicMock(),
            minimum_score=1.5,
        )


def test_custom_confidence_policy_is_used() -> None:
    strict_policy = ConfidencePolicy(
        low_score_threshold=0.70,
        medium_score_threshold=0.80,
        high_score_threshold=0.90,
    )

    mock_llm = MagicMock()

    assistant = RAGAssistant(
        local_llm=mock_llm,
        confidence_policy=strict_policy,
        minimum_score=0.0,
    )

    documents = [
        {
            "id": 1,
            "content": "STM32 test bilgisi",
            "source": "test.txt",
            "score": 0.65,
        }
    ]

    with patch(
        "src.assistant.get_top_documents",
        return_value=documents,
    ):
        response = assistant.answer(
            "STM32 nedir?"
        )

    assert (
        response["confidence"]["level"]
        == ConfidenceLevel.LOW.value
    )

    mock_llm.generate.assert_not_called()


def test_confidence_policy_property_returns_active_policy() -> None:
    policy = ConfidencePolicy()

    assistant = RAGAssistant(
        local_llm=MagicMock(),
        confidence_policy=policy,
    )

    assert (
        assistant.confidence_policy
        is policy
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

    history = (
        assistant.get_conversation_history()
    )

    assert (
        "STM32 nedir?"
        in history
    )

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
        mocked_retrieval
        .call_args_list[1]
        .kwargs["query"]
    )

    assert (
        second_query
        == (
            "STM32 nedir? "
            "Peki PWM ne işe yarar?"
        )
    )


def test_standalone_question_is_not_rewritten() -> None:
    mock_llm = MagicMock()

    mock_llm.generate.side_effect = [
        "İlk cevap",
        "STM32 bir mikrodenetleyicidir.",
    ]

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=SAMPLE_DOCUMENTS,
    ) as mocked_retrieval:
        assistant.answer(
            "PWM nedir?"
        )

        assistant.answer(
            "STM32 nedir?"
        )

    second_query = (
        mocked_retrieval
        .call_args_list[1]
        .kwargs["query"]
    )

    assert (
        second_query
        == "STM32 nedir?"
    )


def test_clear_memory_removes_conversation_history() -> None:
    mock_llm = MagicMock()

    mock_llm.generate.return_value = (
        "PID kontrol algoritmasıdır."
    )

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

    assistant.clear_memory()

    assert assistant.memory_size == 0

    assert (
        assistant.get_conversation_history()
        == ""
    )


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

    assert (
        response["answer"]
        == DEFAULT_FALLBACK_MESSAGE
    )

    assert response["sources"] == []

    assert assistant.memory_size == 0

    mock_llm.generate.assert_not_called()


def test_low_confidence_response_is_not_saved_to_memory() -> None:
    mock_llm = MagicMock()

    documents = [
        {
            "id": 1,
            "content": "STM32 teknik içeriği",
            "source": "stm32.txt",
            "score": 0.50,
        }
    ]

    assistant = RAGAssistant(
        local_llm=mock_llm,
        minimum_score=0.0,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=documents,
    ):
        assistant.answer(
            "Fransa'nın başkenti nedir?"
        )

    assert assistant.memory_size == 0

    mock_llm.generate.assert_not_called()


def test_low_confidence_query_remains_latest_rewrite_context() -> None:
    mock_llm = MagicMock()

    weak_documents = [
        {
            "id": 1,
            "content": "SQLite yerel veritabanıdır.",
            "source": "sqlite.txt",
            "score": 0.20,
        }
    ]

    strong_documents = [
        {
            "id": 2,
            "content": (
                "SQLite yerel uygulamalarda "
                "veri saklamak için kullanılabilir."
            ),
            "source": "sqlite.txt",
            "score": 0.90,
        }
    ]

    mock_llm.generate.return_value = (
        "SQLite yerel depolama sağlar."
    )

    assistant = RAGAssistant(
        local_llm=mock_llm,
        minimum_score=0.0,
    )

    with patch(
        "src.assistant.get_top_documents",
        side_effect=[
            weak_documents,
            strong_documents,
        ],
    ) as mocked_retrieval:
        first_response = assistant.answer(
            (
                "SQLite neden yerel veri "
                "depolamada kullanılabilir?"
            )
        )

        assistant.answer(
            "Bunun avantajı nedir?"
        )

    assert (
        first_response["answer"]
        == DEFAULT_FALLBACK_MESSAGE
    )

    second_query = (
        mocked_retrieval
        .call_args_list[1]
        .kwargs["query"]
    )

    assert (
        second_query
        == (
            "SQLite neden yerel veri "
            "depolamada kullanılabilir? "
            "Bunun avantajı nedir?"
        )
    )


def test_exact_llm_fallback_returns_no_sources() -> None:
    mock_llm = MagicMock()

    mock_llm.generate.return_value = (
        DEFAULT_FALLBACK_MESSAGE
    )

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=SAMPLE_DOCUMENTS,
    ):
        response = assistant.answer(
            "STM32 nedir?"
        )

    assert (
        response["answer"]
        == DEFAULT_FALLBACK_MESSAGE
    )

    assert response["sources"] == []

    assert assistant.memory_size == 0

    mock_llm.generate.assert_called_once()


def test_llm_fallback_with_question_prefix_is_normalized() -> None:
    mock_llm = MagicMock()

    mock_llm.generate.return_value = (
        "SQLite neden yerel veri depolamada kullanılabilir?\n\n"
        f"{DEFAULT_FALLBACK_MESSAGE}"
    )

    documents = [
        {
            "id": 1,
            "content": (
                "SQLite yerel uygulamalarda "
                "kullanılan hafif bir veritabanıdır."
            ),
            "source": "sqlite_notes.txt",
            "score": 0.91,
        }
    ]

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=documents,
    ):
        response = assistant.answer(
            (
                "SQLite neden yerel veri "
                "depolamada kullanılabilir?"
            )
        )

    assert (
        response["answer"]
        == DEFAULT_FALLBACK_MESSAGE
    )

    assert response["sources"] == []

    assert assistant.memory_size == 0


def test_generated_fallback_query_is_still_available_for_followup() -> None:
    mock_llm = MagicMock()

    mock_llm.generate.side_effect = [
        DEFAULT_FALLBACK_MESSAGE,
        "SQLite avantaj cevabı",
    ]

    documents = [
        {
            "id": 1,
            "content": (
                "SQLite yerel veri saklama "
                "senaryolarında kullanılabilir."
            ),
            "source": "sqlite.txt",
            "score": 0.90,
        }
    ]

    assistant = RAGAssistant(
        local_llm=mock_llm,
    )

    with patch(
        "src.assistant.get_top_documents",
        return_value=documents,
    ) as mocked_retrieval:
        assistant.answer(
            "SQLite nedir?"
        )

        assistant.answer(
            "Bunun avantajı nedir?"
        )

    second_query = (
        mocked_retrieval
        .call_args_list[1]
        .kwargs["query"]
    )

    assert (
        second_query
        == (
            "SQLite nedir? "
            "Bunun avantajı nedir?"
        )
    )


def test_successful_answer_still_returns_trusted_sources() -> None:
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
        response = assistant.answer(
            "STM32 nedir?"
        )

    assert (
        response["answer"]
        != DEFAULT_FALLBACK_MESSAGE
    )

    assert set(
        response["sources"]
    ) == {
        "stm32.txt",
        "pid.txt",
    }

    assert assistant.memory_size == 1


def test_is_fallback_answer_detects_exact_message() -> None:
    assert (
        _is_fallback_answer(
            DEFAULT_FALLBACK_MESSAGE
        )
        is True
    )


def test_is_fallback_answer_detects_prefixed_message() -> None:
    answer = (
        "SQLite nedir?\n\n"
        f"{DEFAULT_FALLBACK_MESSAGE}"
    )

    assert (
        _is_fallback_answer(
            answer
        )
        is True
    )


def test_is_fallback_answer_rejects_normal_answer() -> None:
    assert (
        _is_fallback_answer(
            "SQLite hafif bir yerel veritabanıdır."
        )
        is False
    )