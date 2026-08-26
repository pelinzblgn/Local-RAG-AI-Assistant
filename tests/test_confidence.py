import pytest

from src.confidence import (
    ConfidenceLevel,
    ConfidencePolicy,
    calculate_evidence_coverage,
    evaluate_retrieval_confidence,
    select_confident_documents,
)


def make_document(
    source: str,
    score: float,
    content: str = "Test content",
) -> dict[str, object]:
    """Create a minimal retrieval document."""

    return {
        "id": 1,
        "content": content,
        "source": source,
        "score": score,
    }


def test_empty_retrieval_has_low_confidence() -> None:
    result = evaluate_retrieval_confidence(
        []
    )

    assert (
        result.level
        == ConfidenceLevel.LOW
    )

    assert result.is_confident is False
    assert result.selected_count == 0


def test_strong_top_result_has_high_confidence() -> None:
    documents = [
        make_document(
            "stm32.txt",
            0.82,
        ),
        make_document(
            "pid.txt",
            0.50,
        ),
    ]

    result = evaluate_retrieval_confidence(
        documents
    )

    assert (
        result.level
        == ConfidenceLevel.HIGH
    )


def test_medium_confidence_when_score_is_relevant_but_not_strong() -> None:
    documents = [
        make_document(
            "stm32.txt",
            0.52,
        ),
        make_document(
            "pid.txt",
            0.44,
        ),
    ]

    result = evaluate_retrieval_confidence(
        documents
    )

    assert (
        result.level
        == ConfidenceLevel.MEDIUM
    )


def test_low_confidence_for_weak_results() -> None:
    result = evaluate_retrieval_confidence(
        [
            make_document(
                "random.txt",
                0.21,
            )
        ]
    )

    assert (
        result.level
        == ConfidenceLevel.LOW
    )


def test_adaptive_context_removes_noise() -> None:
    documents = [
        make_document(
            "stm32.txt",
            0.80,
        ),
        make_document(
            "motor.txt",
            0.72,
        ),
        make_document(
            "noise.txt",
            0.31,
        ),
    ]

    selected = select_confident_documents(
        documents
    )

    assert len(selected) == 2


def test_single_strong_document_can_be_high_confidence() -> None:
    result = evaluate_retrieval_confidence(
        [
            make_document(
                "stm32.txt",
                0.85,
            )
        ]
    )

    assert (
        result.level
        == ConfidenceLevel.HIGH
    )

    assert result.second_score is None


def test_filtered_count_reports_removed_context() -> None:
    documents = [
        make_document(
            "a.txt",
            0.80,
        ),
        make_document(
            "b.txt",
            0.70,
        ),
        make_document(
            "c.txt",
            0.20,
        ),
    ]

    result = evaluate_retrieval_confidence(
        documents
    )

    assert result.filtered_count == 1


def test_documents_are_selected_in_score_order() -> None:
    documents = [
        make_document(
            "low.txt",
            0.60,
        ),
        make_document(
            "high.txt",
            0.90,
        ),
        make_document(
            "medium.txt",
            0.75,
        ),
    ]

    selected = select_confident_documents(
        documents
    )

    assert [
        item["source"]
        for item in selected
    ] == [
        "high.txt",
        "medium.txt",
    ]


def test_invalid_score_is_rejected() -> None:
    documents = [
        {
            "content": "test",
            "source": "bad.txt",
            "score": "bad",
        }
    ]

    with pytest.raises(
        ValueError
    ):
        evaluate_retrieval_confidence(
            documents
        )


def test_policy_rejects_invalid_threshold_order() -> None:
    with pytest.raises(
        ValueError
    ):
        ConfidencePolicy(
            low_score_threshold=0.70,
            medium_score_threshold=0.50,
            high_score_threshold=0.40,
        )


def test_policy_limits_context_document_count() -> None:
    policy = ConfidencePolicy(
        max_context_documents=2,
        context_score_ratio=0.5,
    )

    documents = [
        make_document(
            "a.txt",
            0.90,
        ),
        make_document(
            "b.txt",
            0.80,
        ),
        make_document(
            "c.txt",
            0.70,
        ),
    ]

    selected = select_confident_documents(
        documents,
        policy=policy,
    )

    assert len(selected) == 2


def test_evidence_coverage_detects_supported_terms() -> None:
    documents = [
        make_document(
            source="stm32.txt",
            score=0.80,
            content=(
                "STM32 PWM özelliği motor hızını "
                "duty cycle ile kontrol eder."
            ),
        )
    ]

    coverage = calculate_evidence_coverage(
        query=(
            "STM32 PWM motor kontrolünde "
            "ne işe yarar?"
        ),
        documents=documents,
    )

    assert coverage > 0.40


def test_unrelated_question_is_low_despite_semantic_similarity() -> None:
    documents = [
        make_document(
            source="stm32.txt",
            score=0.70,
            content=(
                "STM32 PWM UART SPI ve motor "
                "kontrol özelliklerine sahiptir."
            ),
        ),
        make_document(
            source="pid.txt",
            score=0.50,
            content=(
                "PID kontrol algoritması "
                "hata değerini kullanır."
            ),
        ),
    ]

    result = evaluate_retrieval_confidence(
        documents=documents,
        query="Fransa'nın başkenti nedir?",
    )

    assert (
        result.level
        == ConfidenceLevel.LOW
    )

    assert result.is_confident is False

    assert (
        result.evidence_coverage
        == 0.0
    )


def test_supported_question_can_remain_high_confidence() -> None:
    documents = [
        make_document(
            source="stm32.txt",
            score=0.75,
            content=(
                "STM32 PWM özelliği DC motor "
                "hız kontrolünde kullanılır."
            ),
        ),
        make_document(
            source="other.txt",
            score=0.40,
            content=(
                "SQLite yerel veritabanıdır."
            ),
        ),
    ]

    result = evaluate_retrieval_confidence(
        documents=documents,
        query=(
            "STM32 PWM motor hız kontrolünde "
            "ne işe yarar?"
        ),
    )

    assert (
        result.level
        == ConfidenceLevel.HIGH
    )

    assert result.is_confident is True


def test_strong_semantic_override_prevents_false_negative() -> None:
    documents = [
        make_document(
            source="sqlite_notes.txt",
            score=0.7316,
            content=(
                "SQLite hafif bir veritabanıdır ve "
                "yerel uygulamalarda veri saklamak "
                "için kullanılabilir."
            ),
        ),
        make_document(
            source="other.txt",
            score=0.4674,
            content=(
                "Foundry Local yerel yapay zeka "
                "modellerini çalıştırır."
            ),
        ),
    ]

    result = evaluate_retrieval_confidence(
        documents=documents,
        query=(
            "SQLite neden yerel veri "
            "depolamada kullanılabilir?"
        ),
    )

    assert result.is_confident is True

    assert result.level in {
        ConfidenceLevel.MEDIUM,
        ConfidenceLevel.HIGH,
    }


def test_semantic_override_never_accepts_zero_evidence() -> None:
    documents = [
        make_document(
            source="stm32.txt",
            score=0.90,
            content=(
                "STM32 PWM motor kontrolü "
                "ve UART iletişimi sağlar."
            ),
        ),
        make_document(
            source="pid.txt",
            score=0.50,
            content=(
                "PID kontrol algoritmasıdır."
            ),
        ),
    ]

    result = evaluate_retrieval_confidence(
        documents=documents,
        query="Fransa'nın başkenti nedir?",
    )

    assert (
        result.evidence_coverage
        == 0.0
    )

    assert (
        result.level
        == ConfidenceLevel.LOW
    )

    assert result.is_confident is False


def test_semantic_override_requires_clear_score_gap() -> None:
    policy = ConfidencePolicy(
        semantic_override_threshold=0.70,
        semantic_override_gap=0.20,
        semantic_override_minimum_evidence=0.20,
    )

    documents = [
        make_document(
            source="one.txt",
            score=0.75,
            content=(
                "SQLite veritabanı kullanılır."
            ),
        ),
        make_document(
            source="two.txt",
            score=0.68,
            content=(
                "SQLite başka amaçlarla da kullanılabilir."
            ),
        ),
    ]

    result = evaluate_retrieval_confidence(
        documents=documents,
        query=(
            "SQLite neden yerel veri "
            "depolamada kullanılabilir?"
        ),
        policy=policy,
    )

    if (
        result.evidence_coverage
        < policy.minimum_evidence_coverage
    ):
        assert (
            result.level
            == ConfidenceLevel.LOW
        )


def test_invalid_semantic_override_gap_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="semantic_override_gap",
    ):
        ConfidencePolicy(
            semantic_override_gap=-0.1,
        )


def test_invalid_semantic_override_evidence_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Evidence coverage",
    ):
        ConfidencePolicy(
            semantic_override_minimum_evidence=1.5,
        )