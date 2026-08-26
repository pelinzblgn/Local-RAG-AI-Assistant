import math
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation import (
    EvaluationCase,
    EvaluationResult,
    RAGCaseType,
    RAGQualityCase,
    RAGQualityResult,
    calculate_average_latency,
    calculate_hit_rate,
    calculate_max_latency,
    calculate_mean_reciprocal_rank,
    calculate_median_latency,
    calculate_min_latency,
    calculate_quality_summary,
    calculate_reciprocal_rank,
    calculate_warm_average_latency,
    evaluate_rag_quality,
    evaluate_rag_quality_case,
    evaluate_retrieval_case,
)


def test_reciprocal_rank_first_result() -> None:
    result = calculate_reciprocal_rank(
        retrieved_sources=[
            "stm32.txt",
            "pid.txt",
        ],
        expected_sources=(
            "stm32.txt",
        ),
    )

    assert result == 1.0


def test_reciprocal_rank_second_result() -> None:
    result = calculate_reciprocal_rank(
        retrieved_sources=[
            "other.txt",
            "pid.txt",
        ],
        expected_sources=(
            "pid.txt",
        ),
    )

    assert result == 0.5


def test_reciprocal_rank_returns_zero_without_hit() -> None:
    result = calculate_reciprocal_rank(
        retrieved_sources=[
            "a.txt",
            "b.txt",
        ],
        expected_sources=(
            "c.txt",
        ),
    )

    assert result == 0.0


def test_evaluate_retrieval_case_detects_hit() -> None:
    case = EvaluationCase(
        question="STM32 nedir?",
        expected_sources=(
            "stm32_notes.txt",
        ),
    )

    with patch(
        "src.evaluation.get_top_documents",
        return_value=[
            {
                "id": 1,
                "content": "STM32 content",
                "source": "stm32_notes.txt",
                "score": 0.91,
            }
        ],
    ):
        result = evaluate_retrieval_case(
            case
        )

    assert result.hit is True

    assert (
        result.reciprocal_rank
        == 1.0
    )


def test_hit_rate() -> None:
    results = [
        EvaluationResult(
            question="A",
            expected_sources=("a.txt",),
            retrieved_sources=("a.txt",),
            hit=True,
            reciprocal_rank=1.0,
            latency_seconds=0.1,
        ),
        EvaluationResult(
            question="B",
            expected_sources=("b.txt",),
            retrieved_sources=("c.txt",),
            hit=False,
            reciprocal_rank=0.0,
            latency_seconds=0.2,
        ),
    ]

    assert (
        calculate_hit_rate(
            results
        )
        == 0.5
    )


def test_mean_reciprocal_rank() -> None:
    results = [
        EvaluationResult(
            question="A",
            expected_sources=("a.txt",),
            retrieved_sources=("a.txt",),
            hit=True,
            reciprocal_rank=1.0,
            latency_seconds=0.1,
        ),
        EvaluationResult(
            question="B",
            expected_sources=("b.txt",),
            retrieved_sources=(
                "x.txt",
                "b.txt",
            ),
            hit=True,
            reciprocal_rank=0.5,
            latency_seconds=0.2,
        ),
    ]

    assert (
        calculate_mean_reciprocal_rank(
            results
        )
        == 0.75
    )


def test_average_latency() -> None:
    results = [
        EvaluationResult(
            question="A",
            expected_sources=("a.txt",),
            retrieved_sources=("a.txt",),
            hit=True,
            reciprocal_rank=1.0,
            latency_seconds=0.2,
        ),
        EvaluationResult(
            question="B",
            expected_sources=("b.txt",),
            retrieved_sources=("b.txt",),
            hit=True,
            reciprocal_rank=1.0,
            latency_seconds=0.4,
        ),
    ]

    assert math.isclose(
        calculate_average_latency(
            results
        ),
        0.3,
        abs_tol=1e-9,
    )


def test_median_latency() -> None:
    results = [
        EvaluationResult(
            question="A",
            expected_sources=("a.txt",),
            retrieved_sources=("a.txt",),
            hit=True,
            reciprocal_rank=1.0,
            latency_seconds=0.2,
        ),
        EvaluationResult(
            question="B",
            expected_sources=("b.txt",),
            retrieved_sources=("b.txt",),
            hit=True,
            reciprocal_rank=1.0,
            latency_seconds=0.6,
        ),
        EvaluationResult(
            question="C",
            expected_sources=("c.txt",),
            retrieved_sources=("c.txt",),
            hit=True,
            reciprocal_rank=1.0,
            latency_seconds=0.4,
        ),
    ]

    assert math.isclose(
        calculate_median_latency(
            results
        ),
        0.4,
        abs_tol=1e-9,
    )


def test_min_and_max_latency() -> None:
    results = [
        EvaluationResult(
            question="A",
            expected_sources=("a.txt",),
            retrieved_sources=("a.txt",),
            hit=True,
            reciprocal_rank=1.0,
            latency_seconds=0.2,
        ),
        EvaluationResult(
            question="B",
            expected_sources=("b.txt",),
            retrieved_sources=("b.txt",),
            hit=True,
            reciprocal_rank=1.0,
            latency_seconds=0.8,
        ),
    ]

    assert (
        calculate_min_latency(
            results
        )
        == 0.2
    )

    assert (
        calculate_max_latency(
            results
        )
        == 0.8
    )


def test_warm_average_excludes_first_result() -> None:
    results = [
        EvaluationResult(
            question="Cold",
            expected_sources=("a.txt",),
            retrieved_sources=("a.txt",),
            hit=True,
            reciprocal_rank=1.0,
            latency_seconds=10.0,
        ),
        EvaluationResult(
            question="Warm 1",
            expected_sources=("b.txt",),
            retrieved_sources=("b.txt",),
            hit=True,
            reciprocal_rank=1.0,
            latency_seconds=0.4,
        ),
        EvaluationResult(
            question="Warm 2",
            expected_sources=("c.txt",),
            retrieved_sources=("c.txt",),
            hit=True,
            reciprocal_rank=1.0,
            latency_seconds=0.6,
        ),
    ]

    assert math.isclose(
        calculate_warm_average_latency(
            results
        ),
        0.5,
        abs_tol=1e-9,
    )


def test_rag_supported_case_passes() -> None:
    assistant = MagicMock()

    assistant.answer.return_value = {
        "answer": (
            "STM32 bir mikrodenetleyici ailesidir."
        ),
        "sources": [
            "stm32_notes.txt",
        ],
        "confidence": {
            "level": "high",
            "evidence_coverage": 1.0,
        },
        "query_rewrite": {
            "was_rewritten": False,
        },
    }

    case = RAGQualityCase(
        name="STM32 test",
        case_type=RAGCaseType.SUPPORTED,
        question="STM32 nedir?",
        expected_sources=(
            "stm32_notes.txt",
        ),
        expect_fallback=False,
        expect_rewrite=False,
        expected_confidence_levels=(
            "medium",
            "high",
        ),
    )

    result = evaluate_rag_quality_case(
        case=case,
        assistant=assistant,
    )

    assert result.passed is True
    assert result.source_correct is True
    assert result.fallback_correct is True
    assert result.rewrite_correct is True
    assert result.confidence_correct is True
    assert result.grounding_correct is True

    assistant.clear_memory.assert_called_once()


def test_rag_unsupported_case_requires_fallback_and_no_sources() -> None:
    assistant = MagicMock()

    assistant.answer.return_value = {
        "answer": (
            "Bu bilgi mevcut yerel belgelerde bulunamadı."
        ),
        "sources": [],
        "confidence": {
            "level": "low",
            "evidence_coverage": 0.0,
        },
        "query_rewrite": {
            "was_rewritten": False,
        },
    }

    case = RAGQualityCase(
        name="Unsupported",
        case_type=RAGCaseType.UNSUPPORTED,
        question="Fransa'nın başkenti nedir?",
        expect_fallback=True,
        expect_rewrite=False,
        expected_confidence_levels=(
            "low",
        ),
    )

    result = evaluate_rag_quality_case(
        case=case,
        assistant=assistant,
    )

    assert result.passed is True
    assert result.actual_fallback is True
    assert result.actual_sources == ()
    assert result.grounding_correct is True


def test_followup_case_runs_setup_question() -> None:
    assistant = MagicMock()

    assistant.answer.side_effect = [
        {
            "answer": "STM32 cevabı",
            "sources": [
                "stm32_notes.txt",
            ],
            "confidence": {
                "level": "high",
                "evidence_coverage": 1.0,
            },
            "query_rewrite": {
                "was_rewritten": False,
            },
        },
        {
            "answer": "PWM cevabı",
            "sources": [
                "stm32_notes.txt",
            ],
            "confidence": {
                "level": "high",
                "evidence_coverage": 1.0,
            },
            "query_rewrite": {
                "was_rewritten": True,
            },
        },
    ]

    case = RAGQualityCase(
        name="Follow-up",
        case_type=RAGCaseType.FOLLOW_UP,
        setup_questions=(
            "STM32 nedir?",
        ),
        question=(
            "Peki PWM ne işe yarar?"
        ),
        expected_sources=(
            "stm32_notes.txt",
        ),
        expect_fallback=False,
        expect_rewrite=True,
        expected_confidence_levels=(
            "high",
        ),
    )

    result = evaluate_rag_quality_case(
        case=case,
        assistant=assistant,
    )

    assert result.passed is True

    assert (
        assistant.answer.call_count
        == 2
    )

    assert (
        assistant.answer.call_args_list[0]
        .args[0]
        == "STM32 nedir?"
    )

    assert (
        assistant.answer.call_args_list[1]
        .args[0]
        == "Peki PWM ne işe yarar?"
    )


def _make_quality_result(
    *,
    passed: bool = True,
    source_correct: bool = True,
    fallback_correct: bool = True,
    rewrite_correct: bool = True,
    confidence_correct: bool = True,
    grounding_correct: bool = True,
) -> RAGQualityResult:
    return RAGQualityResult(
        name="Case",
        case_type=RAGCaseType.SUPPORTED,
        question="Question",
        setup_questions=(),
        expected_sources=("a.txt",),
        actual_sources=("a.txt",),
        expected_fallback=False,
        actual_fallback=False,
        expected_rewrite=False,
        actual_rewrite=False,
        expected_confidence_levels=("high",),
        actual_confidence_level="high",
        evidence_coverage=1.0,
        source_correct=source_correct,
        fallback_correct=fallback_correct,
        rewrite_correct=rewrite_correct,
        confidence_correct=confidence_correct,
        grounding_correct=grounding_correct,
        passed=passed,
        failures=(),
        latency_seconds=0.1,
    )


def test_quality_summary_perfect_score_passes_gate() -> None:
    results = [
        _make_quality_result(),
        _make_quality_result(),
    ]

    summary = calculate_quality_summary(
        results
    )

    assert (
        summary.overall_quality_score
        == 1.0
    )

    assert (
        summary.quality_gate_passed
        is True
    )

    assert summary.passed_count == 2
    assert summary.failed_count == 0


def test_quality_gate_requires_perfect_fallback_accuracy() -> None:
    results = [
        _make_quality_result(),
        _make_quality_result(
            passed=False,
            fallback_correct=False,
        ),
    ]

    summary = calculate_quality_summary(
        results,
        minimum_quality_score=0.70,
    )

    assert (
        summary.fallback_accuracy
        == 0.5
    )

    assert (
        summary.quality_gate_passed
        is False
    )


def test_quality_gate_rejects_low_grounding_accuracy() -> None:
    results = [
        _make_quality_result(),
        _make_quality_result(
            passed=False,
            grounding_correct=False,
        ),
    ]

    summary = calculate_quality_summary(
        results,
        minimum_quality_score=0.90,
    )

    assert (
        summary.grounding_accuracy
        == 0.5
    )

    assert (
        summary.quality_gate_passed
        is False
    )


def test_invalid_quality_threshold_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_quality_score",
    ):
        calculate_quality_summary(
            [
                _make_quality_result()
            ],
            minimum_quality_score=1.5,
        )


def test_invalid_rag_confidence_level_is_rejected() -> None:
    assistant = MagicMock()

    case = RAGQualityCase(
        name="Bad level",
        case_type=RAGCaseType.SUPPORTED,
        question="STM32 nedir?",
        expected_confidence_levels=(
            "super-high",
        ),
    )

    with pytest.raises(
        ValueError,
        match="confidence level",
    ):
        evaluate_rag_quality_case(
            case=case,
            assistant=assistant,
        )


def test_quality_evaluation_requires_cases() -> None:
    assistant = MagicMock()

    with pytest.raises(
        ValueError,
        match="At least one RAG quality case",
    ):
        evaluate_rag_quality(
            cases=[],
            assistant=assistant,
        )