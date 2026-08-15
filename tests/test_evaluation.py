from unittest.mock import patch

from src.evaluation import (
    EvaluationCase,
    EvaluationResult,
    calculate_average_latency,
    calculate_hit_rate,
    calculate_mean_reciprocal_rank,
    calculate_reciprocal_rank,
    evaluate_retrieval_case,
    calculate_max_latency,
    calculate_median_latency,
    calculate_min_latency,
    calculate_warm_average_latency,
)

import math 


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
    assert result.reciprocal_rank == 1.0


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

    assert calculate_hit_rate(
        results
    ) == 0.5


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
            retrieved_sources=("x.txt", "b.txt"),
            hit=True,
            reciprocal_rank=0.5,
            latency_seconds=0.2,
        ),
    ]

    assert calculate_mean_reciprocal_rank(
        results
    ) == 0.75


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

    assert calculate_average_latency(
        results
    ) == 0.30000000000000004
    
    
    assert math.isclose(
    calculate_average_latency(results),
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
        calculate_median_latency(results),
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

    assert calculate_min_latency(results) == 0.2
    assert calculate_max_latency(results) == 0.8


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
        calculate_warm_average_latency(results),
        0.5,
        abs_tol=1e-9,
    )
