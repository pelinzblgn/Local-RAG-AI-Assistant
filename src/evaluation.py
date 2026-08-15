from dataclasses import dataclass
from statistics import median
from time import perf_counter

from src.retrieval import get_top_documents


@dataclass(frozen=True)
class EvaluationCase:
    """
    One retrieval evaluation case.

    expected_sources contains the source files that should
    appear in the retrieved results.
    """

    question: str
    expected_sources: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """Result of one retrieval evaluation case."""

    question: str
    expected_sources: tuple[str, ...]
    retrieved_sources: tuple[str, ...]
    hit: bool
    reciprocal_rank: float
    latency_seconds: float


def _validate_case(
    case: EvaluationCase,
) -> None:
    """Validate one evaluation case."""

    if not case.question.strip():
        raise ValueError(
            "Evaluation question cannot be empty."
        )

    if not case.expected_sources:
        raise ValueError(
            "At least one expected source is required."
        )

    for source in case.expected_sources:
        if not isinstance(source, str):
            raise TypeError(
                "Expected source names must be strings."
            )

        if not source.strip():
            raise ValueError(
                "Expected source names cannot be empty."
            )


def calculate_reciprocal_rank(
    retrieved_sources: list[str],
    expected_sources: tuple[str, ...],
) -> float:
    """
    Calculate reciprocal rank of the first relevant source.

    Returns:
        1.0 if the first result is relevant,
        0.5 if the second result is relevant,
        0.333... if the third is relevant,
        or 0.0 if no expected source was retrieved.
    """

    expected_set = set(expected_sources)

    for rank, source in enumerate(
        retrieved_sources,
        start=1,
    ):
        if source in expected_set:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval_case(
    case: EvaluationCase,
    top_k: int = 3,
) -> EvaluationResult:
    """Evaluate retrieval for one question."""

    _validate_case(case)

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    start_time = perf_counter()

    documents = get_top_documents(
        query=case.question,
        top_k=top_k,
    )

    latency_seconds = (
        perf_counter() - start_time
    )

    retrieved_sources = [
        document["source"]
        for document in documents
    ]

    hit = any(
        source in case.expected_sources
        for source in retrieved_sources
    )

    reciprocal_rank = calculate_reciprocal_rank(
        retrieved_sources=retrieved_sources,
        expected_sources=case.expected_sources,
    )

    return EvaluationResult(
        question=case.question,
        expected_sources=case.expected_sources,
        retrieved_sources=tuple(
            retrieved_sources
        ),
        hit=hit,
        reciprocal_rank=reciprocal_rank,
        latency_seconds=latency_seconds,
    )


def evaluate_retrieval(
    cases: list[EvaluationCase],
    top_k: int = 3,
) -> list[EvaluationResult]:
    """Evaluate multiple retrieval cases."""

    if not cases:
        raise ValueError(
            "At least one evaluation case is required."
        )

    return [
        evaluate_retrieval_case(
            case=case,
            top_k=top_k,
        )
        for case in cases
    ]


def calculate_hit_rate(
    results: list[EvaluationResult],
) -> float:
    """Return the percentage of cases with at least one hit."""

    if not results:
        raise ValueError(
            "At least one evaluation result is required."
        )

    hit_count = sum(
        result.hit
        for result in results
    )

    return hit_count / len(results)


def calculate_mean_reciprocal_rank(
    results: list[EvaluationResult],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""

    if not results:
        raise ValueError(
            "At least one evaluation result is required."
        )

    return sum(
        result.reciprocal_rank
        for result in results
    ) / len(results)


def calculate_average_latency(
    results: list[EvaluationResult],
) -> float:
    """Calculate average retrieval latency."""

    if not results:
        raise ValueError(
            "At least one evaluation result is required."
        )

    return sum(
        result.latency_seconds
        for result in results
    ) / len(results)


def calculate_median_latency(
    results: list[EvaluationResult],
) -> float:
    """Calculate median retrieval latency."""

    if not results:
        raise ValueError(
            "At least one evaluation result is required."
        )

    return median(
        result.latency_seconds
        for result in results
    )


def calculate_min_latency(
    results: list[EvaluationResult],
) -> float:
    """Return fastest retrieval latency."""

    if not results:
        raise ValueError(
            "At least one evaluation result is required."
        )

    return min(
        result.latency_seconds
        for result in results
    )


def calculate_max_latency(
    results: list[EvaluationResult],
) -> float:
    """Return slowest retrieval latency."""

    if not results:
        raise ValueError(
            "At least one evaluation result is required."
        )

    return max(
        result.latency_seconds
        for result in results
    )


def calculate_warm_average_latency(
    results: list[EvaluationResult],
) -> float:
    """
    Calculate average latency excluding the first request.

    The first request may include model cold-start/loading cost.
    """

    if len(results) < 2:
        raise ValueError(
            "At least two results are required "
            "for warm latency calculation."
        )

    warm_results = results[1:]

    return calculate_average_latency(
        warm_results
    )