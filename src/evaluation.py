from dataclasses import dataclass
from enum import Enum
from statistics import median
from time import perf_counter
from typing import Protocol

from src.prompts import DEFAULT_FALLBACK_MESSAGE
from src.retrieval import get_top_documents


# ==========================================================
# Retrieval Evaluation
# ==========================================================


@dataclass(frozen=True)
class EvaluationCase:
    """
    One retrieval evaluation case.

    expected_sources contains source files that should appear
    somewhere in Top-K retrieval results.
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


# ==========================================================
# RAG Quality Evaluation
# ==========================================================


class RAGCaseType(str, Enum):
    """Supported RAG evaluation case categories."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FOLLOW_UP = "follow_up"


@dataclass(frozen=True)
class RAGQualityCase:
    """
    End-to-end RAG quality evaluation case.

    setup_questions:
        Optional questions executed before the target question.
        They build conversational state for follow-up evaluation.

    expected_sources:
        Trusted application sources expected in the final response.

    expect_fallback:
        Whether the canonical local-document fallback is expected.

    expect_rewrite:
        Whether conversational query rewriting is expected.

    expected_confidence_levels:
        Accepted retrieval-confidence levels for this case.
    """

    name: str
    case_type: RAGCaseType
    question: str

    expected_sources: tuple[str, ...] = ()

    expect_fallback: bool = False
    expect_rewrite: bool = False

    expected_confidence_levels: tuple[str, ...] = (
        "medium",
        "high",
    )

    setup_questions: tuple[str, ...] = ()


@dataclass(frozen=True)
class RAGQualityResult:
    """Result of one end-to-end RAG quality case."""

    name: str
    case_type: RAGCaseType

    question: str
    setup_questions: tuple[str, ...]

    expected_sources: tuple[str, ...]
    actual_sources: tuple[str, ...]

    expected_fallback: bool
    actual_fallback: bool

    expected_rewrite: bool
    actual_rewrite: bool

    expected_confidence_levels: tuple[str, ...]
    actual_confidence_level: str

    evidence_coverage: float

    source_correct: bool
    fallback_correct: bool
    rewrite_correct: bool
    confidence_correct: bool
    grounding_correct: bool

    passed: bool
    failures: tuple[str, ...]

    latency_seconds: float


@dataclass(frozen=True)
class QualitySummary:
    """Aggregate end-to-end RAG quality metrics."""

    case_count: int
    passed_count: int
    failed_count: int

    source_accuracy: float
    fallback_accuracy: float
    rewrite_accuracy: float
    confidence_accuracy: float
    grounding_accuracy: float

    overall_quality_score: float

    quality_gate_passed: bool


class AssistantProtocol(Protocol):
    """Minimal assistant interface required by evaluation."""

    def answer(
        self,
        question: str,
    ) -> dict:
        ...

    def clear_memory(
        self,
    ) -> None:
        ...


# ==========================================================
# Validation
# ==========================================================


def _validate_case(
    case: EvaluationCase,
) -> None:
    """Validate one retrieval evaluation case."""

    if not isinstance(
        case.question,
        str,
    ):
        raise TypeError(
            "Evaluation question must be a string."
        )

    if not case.question.strip():
        raise ValueError(
            "Evaluation question cannot be empty."
        )

    if not case.expected_sources:
        raise ValueError(
            "At least one expected source is required."
        )

    for source in case.expected_sources:
        if not isinstance(
            source,
            str,
        ):
            raise TypeError(
                "Expected source names must be strings."
            )

        if not source.strip():
            raise ValueError(
                "Expected source names cannot be empty."
            )


def _validate_rag_quality_case(
    case: RAGQualityCase,
) -> None:
    """Validate one end-to-end RAG quality case."""

    if not isinstance(
        case.name,
        str,
    ):
        raise TypeError(
            "Quality case name must be a string."
        )

    if not case.name.strip():
        raise ValueError(
            "Quality case name cannot be empty."
        )

    if not isinstance(
        case.question,
        str,
    ):
        raise TypeError(
            "Quality case question must be a string."
        )

    if not case.question.strip():
        raise ValueError(
            "Quality case question cannot be empty."
        )

    for source in case.expected_sources:
        if not isinstance(
            source,
            str,
        ):
            raise TypeError(
                "Expected source names must be strings."
            )

        if not source.strip():
            raise ValueError(
                "Expected source names cannot be empty."
            )

    for question in case.setup_questions:
        if not isinstance(
            question,
            str,
        ):
            raise TypeError(
                "Setup questions must be strings."
            )

        if not question.strip():
            raise ValueError(
                "Setup questions cannot be empty."
            )

    allowed_levels = {
        "low",
        "medium",
        "high",
    }

    if not case.expected_confidence_levels:
        raise ValueError(
            "At least one expected confidence level is required."
        )

    for level in case.expected_confidence_levels:
        if level not in allowed_levels:
            raise ValueError(
                "Expected confidence level must be "
                "'low', 'medium', or 'high'."
            )


# ==========================================================
# Retrieval Metrics
# ==========================================================


def calculate_reciprocal_rank(
    retrieved_sources: list[str],
    expected_sources: tuple[str, ...],
) -> float:
    """
    Calculate reciprocal rank of the first relevant source.
    """

    expected_set = set(
        expected_sources
    )

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

    _validate_case(
        case
    )

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
        perf_counter()
        - start_time
    )

    retrieved_sources = [
        document["source"]
        for document in documents
    ]

    hit = any(
        source in case.expected_sources
        for source in retrieved_sources
    )

    reciprocal_rank = (
        calculate_reciprocal_rank(
            retrieved_sources=retrieved_sources,
            expected_sources=case.expected_sources,
        )
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
    """Return percentage of retrieval cases with a hit."""

    if not results:
        raise ValueError(
            "At least one evaluation result is required."
        )

    hit_count = sum(
        result.hit
        for result in results
    )

    return (
        hit_count
        / len(results)
    )


def calculate_mean_reciprocal_rank(
    results: list[EvaluationResult],
) -> float:
    """Calculate Mean Reciprocal Rank."""

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
    Calculate average retrieval latency excluding
    the first cold-start request.
    """

    if len(results) < 2:
        raise ValueError(
            "At least two results are required "
            "for warm latency calculation."
        )

    return calculate_average_latency(
        results[1:]
    )


# ==========================================================
# RAG Quality Evaluation
# ==========================================================


def _is_fallback_response(
    answer: object,
) -> bool:
    """
    Detect canonical local-document fallback.
    """

    if not isinstance(
        answer,
        str,
    ):
        return False

    return (
        " ".join(
            answer.strip().split()
        )
        == " ".join(
            DEFAULT_FALLBACK_MESSAGE
            .strip()
            .split()
        )
    )


def _calculate_source_correct(
    expected_sources: tuple[str, ...],
    actual_sources: tuple[str, ...],
    expect_fallback: bool,
) -> bool:
    """
    Evaluate trusted source behavior.

    Unsupported/fallback cases must expose no sources.

    Supported cases pass when at least one expected source is
    present in trusted application sources.
    """

    if expect_fallback:
        return (
            len(actual_sources)
            == 0
        )

    if not expected_sources:
        return True

    expected_set = set(
        expected_sources
    )

    return any(
        source in expected_set
        for source in actual_sources
    )


def evaluate_rag_quality_case(
    case: RAGQualityCase,
    assistant: AssistantProtocol,
) -> RAGQualityResult:
    """
    Evaluate one end-to-end RAG behavior case.

    Memory is cleared before each case to prevent evaluation
    contamination.

    setup_questions are executed first when conversational state
    is required.
    """

    _validate_rag_quality_case(
        case
    )

    assistant.clear_memory()

    for setup_question in case.setup_questions:
        assistant.answer(
            setup_question
        )

    start_time = perf_counter()

    response = assistant.answer(
        case.question
    )

    latency_seconds = (
        perf_counter()
        - start_time
    )

    answer = response.get(
        "answer",
        "",
    )

    raw_sources = response.get(
        "sources",
        [],
    )

    actual_sources = tuple(
        str(source)
        for source in raw_sources
    )

    confidence = response.get(
        "confidence",
        {},
    )

    actual_confidence_level = str(
        confidence.get(
            "level",
            "unknown",
        )
    ).lower()

    evidence_coverage = float(
        confidence.get(
            "evidence_coverage",
            0.0,
        )
    )

    query_rewrite = response.get(
        "query_rewrite",
        {},
    )

    actual_rewrite = bool(
        query_rewrite.get(
            "was_rewritten",
            False,
        )
    )

    actual_fallback = (
        _is_fallback_response(
            answer
        )
    )

    source_correct = (
        _calculate_source_correct(
            expected_sources=case.expected_sources,
            actual_sources=actual_sources,
            expect_fallback=case.expect_fallback,
        )
    )

    fallback_correct = (
        actual_fallback
        == case.expect_fallback
    )

    rewrite_correct = (
        actual_rewrite
        == case.expect_rewrite
    )

    confidence_correct = (
        actual_confidence_level
        in case.expected_confidence_levels
    )

    if case.expect_fallback:
        grounding_correct = (
            actual_fallback
            and not actual_sources
        )
    else:
        grounding_correct = (
            not actual_fallback
            and source_correct
        )

    failures: list[str] = []

    if not source_correct:
        failures.append(
            "Trusted source behavior did not match expectation."
        )

    if not fallback_correct:
        failures.append(
            "Fallback behavior did not match expectation."
        )

    if not rewrite_correct:
        failures.append(
            "Query rewrite behavior did not match expectation."
        )

    if not confidence_correct:
        failures.append(
            "Confidence level did not match expectation."
        )

    if not grounding_correct:
        failures.append(
            "Grounding behavior did not match expectation."
        )

    passed = not failures

    return RAGQualityResult(
        name=case.name,
        case_type=case.case_type,
        question=case.question,
        setup_questions=case.setup_questions,
        expected_sources=case.expected_sources,
        actual_sources=actual_sources,
        expected_fallback=case.expect_fallback,
        actual_fallback=actual_fallback,
        expected_rewrite=case.expect_rewrite,
        actual_rewrite=actual_rewrite,
        expected_confidence_levels=(
            case.expected_confidence_levels
        ),
        actual_confidence_level=(
            actual_confidence_level
        ),
        evidence_coverage=evidence_coverage,
        source_correct=source_correct,
        fallback_correct=fallback_correct,
        rewrite_correct=rewrite_correct,
        confidence_correct=confidence_correct,
        grounding_correct=grounding_correct,
        passed=passed,
        failures=tuple(
            failures
        ),
        latency_seconds=latency_seconds,
    )


def evaluate_rag_quality(
    cases: list[RAGQualityCase],
    assistant: AssistantProtocol,
) -> list[RAGQualityResult]:
    """Evaluate multiple end-to-end RAG quality cases."""

    if not cases:
        raise ValueError(
            "At least one RAG quality case is required."
        )

    return [
        evaluate_rag_quality_case(
            case=case,
            assistant=assistant,
        )
        for case in cases
    ]


def calculate_quality_summary(
    results: list[RAGQualityResult],
    minimum_quality_score: float = 0.90,
) -> QualitySummary:
    """
    Calculate aggregate RAG quality metrics.

    Quality Gate requires:
        - overall score >= minimum_quality_score
        - fallback accuracy == 100%
        - grounding accuracy >= minimum_quality_score
    """

    if not results:
        raise ValueError(
            "At least one RAG quality result is required."
        )

    if not 0.0 <= minimum_quality_score <= 1.0:
        raise ValueError(
            "minimum_quality_score must be between 0 and 1."
        )

    case_count = len(
        results
    )

    passed_count = sum(
        result.passed
        for result in results
    )

    failed_count = (
        case_count
        - passed_count
    )

    source_accuracy = sum(
        result.source_correct
        for result in results
    ) / case_count

    fallback_accuracy = sum(
        result.fallback_correct
        for result in results
    ) / case_count

    rewrite_accuracy = sum(
        result.rewrite_correct
        for result in results
    ) / case_count

    confidence_accuracy = sum(
        result.confidence_correct
        for result in results
    ) / case_count

    grounding_accuracy = sum(
        result.grounding_correct
        for result in results
    ) / case_count

    overall_quality_score = (
        source_accuracy
        + fallback_accuracy
        + rewrite_accuracy
        + confidence_accuracy
        + grounding_accuracy
    ) / 5.0

    quality_gate_passed = (
        overall_quality_score
        >= minimum_quality_score
        and fallback_accuracy == 1.0
        and grounding_accuracy
        >= minimum_quality_score
    )

    return QualitySummary(
        case_count=case_count,
        passed_count=passed_count,
        failed_count=failed_count,
        source_accuracy=source_accuracy,
        fallback_accuracy=fallback_accuracy,
        rewrite_accuracy=rewrite_accuracy,
        confidence_accuracy=confidence_accuracy,
        grounding_accuracy=grounding_accuracy,
        overall_quality_score=overall_quality_score,
        quality_gate_passed=quality_gate_passed,
    )