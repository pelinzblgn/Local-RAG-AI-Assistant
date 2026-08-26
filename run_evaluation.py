from evaluation.evaluation_cases import (
    RAG_QUALITY_CASES,
    RETRIEVAL_EVALUATION_CASES,
)
from src.assistant import RAGAssistant
from src.evaluation import (
    calculate_average_latency,
    calculate_hit_rate,
    calculate_max_latency,
    calculate_mean_reciprocal_rank,
    calculate_median_latency,
    calculate_min_latency,
    calculate_quality_summary,
    calculate_warm_average_latency,
    evaluate_rag_quality,
    evaluate_retrieval,
)
from src.logging_config import configure_logging


def _print_retrieval_evaluation() -> None:
    """Run and print retrieval benchmark."""

    print()
    print("=" * 70)
    print("RETRIEVAL BENCHMARK")
    print("=" * 70)

    results = evaluate_retrieval(
        cases=RETRIEVAL_EVALUATION_CASES,
        top_k=3,
    )

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n[{index}] {result.question}"
        )

        print(
            "Expected : "
            + ", ".join(
                result.expected_sources
            )
        )

        retrieved = (
            ", ".join(
                result.retrieved_sources
            )
            if result.retrieved_sources
            else "NONE"
        )

        print(
            f"Retrieved: {retrieved}"
        )

        print(
            "Hit      : "
            f"{'YES' if result.hit else 'NO'}"
        )

        print(
            "RR       : "
            f"{result.reciprocal_rank:.4f}"
        )

        print(
            "Latency  : "
            f"{result.latency_seconds:.4f} s"
        )

    hit_rate = calculate_hit_rate(
        results
    )

    mrr = calculate_mean_reciprocal_rank(
        results
    )

    average_latency = (
        calculate_average_latency(
            results
        )
    )

    median_latency = (
        calculate_median_latency(
            results
        )
    )

    minimum_latency = (
        calculate_min_latency(
            results
        )
    )

    maximum_latency = (
        calculate_max_latency(
            results
        )
    )

    warm_average_latency = (
        calculate_warm_average_latency(
            results
        )
    )

    print()
    print("-" * 70)
    print("RETRIEVAL SUMMARY")
    print("-" * 70)

    print(
        f"Cases                  : {len(results)}"
    )

    print(
        f"Hit Rate               : {hit_rate:.2%}"
    )

    print(
        f"MRR                    : {mrr:.4f}"
    )

    print(
        "Cold Start             : "
        f"{results[0].latency_seconds:.4f} s"
    )

    print(
        "Avg Retrieval Time     : "
        f"{average_latency:.4f} s"
    )

    print(
        "Warm Avg Retrieval     : "
        f"{warm_average_latency:.4f} s"
    )

    print(
        "Median Retrieval       : "
        f"{median_latency:.4f} s"
    )

    print(
        "Fastest Retrieval      : "
        f"{minimum_latency:.4f} s"
    )

    print(
        "Slowest Retrieval      : "
        f"{maximum_latency:.4f} s"
    )


def _print_quality_evaluation() -> bool:
    """
    Run end-to-end RAG quality evaluation.

    Returns:
        Whether the quality gate passed.
    """

    print()
    print("=" * 70)
    print("END-TO-END RAG QUALITY EVALUATION")
    print("=" * 70)

    with RAGAssistant() as assistant:
        print(
            "\nYerel chat modeli hazırlanıyor..."
        )

        assistant.warm_up()

        print(
            "Chat modeli hazır."
        )

        results = evaluate_rag_quality(
            cases=RAG_QUALITY_CASES,
            assistant=assistant,
        )

    for index, result in enumerate(
        results,
        start=1,
    ):
        status = (
            "PASS"
            if result.passed
            else "FAIL"
        )

        print()
        print(
            f"[{index}] [{status}] "
            f"{result.name}"
        )

        print(
            f"Type       : "
            f"{result.case_type.value.upper()}"
        )

        print(
            f"Question   : "
            f"{result.question}"
        )

        if result.setup_questions:
            print(
                "Setup      : "
                + " | ".join(
                    result.setup_questions
                )
            )

        expected_sources = (
            ", ".join(
                result.expected_sources
            )
            if result.expected_sources
            else "NONE"
        )

        actual_sources = (
            ", ".join(
                result.actual_sources
            )
            if result.actual_sources
            else "NONE"
        )

        print(
            f"Expected Src: {expected_sources}"
        )

        print(
            f"Actual Src  : {actual_sources}"
        )

        print(
            "Fallback     : "
            f"{'YES' if result.actual_fallback else 'NO'}"
        )

        print(
            "Rewrite      : "
            f"{'YES' if result.actual_rewrite else 'NO'}"
        )

        print(
            "Confidence   : "
            f"{result.actual_confidence_level.upper()}"
        )

        print(
            "Evidence     : "
            f"{result.evidence_coverage:.2%}"
        )

        print(
            "Latency      : "
            f"{result.latency_seconds:.4f} s"
        )

        if result.failures:
            for failure in result.failures:
                print(
                    f"Failure      : {failure}"
                )

    summary = (
        calculate_quality_summary(
            results
        )
    )

    print()
    print("-" * 70)
    print("RAG QUALITY SUMMARY")
    print("-" * 70)

    print(
        f"Cases                 : "
        f"{summary.case_count}"
    )

    print(
        f"Passed                : "
        f"{summary.passed_count}"
    )

    print(
        f"Failed                : "
        f"{summary.failed_count}"
    )

    print(
        f"Source Accuracy       : "
        f"{summary.source_accuracy:.2%}"
    )

    print(
        f"Fallback Accuracy     : "
        f"{summary.fallback_accuracy:.2%}"
    )

    print(
        f"Rewrite Accuracy      : "
        f"{summary.rewrite_accuracy:.2%}"
    )

    print(
        f"Confidence Accuracy   : "
        f"{summary.confidence_accuracy:.2%}"
    )

    print(
        f"Grounding Accuracy    : "
        f"{summary.grounding_accuracy:.2%}"
    )

    print(
        f"Overall Quality Score : "
        f"{summary.overall_quality_score:.2%}"
    )

    print(
        "Quality Gate          : "
        + (
            "PASSED"
            if summary.quality_gate_passed
            else "FAILED"
        )
    )

    return (
        summary.quality_gate_passed
    )


def main() -> None:
    """
    Run complete Local RAG evaluation suite.
    """

    configure_logging()

    print(
        "Local RAG AI Assistant"
    )

    print(
        "Evaluation & Quality Gate v2"
    )

    _print_retrieval_evaluation()

    quality_gate_passed = (
        _print_quality_evaluation()
    )

    print()
    print("=" * 70)

    if quality_gate_passed:
        print(
            "FINAL STATUS: QUALITY GATE PASSED"
        )
    else:
        print(
            "FINAL STATUS: QUALITY GATE FAILED"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()