from evaluation.evaluation_cases import (
    EVALUATION_CASES,
)
from src.evaluation import (
    calculate_average_latency,
    calculate_hit_rate,
    calculate_max_latency,
    calculate_mean_reciprocal_rank,
    calculate_median_latency,
    calculate_min_latency,
    calculate_warm_average_latency,
    evaluate_retrieval,
)
from src.logging_config import configure_logging


def main() -> None:
    """Run retrieval evaluation."""

    configure_logging()

    print("Local RAG Retrieval Evaluation")
    print("=" * 60)

    results = evaluate_retrieval(
        cases=EVALUATION_CASES,
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
            "Expected:",
            ", ".join(
                result.expected_sources
            ),
        )

        print(
            "Retrieved:",
            ", ".join(
                result.retrieved_sources
            ),
        )

        print(
            f"Hit: {'YES' if result.hit else 'NO'}"
        )

        print(
            "Reciprocal Rank: "
            f"{result.reciprocal_rank:.4f}"
        )

        print(
            "Latency: "
            f"{result.latency_seconds:.4f} s"
        )

    hit_rate = calculate_hit_rate(
        results
    )

    mrr = calculate_mean_reciprocal_rank(
        results
    )

    average_latency = calculate_average_latency(
        results
    )

    median_latency = calculate_median_latency(
        results
    )

    minimum_latency = calculate_min_latency(
        results
    )

    maximum_latency = calculate_max_latency(
        results
    )

    warm_average_latency = (
        calculate_warm_average_latency(
            results
        )
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

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


if __name__ == "__main__":
    main()