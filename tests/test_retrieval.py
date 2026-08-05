from unittest.mock import patch

from src.retrieval import get_top_documents


SAMPLE_DOCUMENTS = [
    {
        "id": 1,
        "content": "PWM duty cycle motor hızını kontrol eder.",
        "source": "stm32_notes.txt",
        "embedding": [1.0, 0.0],
        "created_at": "2026-08-01 10:00:00",
    },
    {
        "id": 2,
        "content": "SQLite hafif bir yerel veritabanıdır.",
        "source": "sqlite_notes.txt",
        "embedding": [0.0, 1.0],
        "created_at": "2026-08-01 10:01:00",
    },
    {
        "id": 3,
        "content": "PID motor hızlarında düzeltme uygular.",
        "source": "pid_notes.txt",
        "embedding": [0.8, 0.2],
        "created_at": "2026-08-01 10:02:00",
    },
]


def test_documents_are_sorted_by_similarity() -> None:
    with (
        patch(
            "src.retrieval.get_all_documents",
            return_value=SAMPLE_DOCUMENTS,
        ),
        patch(
            "src.retrieval.generate_embedding",
            return_value=[1.0, 0.0],
        ),
    ):
        results = get_top_documents(
            query="Motor hızı nasıl kontrol edilir?",
            top_k=3,
            minimum_score=None,
        )

    assert len(results) == 3
    assert results[0]["source"] == "stm32_notes.txt"
    assert results[1]["source"] == "pid_notes.txt"
    assert results[2]["source"] == "sqlite_notes.txt"

    assert results[0]["score"] >= results[1]["score"]
    assert results[1]["score"] >= results[2]["score"]

def test_top_k_limits_result_count() -> None:
    with (
        patch(
            "src.retrieval.get_all_documents",
            return_value=SAMPLE_DOCUMENTS,
        ),
        patch(
            "src.retrieval.generate_embedding",
            return_value=[1.0, 0.0],
        ),
    ):
        results = get_top_documents(
            query="PWM nedir?",
            top_k=2,
        )

    assert len(results) == 2


def test_minimum_score_filters_low_scores() -> None:
    with (
        patch(
            "src.retrieval.get_all_documents",
            return_value=SAMPLE_DOCUMENTS,
        ),
        patch(
            "src.retrieval.generate_embedding",
            return_value=[1.0, 0.0],
        ),
    ):
        results = get_top_documents(
            query="PWM nedir?",
            top_k=3,
            minimum_score=0.50,
        )

    assert len(results) == 2

    assert all(
        result["score"] >= 0.50
        for result in results
    )

    returned_sources = {
        result["source"]
        for result in results
    }

    assert "stm32_notes.txt" in returned_sources
    assert "pid_notes.txt" in returned_sources
    assert "sqlite_notes.txt" not in returned_sources


def test_high_minimum_score_can_return_empty_list() -> None:
    with (
        patch(
            "src.retrieval.get_all_documents",
            return_value=SAMPLE_DOCUMENTS,
        ),
        patch(
            "src.retrieval.generate_embedding",
            return_value=[0.7, 0.7],
        ),
    ):
        results = get_top_documents(
            query="İlgisiz bir soru",
            top_k=3,
            minimum_score=0.99,
        )

    assert results == []


def test_result_contains_required_fields() -> None:
    with (
        patch(
            "src.retrieval.get_all_documents",
            return_value=SAMPLE_DOCUMENTS,
        ),
        patch(
            "src.retrieval.generate_embedding",
            return_value=[1.0, 0.0],
        ),
    ):
        results = get_top_documents(
            query="PWM nedir?",
            top_k=1,
        )

    result = results[0]

    assert result["id"] == 1
    assert result["source"] == "stm32_notes.txt"
    assert result["content"]
    assert isinstance(result["score"], float)


def test_query_whitespace_is_removed() -> None:
    with (
        patch(
            "src.retrieval.get_all_documents",
            return_value=SAMPLE_DOCUMENTS,
        ),
        patch(
            "src.retrieval.generate_embedding",
            return_value=[1.0, 0.0],
        ) as mocked_embedding,
    ):
        get_top_documents(
            query="   PWM nedir?   ",
            top_k=1,
        )

    mocked_embedding.assert_called_once_with(
        "PWM nedir?"
    )


def test_non_string_query_raises_error() -> None:
    try:
        get_top_documents(
            query=123,  # type: ignore[arg-type]
            top_k=3,
        )
    except TypeError as error:
        assert "Query must be a string" in str(error)
        return

    raise AssertionError(
        "Expected TypeError for non-string query."
    )


def test_empty_query_raises_error() -> None:
    try:
        get_top_documents(
            query="   ",
            top_k=3,
        )
    except ValueError as error:
        assert "Query cannot be empty" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for empty query."
    )


def test_invalid_top_k_raises_error() -> None:
    try:
        get_top_documents(
            query="Test",
            top_k=0,
        )
    except ValueError as error:
        assert "top_k" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for invalid top_k."
    )


def test_invalid_minimum_score_raises_error() -> None:
    try:
        get_top_documents(
            query="Test",
            top_k=3,
            minimum_score=1.50,
        )
    except ValueError as error:
        assert "minimum_score" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for invalid minimum score."
    )


def test_empty_database_raises_error() -> None:
    with patch(
        "src.retrieval.get_all_documents",
        return_value=[],
    ):
        try:
            get_top_documents(
                query="Test",
                top_k=3,
            )
        except RuntimeError as error:
            assert (
                "does not contain any documents"
                in str(error)
            )
            return

    raise AssertionError(
        "Expected RuntimeError for empty database."
    )


def run_tests() -> None:
    tests = [
        test_documents_are_sorted_by_similarity,
        test_top_k_limits_result_count,
        test_minimum_score_filters_low_scores,
        test_high_minimum_score_can_return_empty_list,
        test_result_contains_required_fields,
        test_query_whitespace_is_removed,
        test_non_string_query_raises_error,
        test_empty_query_raises_error,
        test_invalid_top_k_raises_error,
        test_invalid_minimum_score_raises_error,
        test_empty_database_raises_error,
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