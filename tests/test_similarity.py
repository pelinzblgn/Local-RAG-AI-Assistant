import math

from src.similarity import cosine_similarity


def test_identical_vectors_have_similarity_one() -> None:
    result = cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    )

    assert math.isclose(result, 1.0, abs_tol=1e-9)


def test_opposite_vectors_have_similarity_minus_one() -> None:
    result = cosine_similarity(
        [1.0, 2.0, 3.0],
        [-1.0, -2.0, -3.0],
    )

    assert math.isclose(result, -1.0, abs_tol=1e-9)


def test_orthogonal_vectors_have_similarity_zero() -> None:
    result = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert math.isclose(result, 0.0, abs_tol=1e-9)


def test_similar_vectors_have_positive_score() -> None:
    result = cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 2.5],
    )

    assert 0.0 < result <= 1.0


def test_zero_vector_returns_zero() -> None:
    result = cosine_similarity(
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
    )

    assert result == 0.0


def test_different_vector_lengths_raise_error() -> None:
    try:
        cosine_similarity(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
        )
    except ValueError as error:
        assert "same length" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for vectors with different lengths."
    )


def test_empty_vectors_raise_error() -> None:
    try:
        cosine_similarity([], [])
    except ValueError as error:
        assert "empty" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for empty vectors."
    )


def run_tests() -> None:
    tests = [
        test_identical_vectors_have_similarity_one,
        test_opposite_vectors_have_similarity_minus_one,
        test_orthogonal_vectors_have_similarity_zero,
        test_similar_vectors_have_positive_score,
        test_zero_vector_returns_zero,
        test_different_vector_lengths_raise_error,
        test_empty_vectors_raise_error,
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
    print(f"Sonuç: {passed}/{len(tests)} test başarılı.")


if __name__ == "__main__":
    run_tests()