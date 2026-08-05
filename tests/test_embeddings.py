import math

from src.embeddings import (
    _clean_texts,
    _validate_embedding,
)


def test_clean_texts_strips_whitespace() -> None:
    result = _clean_texts(
        [
            "  STM32 PWM  ",
            "  PID kontrol  ",
        ]
    )

    assert result == [
        "STM32 PWM",
        "PID kontrol",
    ]


def test_clean_texts_rejects_empty_list() -> None:
    try:
        _clean_texts([])
    except ValueError as error:
        assert "At least one text" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for empty text list."
    )


def test_clean_texts_rejects_blank_text() -> None:
    try:
        _clean_texts(
            [
                "STM32",
                "   ",
            ]
        )
    except ValueError as error:
        assert "index 1" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for blank text."
    )


def test_clean_texts_rejects_non_string_value() -> None:
    try:
        _clean_texts(
            [
                "STM32",
                123,
            ]
        )
    except TypeError as error:
        assert "index 1" in str(error)
        return

    raise AssertionError(
        "Expected TypeError for non-string input."
    )


def test_validate_embedding_converts_integers_to_floats() -> None:
    result = _validate_embedding(
        [
            1,
            2.5,
            3,
        ]
    )

    assert result == [
        1.0,
        2.5,
        3.0,
    ]


def test_validate_embedding_rejects_empty_vector() -> None:
    try:
        _validate_embedding([])
    except ValueError as error:
        assert "empty vector" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for empty embedding."
    )


def test_validate_embedding_rejects_boolean_values() -> None:
    try:
        _validate_embedding(
            [
                0.1,
                True,
                0.3,
            ]
        )
    except ValueError as error:
        assert "numerical" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for boolean embedding value."
    )


def test_validate_embedding_rejects_non_numeric_values() -> None:
    try:
        _validate_embedding(
            [
                0.1,
                "invalid",
                0.3,
            ]
        )
    except ValueError as error:
        assert "numerical" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for non-numeric embedding value."
    )


def test_validate_embedding_rejects_nan() -> None:
    try:
        _validate_embedding(
            [
                0.1,
                math.nan,
                0.3,
            ]
        )
    except ValueError as error:
        assert "finite" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for NaN embedding value."
    )


def test_validate_embedding_rejects_infinity() -> None:
    try:
        _validate_embedding(
            [
                0.1,
                math.inf,
                0.3,
            ]
        )
    except ValueError as error:
        assert "finite" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for infinite embedding value."
    )


def run_tests() -> None:
    tests = [
        test_clean_texts_strips_whitespace,
        test_clean_texts_rejects_empty_list,
        test_clean_texts_rejects_blank_text,
        test_clean_texts_rejects_non_string_value,
        test_validate_embedding_converts_integers_to_floats,
        test_validate_embedding_rejects_empty_vector,
        test_validate_embedding_rejects_boolean_values,
        test_validate_embedding_rejects_non_numeric_values,
        test_validate_embedding_rejects_nan,
        test_validate_embedding_rejects_infinity,
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