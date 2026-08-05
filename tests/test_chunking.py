from src.chunker import split_into_chunks


def test_short_text_creates_one_chunk() -> None:
    text = "Bu kısa bir test metnidir."

    chunks = split_into_chunks(
        text=text,
        max_characters=100,
        overlap_characters=10,
    )

    assert chunks == [text]


def test_long_text_respects_chunk_limit() -> None:
    text = " ".join(
        f"kelime{i}"
        for i in range(150)
    )

    chunks = split_into_chunks(
        text=text,
        max_characters=100,
        overlap_characters=20,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.strip()
        assert len(chunk) <= 100


def test_chunks_have_overlap() -> None:
    text = (
        "Birinci bölüm çizgi takip sensörlerini açıklar. "
        "İkinci bölüm PID hata hesabını açıklar. "
        "Üçüncü bölüm motor hız kontrolünü açıklar."
    )

    chunks = split_into_chunks(
        text=text,
        max_characters=70,
        overlap_characters=20,
    )

    assert len(chunks) >= 2

    first_chunk_words = set(chunks[0].split())
    second_chunk_words = set(chunks[1].split())

    assert first_chunk_words.intersection(
        second_chunk_words
    )


def test_empty_text_returns_empty_list() -> None:
    chunks = split_into_chunks(
        text="   ",
        max_characters=100,
        overlap_characters=10,
    )

    assert chunks == []


def test_invalid_chunk_size_raises_error() -> None:
    try:
        split_into_chunks(
            text="Test",
            max_characters=0,
            overlap_characters=0,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for invalid max_characters."
    )


def test_invalid_overlap_raises_error() -> None:
    try:
        split_into_chunks(
            text="Test metni",
            max_characters=100,
            overlap_characters=100,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for invalid overlap."
    )


def run_tests() -> None:
    tests = [
        test_short_text_creates_one_chunk,
        test_long_text_respects_chunk_limit,
        test_chunks_have_overlap,
        test_empty_text_returns_empty_list,
        test_invalid_chunk_size_raises_error,
        test_invalid_overlap_raises_error,
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