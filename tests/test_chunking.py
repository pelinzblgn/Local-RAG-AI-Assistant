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
def test_paragraph_boundaries_are_preserved() -> None:
    text = (
        "Birinci paragraf STM32 hakkındadır.\n\n"
        "İkinci paragraf PWM hakkındadır.\n\n"
        "Üçüncü paragraf PID hakkındadır."
    )

    chunks = split_into_chunks(
        text=text,
        max_characters=80,
        overlap_characters=10,
    )

    assert len(chunks) >= 2

    combined = "\n\n".join(
        chunks
    )

    assert "Birinci paragraf STM32 hakkındadır." in combined
    assert "İkinci paragraf PWM hakkındadır." in combined
    assert "Üçüncü paragraf PID hakkındadır." in combined


def test_short_paragraphs_can_share_one_chunk() -> None:
    text = (
        "STM32 bir mikrodenetleyicidir.\n\n"
        "PWM motor hızını kontrol eder."
    )

    chunks = split_into_chunks(
        text=text,
        max_characters=100,
        overlap_characters=10,
    )

    assert len(chunks) == 1

    assert "STM32" in chunks[0]
    assert "PWM" in chunks[0]


def test_long_paragraph_uses_fallback_chunking() -> None:
    text = " ".join(
        f"kelime{i}"
        for i in range(120)
    )

    chunks = split_into_chunks(
        text=text,
        max_characters=100,
        overlap_characters=20,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk) <= 100


def test_multiple_blank_lines_are_supported() -> None:
    text = (
        "Birinci paragraf."
        "\n\n\n\n"
        "İkinci paragraf."
    )

    chunks = split_into_chunks(
        text=text,
        max_characters=100,
        overlap_characters=10,
    )

    assert len(chunks) == 1
    assert "Birinci paragraf." in chunks[0]
    assert "İkinci paragraf." in chunks[0]



if __name__ == "__main__":
    run_tests()