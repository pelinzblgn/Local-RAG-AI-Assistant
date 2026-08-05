from src.prompts import (
    DEFAULT_FALLBACK_MESSAGE,
    PromptBuilder,
)


SAMPLE_DOCUMENTS = [
    {
        "source": "pid_notes.txt",
        "content": (
            "Çizgi takip sisteminde hata, çizginin "
            "algılanan konumu ile hedef merkez arasındaki farktır."
        ),
        "score": 0.82,
    },
    {
        "source": "stm32_notes.txt",
        "content": (
            "PWM duty cycle değeri motor hızını kontrol eder."
        ),
        "score": 0.71,
    },
]


def test_system_prompt_contains_grounding_rules() -> None:
    builder = PromptBuilder()

    system_prompt = builder.build_system_prompt()

    assert "only the supplied local document context" in system_prompt
    assert "Do not use outside knowledge" in system_prompt
    assert DEFAULT_FALLBACK_MESSAGE in system_prompt
    assert "Kaynaklar:" in system_prompt


def test_context_contains_sources_and_contents() -> None:
    builder = PromptBuilder()

    context = builder.build_context(SAMPLE_DOCUMENTS)

    assert "[Belge 1]" in context
    assert "[Belge 2]" in context
    assert "pid_notes.txt" in context
    assert "stm32_notes.txt" in context
    assert "Çizgi takip sisteminde hata" in context
    assert "PWM duty cycle" in context


def test_user_prompt_contains_question() -> None:
    builder = PromptBuilder()

    prompt = builder.build_user_prompt(
        question="Çizgi takip hatası nedir?",
        retrieved_documents=SAMPLE_DOCUMENTS,
    )

    assert "Çizgi takip hatası nedir?" in prompt
    assert "YEREL BELGE BAĞLAMI" in prompt
    assert "KULLANICI SORUSU" in prompt


def test_build_returns_system_and_user_prompts() -> None:
    builder = PromptBuilder()

    system_prompt, user_prompt = builder.build(
        question="Motor hızı nasıl kontrol edilir?",
        retrieved_documents=SAMPLE_DOCUMENTS,
    )

    assert isinstance(system_prompt, str)
    assert isinstance(user_prompt, str)
    assert system_prompt.strip()
    assert user_prompt.strip()


def test_question_whitespace_is_removed() -> None:
    builder = PromptBuilder()

    prompt = builder.build_user_prompt(
        question="   PID nedir?   ",
        retrieved_documents=SAMPLE_DOCUMENTS,
    )

    assert "PID nedir?" in prompt
    assert "   PID nedir?   " not in prompt


def test_empty_question_raises_error() -> None:
    builder = PromptBuilder()

    try:
        builder.build_user_prompt(
            question="   ",
            retrieved_documents=SAMPLE_DOCUMENTS,
        )
    except ValueError as error:
        assert "Question cannot be empty" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for empty question."
    )


def test_empty_document_list_raises_error() -> None:
    builder = PromptBuilder()

    try:
        builder.build_context([])
    except ValueError as error:
        assert "At least one retrieved document" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for empty document list."
    )


def test_missing_source_raises_error() -> None:
    builder = PromptBuilder()

    try:
        builder.build_context(
            [
                {
                    "content": "Test content",
                }
            ]
        )
    except ValueError as error:
        assert "string source" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for missing source."
    )


def test_missing_content_raises_error() -> None:
    builder = PromptBuilder()

    try:
        builder.build_context(
            [
                {
                    "source": "test.txt",
                }
            ]
        )
    except ValueError as error:
        assert "string content" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for missing content."
    )


def test_empty_fallback_message_raises_error() -> None:
    try:
        PromptBuilder(fallback_message="   ")
    except ValueError as error:
        assert "Fallback message cannot be empty" in str(error)
        return

    raise AssertionError(
        "Expected ValueError for empty fallback message."
    )


def run_tests() -> None:
    tests = [
        test_system_prompt_contains_grounding_rules,
        test_context_contains_sources_and_contents,
        test_user_prompt_contains_question,
        test_build_returns_system_and_user_prompts,
        test_question_whitespace_is_removed,
        test_empty_question_raises_error,
        test_empty_document_list_raises_error,
        test_missing_source_raises_error,
        test_missing_content_raises_error,
        test_empty_fallback_message_raises_error,
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