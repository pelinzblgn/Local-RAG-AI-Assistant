import pytest

from src.prompts import (
    DEFAULT_FALLBACK_MESSAGE,
    PromptBuilder,
)


SAMPLE_DOCUMENTS = [
    {
        "id": 1,
        "content": (
            "Çizgi takip sisteminde hata, "
            "çizginin algılanan konumu ile "
            "hedef merkez arasındaki farktır."
        ),
        "source": "pid_notes.txt",
        "score": 0.90,
    },
    {
        "id": 2,
        "content": (
            "PWM duty cycle değeri motor "
            "hızını kontrol eder."
        ),
        "source": "stm32_notes.txt",
        "score": 0.80,
    },
]


def test_system_prompt_contains_grounding_rules() -> None:
    builder = PromptBuilder()

    system_prompt = builder.build_system_prompt()

    assert (
        "only the supplied local document content"
        in system_prompt
    )

    assert (
        DEFAULT_FALLBACK_MESSAGE
        in system_prompt
    )

    assert (
        "Do not use outside knowledge."
        in system_prompt
    )

    assert (
        "Do not create a source list."
        in system_prompt
    )

    assert (
        "Source attribution is handled separately"
        in system_prompt
    )


def test_context_contains_sources_and_contents() -> None:
    builder = PromptBuilder()

    context = builder.build_context(
        SAMPLE_DOCUMENTS
    )

    assert (
        "<document_id>1</document_id>"
        in context
    )

    assert (
        "<document_id>2</document_id>"
        in context
    )

    assert (
        "<file_name>pid_notes.txt</file_name>"
        in context
    )

    assert (
        "<file_name>stm32_notes.txt</file_name>"
        in context
    )

    assert (
        "Çizgi takip sisteminde hata"
        in context
    )

    assert (
        "PWM duty cycle değeri motor hızını kontrol eder."
        in context
    )

    assert "<content>" in context
    assert "</content>" in context


def test_user_prompt_contains_question() -> None:
    builder = PromptBuilder()

    prompt = builder.build_user_prompt(
        question="Çizgi takip hatası nedir?",
        retrieved_documents=SAMPLE_DOCUMENTS,
    )

    assert (
        "Çizgi takip hatası nedir?"
        in prompt
    )

    assert "<local_documents>" in prompt
    assert "</local_documents>" in prompt

    assert "<current_question>" in prompt
    assert "</current_question>" in prompt

    assert (
        "YEREL BELGE BAĞLAMI"
        not in prompt
    )

    assert (
        "GÜNCEL KULLANICI SORUSU"
        not in prompt
    )


def test_build_returns_system_and_user_prompts() -> None:
    builder = PromptBuilder()

    system_prompt, user_prompt = builder.build(
        question="PWM nedir?",
        retrieved_documents=SAMPLE_DOCUMENTS,
    )

    assert isinstance(
        system_prompt,
        str,
    )

    assert isinstance(
        user_prompt,
        str,
    )

    assert system_prompt.strip()
    assert user_prompt.strip()

    assert (
        "PWM nedir?"
        in user_prompt
    )


def test_question_whitespace_is_removed() -> None:
    builder = PromptBuilder()

    prompt = builder.build_user_prompt(
        question="   PWM nedir?   ",
        retrieved_documents=SAMPLE_DOCUMENTS,
    )

    assert (
        "<current_question>\n"
        "PWM nedir?\n"
        "</current_question>"
        in prompt
    )


def test_empty_question_raises_error() -> None:
    builder = PromptBuilder()

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        builder.build_user_prompt(
            question="   ",
            retrieved_documents=SAMPLE_DOCUMENTS,
        )


def test_empty_document_list_raises_error() -> None:
    builder = PromptBuilder()

    with pytest.raises(
        ValueError,
        match="At least one retrieved document",
    ):
        builder.build_context([])


def test_missing_source_raises_error() -> None:
    builder = PromptBuilder()

    documents = [
        {
            "content": "Test content",
        }
    ]

    with pytest.raises(
        ValueError,
        match="string source",
    ):
        builder.build_context(
            documents
        )


def test_missing_content_raises_error() -> None:
    builder = PromptBuilder()

    documents = [
        {
            "source": "test.txt",
        }
    ]

    with pytest.raises(
        ValueError,
        match="string content",
    ):
        builder.build_context(
            documents
        )


def test_empty_fallback_message_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Fallback message cannot be empty",
    ):
        PromptBuilder(
            fallback_message="   "
        )


def test_non_string_fallback_message_raises_error() -> None:
    with pytest.raises(
        TypeError,
        match="Fallback message must be a string",
    ):
        PromptBuilder(
            fallback_message=123,  # type: ignore[arg-type]
        )


def test_user_prompt_contains_conversation_history() -> None:
    builder = PromptBuilder()

    history = (
        "[Konuşma 1]\n"
        "Kullanıcı: STM32 nedir?\n"
        "Asistan: STM32 bir mikrodenetleyici ailesidir."
    )

    prompt = builder.build_user_prompt(
        question="Peki PWM ne işe yarar?",
        retrieved_documents=SAMPLE_DOCUMENTS,
        conversation_history=history,
    )

    assert (
        "<conversation_history>"
        in prompt
    )

    assert (
        "</conversation_history>"
        in prompt
    )

    assert (
        "STM32 nedir?"
        in prompt
    )

    assert (
        "STM32 bir mikrodenetleyici ailesidir."
        in prompt
    )

    assert (
        "Peki PWM ne işe yarar?"
        in prompt
    )

    assert (
        "KONUŞMA GEÇMİŞİ"
        not in prompt
    )


def test_empty_conversation_history_is_not_rendered() -> None:
    builder = PromptBuilder()

    prompt = builder.build_user_prompt(
        question="PWM nedir?",
        retrieved_documents=SAMPLE_DOCUMENTS,
        conversation_history="   ",
    )

    assert (
        "<conversation_history>"
        not in prompt
    )

    assert (
        "</conversation_history>"
        not in prompt
    )


def test_non_string_conversation_history_raises_error() -> None:
    builder = PromptBuilder()

    with pytest.raises(
        TypeError,
        match="Conversation history must be a string",
    ):
        builder.build_user_prompt(
            question="PWM nedir?",
            retrieved_documents=SAMPLE_DOCUMENTS,
            conversation_history=123,  # type: ignore[arg-type]
        )


def test_system_prompt_delegates_source_rendering_to_application() -> None:
    builder = PromptBuilder()

    system_prompt = builder.build_system_prompt()

    assert (
        "Do not create a source list."
        in system_prompt
    )

    assert (
        "Source attribution is handled separately"
        in system_prompt
    )

    assert (
        "Do not repeat document metadata"
        in system_prompt
    )


def test_user_prompt_uses_structured_document_boundaries() -> None:
    builder = PromptBuilder()

    documents = [
        {
            "source": "stm32_notes.txt",
            "content": (
                "STM32 PWM ile motor hızını "
                "kontrol edebilir."
            ),
        }
    ]

    user_prompt = builder.build_user_prompt(
        question="PWM ne işe yarar?",
        retrieved_documents=documents,
    )

    assert (
        "<local_documents>"
        in user_prompt
    )

    assert (
        "</local_documents>"
        in user_prompt
    )

    assert (
        "<document>"
        in user_prompt
    )

    assert (
        "</document>"
        in user_prompt
    )

    assert (
        "<file_name>stm32_notes.txt</file_name>"
        in user_prompt
    )

    assert (
        "<current_question>"
        in user_prompt
    )

    assert (
        "</current_question>"
        in user_prompt
    )

    assert (
        "YEREL BELGE BAĞLAMI"
        not in user_prompt
    )

    assert (
        "GÜNCEL KULLANICI SORUSU"
        not in user_prompt
    )


def test_context_rejects_empty_source() -> None:
    builder = PromptBuilder()

    documents = [
        {
            "source": "   ",
            "content": "Test content",
        }
    ]

    with pytest.raises(
        ValueError,
        match="empty source",
    ):
        builder.build_context(
            documents
        )


def test_context_rejects_empty_content() -> None:
    builder = PromptBuilder()

    documents = [
        {
            "source": "test.txt",
            "content": "   ",
        }
    ]

    with pytest.raises(
        ValueError,
        match="empty content",
    ):
        builder.build_context(
            documents
        )


def test_context_rejects_non_mapping_document() -> None:
    builder = PromptBuilder()

    documents = [
        "invalid document",
    ]

    with pytest.raises(
        TypeError,
        match="must be a mapping",
    ):
        builder.build_context(
            documents  # type: ignore[arg-type]
        )


def test_source_names_are_present_only_as_document_metadata() -> None:
    builder = PromptBuilder()

    prompt = builder.build_user_prompt(
        question="PWM nedir?",
        retrieved_documents=[
            {
                "source": "stm32_notes.txt",
                "content": (
                    "PWM motor hızını kontrol eder."
                ),
            }
        ],
    )

    assert (
        "<file_name>stm32_notes.txt</file_name>"
        in prompt
    )

    assert (
        "Kaynaklar:"
        not in prompt
    )


def test_user_prompt_tells_model_not_to_render_sources() -> None:
    builder = PromptBuilder()

    prompt = builder.build_user_prompt(
        question="PWM nedir?",
        retrieved_documents=SAMPLE_DOCUMENTS,
    )

    assert (
        "Do not output sources"
        in prompt
    )

    assert (
        "file names"
        in prompt
    )

    assert (
        "The application will render sources separately."
        in prompt
    )