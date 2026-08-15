from collections.abc import Callable

from src.assistant import RAGAssistant


EXIT_COMMANDS = {
    "exit",
    "quit",
    "q",
    "çık",
    "çıkış",
}

CLEAR_COMMANDS = {
    "/clear",
    "/temizle",
}

HISTORY_COMMANDS = {
    "/history",
    "/geçmiş",
}

def _format_retrieved_documents(
    retrieved_documents: list[dict],
) -> list[str]:
    """
    Format retrieved document metadata for CLI display.
    """

    if not retrieved_documents:
        return [
            "Retrieved document bulunamadı."
        ]

    lines: list[str] = [
        "",
        "=" * 50,
        "RETRIEVED DOCUMENTS",
        "=" * 50,
    ]

    for index, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        source = document.get(
            "source",
            "Unknown",
        )

        score = document.get(
            "score",
            0.0,
        )

        lines.extend(
            [
                f"[{index}]",
                f"Source : {source}",
                f"Score  : {float(score):.4f}",
                "",
            ]
        )

    return lines

def run_chat_session(
    assistant: RAGAssistant,
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], None] = print,
) -> None:
    """
    Run an interactive terminal chat session.

    Args:
        assistant: RAG assistant used to answer questions.
        input_function: Function used to read user input.
        output_function: Function used to display output.
    """

    output_function("Local RAG AI Assistant")
    output_function("-" * 50)
    output_function(
        "Çıkmak için 'exit', 'quit', 'q' veya 'çıkış' yaz."
    )
    output_function(
        "Komutlar: /clear, /history"
    )

    while True:
        try:
            question = input_function(
                "\nSoru: "
            ).strip()

        except (EOFError, KeyboardInterrupt):
            output_function(
                "\nSohbet sonlandırıldı."
            )
            break

        if not question:
            output_function(
                "Lütfen boş olmayan bir soru gir."
            )
            continue

        normalized_question = question.lower()

        if normalized_question in EXIT_COMMANDS:
            output_function(
                "Sohbet sonlandırıldı."
            )
            break

        if normalized_question in CLEAR_COMMANDS:
            assistant.clear_memory()

            output_function(
                "Konuşma hafızası temizlendi."
            )
            continue

        if normalized_question in HISTORY_COMMANDS:
            history = (
                assistant.get_conversation_history()
            )

            if not history:
                output_function(
                    "Konuşma geçmişi boş."
                )
            else:
                output_function(
                    "\n" + "=" * 50
                )
                output_function(
                    "KONUŞMA GEÇMİŞİ"
                )
                output_function(
                    "=" * 50
                )
                output_function(
                    history
                )

            continue

        try:
            response = assistant.answer(
                question
            )

        except Exception as error:
            output_function(
                f"Bir hata oluştu: {error}"
            )
            continue

        output_function(
            "\n" + "=" * 50
        )
        output_function(
            "RAG CEVABI"
        )
        output_function(
            "=" * 50
        )
        output_function(
            response["answer"]
        )
        
        for line in _format_retrieved_documents(
           response["retrieved_documents"]
        ):
          output_function(line)