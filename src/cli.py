from collections.abc import Callable

from src.assistant import RAGAssistant


EXIT_COMMANDS = {
    "exit",
    "quit",
    "q",
    "çık",
    "çıkış",
}


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

    while True:
        try:
            question = input_function("\nSoru: ").strip()

        except (EOFError, KeyboardInterrupt):
            output_function("\nSohbet sonlandırıldı.")
            break

        if not question:
            output_function(
                "Lütfen boş olmayan bir soru gir."
            )
            continue

        if question.lower() in EXIT_COMMANDS:
            output_function("Sohbet sonlandırıldı.")
            break

        try:
            response = assistant.answer(question)

        except Exception as error:
            output_function(
                f"Bir hata oluştu: {error}"
            )
            continue

        output_function("\n" + "=" * 50)
        output_function("RAG CEVABI")
        output_function("=" * 50)
        output_function(response["answer"])