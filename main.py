import logging

from src.assistant import RAGAssistant
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)


def main() -> None:
    """Application entry point."""

    configure_logging()

    print("Local RAG AI Assistant")
    print("-" * 50)

    question = (
        "Çizgi takip sisteminde hata nasıl hesaplanır?"
    )

    print(f"\nSoru: {question}")

    try:
        with RAGAssistant(top_k=3) as assistant:
            response = assistant.answer(question)

        print("\n" + "=" * 50)
        print("RAG CEVABI")
        print("=" * 50)
        print(response["answer"])


    except Exception:
        logger.exception(
            "Uygulama çalışırken beklenmeyen bir hata oluştu."
        )


if __name__ == "__main__":
    main()
    