import logging

from src.assistant import RAGAssistant
from src.cli import run_chat_session
from src.database import initialize_database
from src.embeddings import unload_embedding_model
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)


def main() -> None:
    """Start the interactive Local RAG application."""

    configure_logging()
    initialize_database()

    try:
        with RAGAssistant() as assistant:
            run_chat_session(assistant)

    except Exception:
        logger.exception(
            "Uygulama çalışırken beklenmeyen bir hata oluştu."
        )

    finally:
        unload_embedding_model()


if __name__ == "__main__":
    main()