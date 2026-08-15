import argparse
import logging
from collections.abc import Sequence

from src.assistant import RAGAssistant
from src.cli import run_chat_session
from src.config import (
    DATABASE_PATH,
    RAW_DATA_DIRECTORY,
)
from src.database import (
    get_document_count,
    initialize_database,
)
from src.document_loader import find_text_files
from src.embeddings import (
    unload_embedding_model,
    warm_up_embedding_model,
)
from src.ingestion import ingest_text_files
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="Local RAG AI Assistant",
        description=(
            "Run and manage the local Foundry Local RAG assistant."
        ),
    )

    actions = parser.add_mutually_exclusive_group()

    actions.add_argument(
        "--chat",
        action="store_true",
        help="Start the interactive RAG chat.",
    )

    actions.add_argument(
        "--ingest",
        action="store_true",
        help=(
            "Process documents in data/raw and add new "
            "chunks to the knowledge base."
        ),
    )

    actions.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Clear the current knowledge base and rebuild it "
            "from documents in data/raw."
        ),
    )

    actions.add_argument(
        "--stats",
        action="store_true",
        help="Display local knowledge-base statistics.",
    )

    return parser


def run_chat() -> None:
    """Start the interactive RAG chat session."""

    initialize_database()

    print(
        "Yerel modeller hazırlanıyor..."
    )

    warm_up_embedding_model()

    with RAGAssistant() as assistant:
        assistant.warm_up()

        print(
            "Modeller hazır.\n"
        )

        run_chat_session(
            assistant
        )


def run_ingestion(
    reset_database: bool,
) -> None:
    """Run document ingestion."""

    action_name = (
        "Knowledge base yeniden oluşturuluyor."
        if reset_database
        else "Yeni belgeler işleniyor."
    )

    print(action_name)

    inserted_count = ingest_text_files(
        reset_database=reset_database,
    )

    total_count = get_document_count()

    print("\nIngestion tamamlandı.")
    print(
        f"Yeni eklenen chunk: {inserted_count}"
    )
    print(
        f"Toplam kayıtlı chunk: {total_count}"
    )


def show_stats() -> None:
    """Display local knowledge-base statistics."""

    initialize_database()

    document_count = get_document_count()

    try:
        source_files = find_text_files(
            RAW_DATA_DIRECTORY
        )
        source_count = len(source_files)

    except (FileNotFoundError, ValueError):
        source_count = 0

    if DATABASE_PATH.exists():
        database_size_bytes = (
            DATABASE_PATH.stat().st_size
        )
    else:
        database_size_bytes = 0

    database_size_kb = (
        database_size_bytes / 1024
    )

    print("Local RAG Knowledge Base")
    print("=" * 50)
    print(
        f"Kaynak TXT dosyası : {source_count}"
    )
    print(
        f"Kayıtlı chunk      : {document_count}"
    )
    print(
        f"Veritabanı boyutu  : {database_size_kb:.2f} KB"
    )
    print(
        f"Veritabanı yolu    : {DATABASE_PATH}"
    )


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Application entry point."""

    configure_logging()

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.ingest:
            run_ingestion(
                reset_database=False
            )

        elif args.reset:
            run_ingestion(
                reset_database=True
            )

        elif args.stats:
            show_stats()

        else:
            # Default behavior and --chat both start chat mode.
            run_chat()

    except Exception:
        logger.exception(
            "Uygulama çalışırken beklenmeyen bir hata oluştu."
        )

    finally:
        unload_embedding_model()


if __name__ == "__main__":
    main()