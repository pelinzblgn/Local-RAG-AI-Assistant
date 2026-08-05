import logging
from pathlib import Path
from typing import TypedDict

from src.chunker import split_into_chunks
from src.config import RAW_DATA_DIRECTORY
from src.database import (
    delete_all_documents,
    get_document_count,
    initialize_database,
    insert_document,
)
from src.document_loader import (
    find_text_files,
    read_text_file,
)
from src.embeddings import generate_embeddings


logger = logging.getLogger(__name__)


class ChunkRecord(TypedDict):
    """Structure of a chunk prepared for database insertion."""

    content: str
    source: str


def _collect_chunk_records(
    text_files: list[Path],
) -> list[ChunkRecord]:
    """
    Read text files and prepare chunk records.

    Invalid or empty TXT files are skipped with a warning.
    """

    chunk_records: list[ChunkRecord] = []

    for file_path in text_files:
        logger.info(
            "Belge okunuyor: %s",
            file_path.name,
        )

        try:
            text = read_text_file(
                file_path
            )
        except ValueError as error:
            logger.warning(
                "Belge atlandı: %s",
                error,
            )
            continue

        chunks = split_into_chunks(text)

        logger.info(
            "%s dosyasından %d chunk üretildi.",
            file_path.name,
            len(chunks),
        )

        for chunk in chunks:
            chunk_records.append(
                {
                    "content": chunk,
                    "source": file_path.name,
                }
            )

    return chunk_records


def ingest_text_files(
    reset_database: bool = False,
    raw_data_directory: Path = RAW_DATA_DIRECTORY,
) -> int:
    """
    Read TXT files, create embeddings and store chunks in SQLite.

    Args:
        reset_database: Delete current document records first.
        raw_data_directory: Directory containing TXT documents.

    Returns:
        Number of newly stored chunks.

    Raises:
        RuntimeError: If the directory contains no TXT files
            or no valid chunks can be generated.
    """

    initialize_database()

    if reset_database:
        logger.warning(
            "Mevcut belge kayıtları siliniyor."
        )
        delete_all_documents()

    text_files = find_text_files(
        raw_data_directory
    )

    if not text_files:
        raise RuntimeError(
            f"No TXT files were found in "
            f"{raw_data_directory}."
        )

    logger.info(
        "%d TXT dosyası bulundu.",
        len(text_files),
    )

    chunk_records = _collect_chunk_records(
        text_files
    )

    if not chunk_records:
        raise RuntimeError(
            "No valid text chunks were generated."
        )

    contents = [
        record["content"]
        for record in chunk_records
    ]

    logger.info(
        "Toplam %d chunk için embedding oluşturuluyor.",
        len(contents),
    )

    embeddings = generate_embeddings(
        contents
    )

    if len(embeddings) != len(chunk_records):
        raise RuntimeError(
            "Chunk and embedding counts do not match."
        )

    count_before = get_document_count()

    for record, embedding in zip(
        chunk_records,
        embeddings,
        strict=True,
    ):
        insert_document(
            content=record["content"],
            source=record["source"],
            embedding=embedding,
        )

    count_after = get_document_count()
    inserted_count = count_after - count_before

    logger.info(
        "Ingestion tamamlandı. İşlenen: %d, yeni kayıt: %d",
        len(chunk_records),
        inserted_count,
    )

    return inserted_count