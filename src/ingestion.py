import hashlib
import logging
from pathlib import Path
from typing import TypedDict

from src.chunker import split_into_chunks
from src.config import (
    DATABASE_PATH,
    RAW_DATA_DIRECTORY,
)
from src.database import (
    delete_all_documents,
    get_document_count,
    initialize_database,
    insert_document,
    upsert_source_file,
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


def _calculate_file_hash(
    file_path: Path,
) -> str:
    """Calculate SHA-256 hash of a source file."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            block = file.read(
                64 * 1024
            )

            if not block:
                break

            sha256.update(
                block
            )

    return sha256.hexdigest()


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

        chunks = split_into_chunks(
            text
        )

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


def ingest_text_file(
    file_path: Path,
    database_path: Path = DATABASE_PATH,
) -> int:
    """
    Ingest one TXT file into the local knowledge base.

    Args:
        file_path: TXT file to process.
        database_path: SQLite knowledge-base path.

    Returns:
        Number of newly inserted chunks.
    """

    initialize_database(
        database_path
    )

    text = read_text_file(
        file_path
    )

    chunks = split_into_chunks(
        text
    )

    if not chunks:
        raise RuntimeError(
            f"No valid chunks were generated from "
            f"{file_path.name}."
        )

    logger.info(
        "%s dosyasından %d chunk üretildi.",
        file_path.name,
        len(chunks),
    )

    embeddings = generate_embeddings(
        chunks
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Chunk and embedding counts do not match."
        )

    count_before = get_document_count(
        database_path
    )

    for chunk, embedding in zip(
        chunks,
        embeddings,
        strict=True,
    ):
        insert_document(
            content=chunk,
            source=file_path.name,
            embedding=embedding,
            database_path=database_path,
        )

    count_after = get_document_count(
        database_path
    )

    inserted_count = (
        count_after - count_before
    )

    upsert_source_file(
        source=file_path.name,
        file_hash=_calculate_file_hash(
            file_path
        ),
        modified_at=file_path.stat().st_mtime,
        database_path=database_path,
    )

    logger.info(
        "%s dosyası işlendi. Yeni chunk: %d",
        file_path.name,
        inserted_count,
    )

    return inserted_count


def ingest_text_files(
    reset_database: bool = False,
    raw_data_directory: Path = RAW_DATA_DIRECTORY,
    database_path: Path = DATABASE_PATH,
) -> int:
    """
    Read TXT files, create embeddings and store chunks in SQLite.

    Source fingerprints are recorded for incremental sync.

    Args:
        reset_database: Clear the existing knowledge base first.
        raw_data_directory: Directory containing TXT documents.
        database_path: SQLite knowledge-base path.

    Returns:
        Number of newly stored chunks.
    """

    initialize_database(
        database_path
    )

    if reset_database:
        logger.warning(
            "Mevcut belge kayıtları siliniyor."
        )

        delete_all_documents(
            database_path
        )

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

    count_before = get_document_count(
        database_path
    )

    for record, embedding in zip(
        chunk_records,
        embeddings,
        strict=True,
    ):
        insert_document(
            content=record["content"],
            source=record["source"],
            embedding=embedding,
            database_path=database_path,
        )

    for file_path in text_files:
        upsert_source_file(
            source=file_path.name,
            file_hash=_calculate_file_hash(
                file_path
            ),
            modified_at=file_path.stat().st_mtime,
            database_path=database_path,
        )

    count_after = get_document_count(
        database_path
    )

    inserted_count = (
        count_after - count_before
    )

    logger.info(
        (
            "Ingestion tamamlandı. "
            "İşlenen: %d, yeni kayıt: %d, "
            "manifest kaydı: %d"
        ),
        len(chunk_records),
        inserted_count,
        len(text_files),
    )

    return inserted_count