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
    delete_documents_by_source,
    get_document_count,
    initialize_database,
    insert_document,
    upsert_source_file,
)
from src.document_loader import (
    collect_document_files,
    find_text_files,
    is_supported_document_file,
    read_document_file,
    read_text_file,
)
from src.embeddings import generate_embeddings


logger = logging.getLogger(__name__)


EXTERNAL_SOURCE_PREFIX = "external"


class ChunkRecord(TypedDict):
    """Structure of a chunk prepared for database insertion."""

    content: str
    source: str


class IngestionResult(TypedDict):
    """Summary of a user-selected ingestion operation."""

    source_path: str
    file_count: int
    inserted_chunks: int
    sources: list[str]
    recursive: bool


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


def _validate_source_name(
    source_name: str,
) -> str:
    """
    Validate and normalize a source identifier.

    Source identifiers are stored in the local knowledge base and
    displayed to the user. They must not be empty.
    """

    if not isinstance(source_name, str):
        raise TypeError(
            "source_name must be a string."
        )

    clean_source_name = source_name.strip()

    if not clean_source_name:
        raise ValueError(
            "source_name cannot be empty."
        )

    return clean_source_name


def _build_external_source_name(
    file_path: Path,
    selected_path: Path,
) -> str:
    """
    Build a privacy-safe source identifier for an external document.

    Absolute computer paths are intentionally not stored in the
    knowledge base.

    For a selected directory, the directory name and relative file
    path are retained so nested files remain distinguishable.

    For a selected single file, only the file name is retained.

    Examples:
        external/stm32_notes.txt
        external/lecture_notes.pdf
        external/project_report.docx
        external/course_notes/week_1/intro.pdf
    """

    if selected_path.is_file():
        relative_part = Path(
            file_path.name
        )

    else:
        try:
            relative_file = file_path.relative_to(
                selected_path
            )

        except ValueError as error:
            raise ValueError(
                "External file is outside the selected directory."
            ) from error

        relative_part = (
            Path(selected_path.name)
            / relative_file
        )

    safe_parts = [
        part
        for part in relative_part.parts
        if part not in {
            "",
            ".",
            "..",
        }
    ]

    if not safe_parts:
        raise ValueError(
            "Could not generate a valid external source name."
        )

    return (
        EXTERNAL_SOURCE_PREFIX
        + "/"
        + "/".join(safe_parts)
    )


def _collect_chunk_records(
    text_files: list[Path],
) -> list[ChunkRecord]:
    """
    Read managed TXT files and prepare chunk records.

    This helper intentionally remains TXT-specific because the
    project's managed data/raw workflow and Smart Folder Sync are
    currently based on TXT documents.
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


def ingest_document_file(
    file_path: Path,
    database_path: Path = DATABASE_PATH,
    *,
    source_name: str | None = None,
    replace_existing: bool = False,
) -> int:
    """
    Ingest one supported document into the local knowledge base.

    Supported document formats:
        - UTF-8 TXT
        - text-based PDF
        - DOCX

    Args:
        file_path:
            Document file to process.

        database_path:
            SQLite knowledge-base path.

        source_name:
            Optional source identifier stored in the knowledge base.
            If omitted, the file name is used.

        replace_existing:
            Remove existing chunks belonging to the same source before
            storing the newly generated chunks.

    Returns:
        Number of newly inserted chunks.

    Notes:
        Text extraction, chunking, and embedding generation are
        completed before existing source records are removed. This
        prevents a failed import from destroying a previously valid
        indexed version of the same source.
    """

    if not isinstance(file_path, Path):
        raise TypeError(
            "file_path must be a pathlib.Path object."
        )

    if not isinstance(database_path, Path):
        raise TypeError(
            "database_path must be a pathlib.Path object."
        )

    if not isinstance(replace_existing, bool):
        raise TypeError(
            "replace_existing must be a bool."
        )

    initialize_database(
        database_path
    )

    if not is_supported_document_file(
        file_path
    ):
        raise ValueError(
            f"Unsupported document type: "
            f"{file_path.suffix or '<no extension>'}"
        )

    resolved_source_name = (
        file_path.name
        if source_name is None
        else _validate_source_name(
            source_name
        )
    )

    text = read_document_file(
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

    if replace_existing:
        deleted_count = delete_documents_by_source(
            source=resolved_source_name,
            database_path=database_path,
        )

        if deleted_count:
            logger.info(
                (
                    "%s kaynağı için %d eski chunk "
                    "silindi."
                ),
                resolved_source_name,
                deleted_count,
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
            source=resolved_source_name,
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
        source=resolved_source_name,
        file_hash=_calculate_file_hash(
            file_path
        ),
        modified_at=file_path.stat().st_mtime,
        database_path=database_path,
    )

    logger.info(
        (
            "%s dosyası işlendi. "
            "Kaynak: %s, yeni chunk: %d"
        ),
        file_path.name,
        resolved_source_name,
        inserted_count,
    )

    return inserted_count


def ingest_text_file(
    file_path: Path,
    database_path: Path = DATABASE_PATH,
    *,
    source_name: str | None = None,
    replace_existing: bool = False,
) -> int:
    """
    Ingest one TXT file into the local knowledge base.

    This function is retained for backward compatibility with the
    existing managed TXT ingestion and Smart Folder Sync workflow.

    Args:
        file_path:
            TXT file to process.

        database_path:
            SQLite knowledge-base path.

        source_name:
            Optional source identifier stored in the knowledge base.

        replace_existing:
            Remove existing chunks belonging to the same source before
            storing the newly generated chunks.

    Returns:
        Number of newly inserted chunks.
    """

    if not isinstance(file_path, Path):
        raise TypeError(
            "file_path must be a pathlib.Path object."
        )

    if file_path.suffix.lower() != ".txt":
        raise ValueError(
            "ingest_text_file only supports .txt files."
        )

    return ingest_document_file(
        file_path=file_path,
        database_path=database_path,
        source_name=source_name,
        replace_existing=replace_existing,
    )


def ingest_selected_source(
    source_path: Path,
    database_path: Path = DATABASE_PATH,
    *,
    recursive: bool = False,
    external_source_name: str | None = None,
) -> IngestionResult:
    """
    Add user-selected documents to the local knowledge base.

    Supported external formats:
        - UTF-8 TXT
        - text-based PDF
        - DOCX

    The application processes only the file or directory explicitly
    selected by the user. It does not scan arbitrary computer
    directories.

    Absolute local paths are not stored in document-source metadata.
    External sources receive privacy-safe identifiers beginning with
    ``external/``.

    A custom external source name may be supplied when the physical
    file path is temporary, such as a file received through the web
    API. This allows the original user-visible file name to be
    preserved without exposing temporary operating-system paths.

    If a previously imported source is selected again, its existing
    chunks are replaced instead of leaving stale duplicate data.

    Args:
        source_path:
            Supported document file or directory explicitly selected
            by the user.

        database_path:
            SQLite knowledge-base path.

        recursive:
            When source_path is a directory, include supported
            documents in subdirectories if True.

        external_source_name:
            Optional original file name used for a single selected
            file. This is primarily intended for web uploads whose
            physical source_path is a temporary file.

    Returns:
        Structured ingestion summary.

    Raises:
        TypeError:
            If arguments have invalid types.

        FileNotFoundError:
            If source_path does not exist.

        ValueError:
            If the selected path, document type, or external source
            name is unsupported.

        RuntimeError:
            If no supported documents were found or indexed.
    """

    if not isinstance(source_path, Path):
        raise TypeError(
            "source_path must be a pathlib.Path object."
        )

    if not isinstance(database_path, Path):
        raise TypeError(
            "database_path must be a pathlib.Path object."
        )

    if not isinstance(recursive, bool):
        raise TypeError(
            "recursive must be a bool."
        )

    if (
        external_source_name is not None
        and not isinstance(
            external_source_name,
            str,
        )
    ):
        raise TypeError(
            "external_source_name must be a string or None."
        )

    if (
        external_source_name is not None
        and not source_path.is_file()
    ):
        raise ValueError(
            "external_source_name can only be used "
            "with a single selected file."
        )

    initialize_database(
        database_path
    )

    document_files = collect_document_files(
        source_path,
        recursive=recursive,
    )

    if not document_files:
        raise RuntimeError(
            "No supported documents were found in "
            "the selected source."
        )

    logger.info(
        (
            "Kullanıcı tarafından seçilen %d "
            "belge işlenecek."
        ),
        len(document_files),
    )

    total_inserted_chunks = 0
    indexed_sources: list[str] = []

    for file_path in document_files:
        if external_source_name is not None:
            safe_file_name = Path(
                external_source_name
            ).name.strip()

            if not safe_file_name:
                raise ValueError(
                    "External source name cannot be empty."
                )

            if not is_supported_document_file(
                Path(safe_file_name)
            ):
                raise ValueError(
                    "External source name must use a "
                    "supported extension: .txt, .pdf, or .docx."
                )

            source_name = (
                EXTERNAL_SOURCE_PREFIX
                + "/"
                + safe_file_name
            )

        else:
            source_name = _build_external_source_name(
                file_path=file_path,
                selected_path=source_path,
            )

        inserted_count = ingest_document_file(
            file_path=file_path,
            database_path=database_path,
            source_name=source_name,
            replace_existing=True,
        )

        total_inserted_chunks += (
            inserted_count
        )

        indexed_sources.append(
            source_name
        )

    logger.info(
        (
            "Harici belge ingestion tamamlandı. "
            "Dosya: %d, yeni chunk: %d"
        ),
        len(document_files),
        total_inserted_chunks,
    )

    return {
        "source_path": str(
            source_path
        ),
        "file_count": len(
            document_files
        ),
        "inserted_chunks": total_inserted_chunks,
        "sources": indexed_sources,
        "recursive": recursive,
    }


def ingest_text_files(
    reset_database: bool = False,
    raw_data_directory: Path = RAW_DATA_DIRECTORY,
    database_path: Path = DATABASE_PATH,
) -> int:
    """
    Read managed TXT files, create embeddings and store chunks
    in SQLite.

    Source fingerprints are recorded for incremental sync.

    This function intentionally preserves the project's original
    managed ``data/raw`` TXT workflow. PDF and DOCX support is
    currently provided for explicitly user-selected external
    documents.

    Args:
        reset_database:
            Clear the existing knowledge base first.

        raw_data_directory:
            Directory containing managed TXT documents.

        database_path:
            SQLite knowledge-base path.

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

    if len(embeddings) != len(
        chunk_records
    ):
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