import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import (
    DATABASE_PATH,
    RAW_DATA_DIRECTORY,
)
from src.database import (
    delete_documents_by_source,
    delete_source_file,
    get_all_source_files,
    initialize_database,
)
from src.document_loader import find_text_files
from src.ingestion import (
    EXTERNAL_SOURCE_PREFIX,
    ingest_text_file,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncResult:
    """Summary of a knowledge-base synchronization operation."""

    new_files: tuple[str, ...]
    modified_files: tuple[str, ...]
    deleted_files: tuple[str, ...]
    unchanged_files: tuple[str, ...]
    inserted_chunks: int
    deleted_chunks: int

    @property
    def changed_file_count(self) -> int:
        """Return number of files that changed the knowledge base."""

        return (
            len(self.new_files)
            + len(self.modified_files)
            + len(self.deleted_files)
        )

    @property
    def has_changes(self) -> bool:
        """Return whether synchronization changed the knowledge base."""

        return self.changed_file_count > 0


def calculate_file_hash(
    file_path: Path,
) -> str:
    """
    Calculate SHA-256 fingerprint of a file.

    File content rather than modification time is used as the
    authoritative change-detection signal.
    """

    if not isinstance(file_path, Path):
        raise TypeError(
            "file_path must be a pathlib.Path object."
        )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

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


def _is_external_source(
    source_name: str,
) -> bool:
    """
    Return whether a source belongs to user-selected external data.

    External sources are intentionally excluded from the managed
    ``data/raw`` Smart Sync lifecycle.

    This prevents the normal knowledge-base synchronization process
    from treating user-imported documents as deleted simply because
    they are not present inside ``data/raw``.
    """

    if not isinstance(source_name, str):
        return False

    prefix = (
        EXTERNAL_SOURCE_PREFIX
        + "/"
    )

    return source_name.startswith(
        prefix
    )


def _build_current_file_map(
    raw_data_directory: Path,
) -> dict[str, Path]:
    """
    Return managed TXT source files indexed by file name.

    Only the project's configured ``data/raw`` directory participates
    in this map. User-selected external sources are managed separately.
    """

    text_files = find_text_files(
        raw_data_directory
    )

    return {
        file_path.name: file_path
        for file_path in text_files
    }


def _build_managed_source_map(
    stored_records: list[dict],
) -> dict[str, dict]:
    """
    Return manifest records managed by the normal Smart Sync process.

    Records whose source identifiers begin with ``external/`` are
    deliberately excluded so that normal synchronization cannot
    modify or delete user-selected external knowledge sources.
    """

    managed_records: dict[str, dict] = {}

    for record in stored_records:
        source_name = record["source"]

        if _is_external_source(
            source_name
        ):
            logger.debug(
                (
                    "Harici kaynak normal Smart Sync "
                    "kapsamı dışında bırakıldı: %s"
                ),
                source_name,
            )
            continue

        managed_records[
            source_name
        ] = record

    return managed_records


def synchronize_knowledge_base(
    raw_data_directory: Path = RAW_DATA_DIRECTORY,
    database_path: Path = DATABASE_PATH,
) -> SyncResult:
    """
    Synchronize managed local TXT documents with the SQLite
    knowledge base.

    Only documents belonging to the managed ``data/raw`` source
    directory participate in this synchronization.

    User-selected sources whose identifiers begin with
    ``external/`` are intentionally preserved and are not treated
    as NEW, MODIFIED, DELETED, or UNCHANGED by this operation.

    Behavior for managed sources:

        NEW:
            Embed and index only the new file.

        MODIFIED:
            Remove old chunks for that source and index the
            current file contents again.

        DELETED:
            Remove document chunks and source metadata.

        UNCHANGED:
            Skip the file completely. No embedding generation
            is performed.

    Args:
        raw_data_directory:
            Directory containing managed TXT source files.

        database_path:
            SQLite knowledge-base path.

    Returns:
        Structured synchronization summary.
    """

    initialize_database(
        database_path
    )

    current_files = _build_current_file_map(
        raw_data_directory
    )

    stored_records = get_all_source_files(
        database_path
    )

    stored_by_name = _build_managed_source_map(
        stored_records
    )

    new_files: list[str] = []
    modified_files: list[str] = []
    deleted_files: list[str] = []
    unchanged_files: list[str] = []

    inserted_chunks = 0
    deleted_chunks = 0

    # ======================================================
    # New / Modified / Unchanged managed sources
    # ======================================================

    for source_name in sorted(
        current_files
    ):
        file_path = current_files[
            source_name
        ]

        current_hash = calculate_file_hash(
            file_path
        )

        stored_record = stored_by_name.get(
            source_name
        )

        # --------------------------------------------------
        # NEW
        # --------------------------------------------------

        if stored_record is None:
            logger.info(
                "Yeni kaynak bulundu: %s",
                source_name,
            )

            inserted_chunks += ingest_text_file(
                file_path=file_path,
                database_path=database_path,
            )

            new_files.append(
                source_name
            )

            continue

        # --------------------------------------------------
        # UNCHANGED
        # --------------------------------------------------

        if (
            stored_record["file_hash"]
            == current_hash
        ):
            logger.debug(
                "Kaynak değişmedi: %s",
                source_name,
            )

            unchanged_files.append(
                source_name
            )

            continue

        # --------------------------------------------------
        # MODIFIED
        # --------------------------------------------------

        logger.info(
            "Değişen kaynak bulundu: %s",
            source_name,
        )

        removed_count = (
            delete_documents_by_source(
                source=source_name,
                database_path=database_path,
            )
        )

        deleted_chunks += (
            removed_count
        )

        try:
            new_chunk_count = ingest_text_file(
                file_path=file_path,
                database_path=database_path,
            )

        except Exception:
            logger.exception(
                (
                    "Değişen kaynak yeniden "
                    "indekslenemedi: %s"
                ),
                source_name,
            )

            raise

        inserted_chunks += (
            new_chunk_count
        )

        modified_files.append(
            source_name
        )

    # ======================================================
    # Deleted managed sources
    # ======================================================

    current_source_names = set(
        current_files
    )

    for source_name in sorted(
        stored_by_name
    ):
        if (
            source_name
            in current_source_names
        ):
            continue

        logger.info(
            "Silinen yönetilen kaynak bulundu: %s",
            source_name,
        )

        removed_count = (
            delete_documents_by_source(
                source=source_name,
                database_path=database_path,
            )
        )

        deleted_chunks += (
            removed_count
        )

        delete_source_file(
            source=source_name,
            database_path=database_path,
        )

        deleted_files.append(
            source_name
        )

    result = SyncResult(
        new_files=tuple(
            new_files
        ),
        modified_files=tuple(
            modified_files
        ),
        deleted_files=tuple(
            deleted_files
        ),
        unchanged_files=tuple(
            unchanged_files
        ),
        inserted_chunks=inserted_chunks,
        deleted_chunks=deleted_chunks,
    )

    external_source_count = sum(
        1
        for record in stored_records
        if _is_external_source(
            record["source"]
        )
    )

    logger.info(
        (
            "Knowledge base sync tamamlandı | "
            "Yeni: %d | "
            "Değişen: %d | "
            "Silinen: %d | "
            "Değişmeyen: %d | "
            "Korunan harici kaynak: %d | "
            "Eklenen chunk: %d | "
            "Silinen chunk: %d"
        ),
        len(result.new_files),
        len(result.modified_files),
        len(result.deleted_files),
        len(result.unchanged_files),
        external_source_count,
        result.inserted_chunks,
        result.deleted_chunks,
    )

    return result