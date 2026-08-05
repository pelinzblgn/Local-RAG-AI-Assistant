import json
import math
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

from src.config import DATABASE_PATH


class DocumentRecord(TypedDict):
    """Structure of a document record returned from SQLite."""

    id: int
    content: str
    source: str
    embedding: list[float]
    created_at: str


def create_connection(
    database_path: Path = DATABASE_PATH,
) -> sqlite3.Connection:
    """
    Create and return a configured SQLite connection.

    The caller is responsible for closing the returned connection.
    Prefer using ``database_connection`` in application code.
    """

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        database_path,
        timeout=30.0,
    )

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")

    return connection


@contextmanager
def database_connection(
    database_path: Path = DATABASE_PATH,
) -> Iterator[sqlite3.Connection]:
    """
    Provide a SQLite connection and always close it afterward.

    The transaction is committed when the block succeeds and
    rolled back when an exception occurs.
    """

    connection = create_connection(database_path)

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def initialize_database(
    database_path: Path = DATABASE_PATH,
) -> None:
    """
    Create the documents table when it does not exist.

    Duplicate content from the same source is prevented by a
    database-level unique constraint.
    """

    with database_connection(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (source, content)
            )
            """
        )


def _validate_embedding(
    embedding: Sequence[float],
) -> list[float]:
    """
    Validate and normalize an embedding vector.

    Raises:
        ValueError: If the embedding is empty or invalid.
    """

    if len(embedding) == 0:
        raise ValueError(
            "Embedding cannot be empty."
        )

    normalized_embedding: list[float] = []

    for value in embedding:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                "Embedding must contain only numerical values."
            )

        normalized_value = float(value)

        if not math.isfinite(normalized_value):
            raise ValueError(
                "Embedding cannot contain NaN or infinite values."
            )

        normalized_embedding.append(
            normalized_value
        )

    return normalized_embedding


def _serialize_embedding(
    embedding: Sequence[float],
) -> str:
    """Validate an embedding and serialize it as JSON."""

    normalized_embedding = _validate_embedding(
        embedding
    )

    return json.dumps(
        normalized_embedding,
        ensure_ascii=False,
    )


def _deserialize_embedding(
    embedding_json: str,
) -> list[float]:
    """
    Deserialize and validate an embedding stored as JSON.

    Raises:
        ValueError: If the stored embedding is corrupted.
    """

    try:
        embedding = json.loads(
            embedding_json
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            "Stored embedding contains invalid JSON."
        ) from error

    if not isinstance(embedding, list):
        raise ValueError(
            "Stored embedding must be a JSON list."
        )

    return _validate_embedding(
        embedding
    )


def insert_document(
    content: str,
    source: str,
    embedding: Sequence[float],
    database_path: Path = DATABASE_PATH,
) -> int:
    """
    Insert a document chunk and return its database ID.

    If the same source and content already exist, the existing
    record ID is returned instead of inserting a duplicate.
    """

    clean_content = content.strip()
    clean_source = source.strip()

    if not clean_content:
        raise ValueError(
            "Document content cannot be empty."
        )

    if not clean_source:
        raise ValueError(
            "Document source cannot be empty."
        )

    embedding_json = _serialize_embedding(
        embedding
    )

    initialize_database(
        database_path
    )

    with database_connection(database_path) as connection:
        existing_row = connection.execute(
            """
            SELECT id
            FROM documents
            WHERE source = ? AND content = ?
            """,
            (
                clean_source,
                clean_content,
            ),
        ).fetchone()

        if existing_row is not None:
            return int(
                existing_row["id"]
            )

        cursor = connection.execute(
            """
            INSERT INTO documents (
                content,
                source,
                embedding
            )
            VALUES (?, ?, ?)
            """,
            (
                clean_content,
                clean_source,
                embedding_json,
            ),
        )

        if cursor.lastrowid is None:
            raise RuntimeError(
                "SQLite did not return an inserted document ID."
            )

        return int(
            cursor.lastrowid
        )


def get_all_documents(
    database_path: Path = DATABASE_PATH,
) -> list[DocumentRecord]:
    """Return all stored document chunks ordered by ID."""

    initialize_database(
        database_path
    )

    with database_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                content,
                source,
                embedding,
                created_at
            FROM documents
            ORDER BY id
            """
        ).fetchall()

    documents: list[DocumentRecord] = []

    for row in rows:
        documents.append(
            {
                "id": int(row["id"]),
                "content": str(row["content"]),
                "source": str(row["source"]),
                "embedding": _deserialize_embedding(
                    str(row["embedding"])
                ),
                "created_at": str(
                    row["created_at"]
                ),
            }
        )

    return documents


def get_document_count(
    database_path: Path = DATABASE_PATH,
) -> int:
    """Return the number of document chunks in the database."""

    initialize_database(
        database_path
    )

    with database_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS document_count
            FROM documents
            """
        ).fetchone()

    if row is None:
        return 0

    return int(
        row["document_count"]
    )


def delete_all_documents(
    database_path: Path = DATABASE_PATH,
) -> None:
    """Delete all document chunks and reset the ID sequence."""

    initialize_database(
        database_path
    )

    with database_connection(database_path) as connection:
        connection.execute(
            "DELETE FROM documents"
        )

        connection.execute(
            """
            DELETE FROM sqlite_sequence
            WHERE name = 'documents'
            """
        )