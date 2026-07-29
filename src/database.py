import json
import sqlite3
from pathlib import Path


DATABASE_PATH = Path("database") / "rag.db"


def get_connection() -> sqlite3.Connection:
    """Create and return a SQLite database connection."""

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database() -> None:
    """Create the documents table if it does not already exist."""

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def insert_document(
    content: str,
    source: str,
    embedding: list[float],
) -> int:
    """Insert a document chunk and return its database ID."""

    clean_content = content.strip()
    clean_source = source.strip()

    if not clean_content:
        raise ValueError("Document content cannot be empty.")

    if not clean_source:
        raise ValueError("Document source cannot be empty.")

    if not embedding:
        raise ValueError("Embedding cannot be empty.")

    embedding_json = json.dumps(embedding)

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents (content, source, embedding)
            VALUES (?, ?, ?)
            """,
            (
                clean_content,
                clean_source,
                embedding_json,
            ),
        )

        return int(cursor.lastrowid)


def get_all_documents() -> list[dict]:
    """Return all stored document chunks and their embeddings."""

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, content, source, embedding, created_at
            FROM documents
            ORDER BY id
            """
        ).fetchall()

    documents = []

    for row in rows:
        documents.append(
            {
                "id": row["id"],
                "content": row["content"],
                "source": row["source"],
                "embedding": json.loads(row["embedding"]),
                "created_at": row["created_at"],
            }
        )

    return documents


def delete_all_documents() -> None:
    """Delete every document from the local database."""

    with get_connection() as connection:
        connection.execute("DELETE FROM documents")