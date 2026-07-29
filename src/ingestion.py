from pathlib import Path

from src.database import (
    delete_all_documents,
    initialize_database,
    insert_document,
)
from src.embeddings import generate_embeddings


RAW_DATA_DIRECTORY = Path("data") / "raw"


def read_text_file(file_path: Path) -> str:
    """Read a UTF-8 text file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    if file_path.suffix.lower() != ".txt":
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}"
        )

    return file_path.read_text(encoding="utf-8").strip()


def split_into_chunks(
    text: str,
    max_characters: int = 500,
) -> list[str]:
    """Split text into paragraph-based chunks."""

    clean_text = text.strip()

    if not clean_text:
        return []

    if max_characters <= 0:
        raise ValueError(
            "max_characters must be greater than zero."
        )

    paragraphs = [
        paragraph.strip()
        for paragraph in clean_text.split("\n\n")
        if paragraph.strip()
    ]

    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        candidate = (
            f"{current_chunk}\n\n{paragraph}".strip()
        )

        if len(candidate) <= max_characters:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)

        current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def ingest_text_files(
    reset_database: bool = False,
) -> int:
    """Read TXT files, create chunks and store them in SQLite."""

    initialize_database()

    if reset_database:
        delete_all_documents()

    if not RAW_DATA_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: "
            f"{RAW_DATA_DIRECTORY}"
        )

    text_files = sorted(
        RAW_DATA_DIRECTORY.glob("*.txt")
    )

    if not text_files:
        raise RuntimeError(
            "No TXT files were found in data/raw."
        )

    chunk_records: list[dict[str, str]] = []

    for file_path in text_files:
        print(f"Belge okunuyor: {file_path.name}")

        text = read_text_file(file_path)
        chunks = split_into_chunks(text)

        for chunk in chunks:
            chunk_records.append(
                {
                    "content": chunk,
                    "source": file_path.name,
                }
            )

    if not chunk_records:
        raise RuntimeError(
            "No valid text chunks were generated."
        )

    contents = [
        record["content"]
        for record in chunk_records
    ]

    print(
        f"Toplam {len(contents)} chunk için "
        "embedding oluşturuluyor..."
    )

    embeddings = generate_embeddings(contents)

    for record, embedding in zip(
        chunk_records,
        embeddings,
    ):
        insert_document(
            content=record["content"],
            source=record["source"],
            embedding=embedding,
        )

    return len(chunk_records)