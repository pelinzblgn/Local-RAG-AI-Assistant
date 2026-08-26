from pathlib import Path
from unittest.mock import patch

import pytest

from src.database import (
    get_all_documents,
    get_all_source_files,
)
from src.ingestion import (
    ingest_text_files,
)
from src.document_loader import (
    read_text_file,
)
from src.chunker import (
    split_into_chunks,
)


def test_read_text_file_returns_clean_content(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "  STM32 test content  ",
        encoding="utf-8",
    )

    result = read_text_file(
        file_path
    )

    assert result == "STM32 test content"


def test_read_text_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    file_path = (
        tmp_path / "missing.txt"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        read_text_file(
            file_path
        )


def test_read_text_file_rejects_non_txt_file(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "test.md"

    file_path.write_text(
        "Markdown content",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError
    ):
        read_text_file(
            file_path
        )


def test_read_text_file_rejects_empty_file(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "empty.txt"

    file_path.write_text(
        "   ",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError
    ):
        read_text_file(
            file_path
        )


def test_split_into_chunks_uses_configured_limits() -> None:
    text = " ".join(
        f"kelime{i}"
        for i in range(200)
    )

    chunks = split_into_chunks(
        text=text,
        max_characters=100,
        overlap_characters=20,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.strip()
        assert len(chunk) <= 100


def test_ingestion_stores_generated_chunks(
    tmp_path: Path,
) -> None:
    raw_directory = (
        tmp_path / "raw"
    )

    raw_directory.mkdir()

    file_path = (
        raw_directory / "stm32.txt"
    )

    file_path.write_text(
        "STM32 PWM UART GPIO",
        encoding="utf-8",
    )

    database_path = (
        tmp_path / "test.db"
    )

    with patch(
        "src.ingestion.generate_embeddings",
        return_value=[
            [0.1, 0.2, 0.3],
        ],
    ):
        inserted_count = ingest_text_files(
            reset_database=True,
            raw_data_directory=raw_directory,
            database_path=database_path,
        )

    documents = get_all_documents(
        database_path
    )

    assert inserted_count == 1
    assert len(documents) == 1

    assert (
        documents[0]["source"]
        == "stm32.txt"
    )

    assert (
        documents[0]["content"]
        == "STM32 PWM UART GPIO"
    )

    assert documents[0]["embedding"] == [
        0.1,
        0.2,
        0.3,
    ]


def test_ingestion_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    raw_directory = (
        tmp_path / "missing"
    )

    database_path = (
        tmp_path / "test.db"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        ingest_text_files(
            raw_data_directory=raw_directory,
            database_path=database_path,
        )


def test_ingestion_rejects_directory_without_txt_files(
    tmp_path: Path,
) -> None:
    raw_directory = (
        tmp_path / "raw"
    )

    raw_directory.mkdir()

    (
        raw_directory / "notes.md"
    ).write_text(
        "Markdown",
        encoding="utf-8",
    )

    database_path = (
        tmp_path / "test.db"
    )

    with pytest.raises(
        RuntimeError
    ):
        ingest_text_files(
            raw_data_directory=raw_directory,
            database_path=database_path,
        )


def test_ingestion_creates_source_manifest(
    tmp_path: Path,
) -> None:
    raw_directory = (
        tmp_path / "raw"
    )

    raw_directory.mkdir()

    file_path = (
        raw_directory / "stm32.txt"
    )

    file_path.write_text(
        "STM32 PWM UART GPIO",
        encoding="utf-8",
    )

    database_path = (
        tmp_path / "test.db"
    )

    with patch(
        "src.ingestion.generate_embeddings",
        return_value=[
            [0.1, 0.2, 0.3],
        ],
    ):
        ingest_text_files(
            reset_database=True,
            raw_data_directory=raw_directory,
            database_path=database_path,
        )

    records = get_all_source_files(
        database_path
    )

    assert len(records) == 1

    assert (
        records[0]["source"]
        == "stm32.txt"
    )

    assert records[0]["file_hash"]

    assert isinstance(
        records[0]["modified_at"],
        float,
    )


def test_manifest_hash_changes_when_file_changes(
    tmp_path: Path,
) -> None:
    raw_directory = (
        tmp_path / "raw"
    )

    raw_directory.mkdir()

    file_path = (
        raw_directory / "stm32.txt"
    )

    file_path.write_text(
        "Version 1",
        encoding="utf-8",
    )

    database_path = (
        tmp_path / "test.db"
    )

    with patch(
        "src.ingestion.generate_embeddings",
        return_value=[
            [0.1, 0.2],
        ],
    ):
        ingest_text_files(
            reset_database=True,
            raw_data_directory=raw_directory,
            database_path=database_path,
        )

    first_record = (
        get_all_source_files(
            database_path
        )[0]
    )

    file_path.write_text(
        "Version 2",
        encoding="utf-8",
    )

    with patch(
        "src.ingestion.generate_embeddings",
        return_value=[
            [0.3, 0.4],
        ],
    ):
        ingest_text_files(
            reset_database=True,
            raw_data_directory=raw_directory,
            database_path=database_path,
        )

    second_record = (
        get_all_source_files(
            database_path
        )[0]
    )

    assert (
        first_record["file_hash"]
        != second_record["file_hash"]
    )