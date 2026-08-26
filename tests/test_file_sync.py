from pathlib import Path
from unittest.mock import patch

import pytest

from src.database import (
    get_all_documents,
    get_all_source_files,
)
from src.file_sync import (
    SyncResult,
    calculate_file_hash,
    synchronize_knowledge_base,
)


def test_same_file_produces_same_hash(
    tmp_path: Path,
) -> None:
    file_path = (
        tmp_path / "test.txt"
    )

    file_path.write_text(
        "STM32 test content",
        encoding="utf-8",
    )

    first_hash = calculate_file_hash(
        file_path
    )

    second_hash = calculate_file_hash(
        file_path
    )

    assert first_hash == second_hash


def test_changed_file_produces_different_hash(
    tmp_path: Path,
) -> None:
    file_path = (
        tmp_path / "test.txt"
    )

    file_path.write_text(
        "Version 1",
        encoding="utf-8",
    )

    first_hash = calculate_file_hash(
        file_path
    )

    file_path.write_text(
        "Version 2",
        encoding="utf-8",
    )

    second_hash = calculate_file_hash(
        file_path
    )

    assert first_hash != second_hash


def test_missing_file_raises_error(
    tmp_path: Path,
) -> None:
    file_path = (
        tmp_path / "missing.txt"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        calculate_file_hash(
            file_path
        )


def test_directory_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError
    ):
        calculate_file_hash(
            tmp_path
        )


def test_new_file_is_indexed(
    tmp_path: Path,
) -> None:
    raw_directory = (
        tmp_path / "raw"
    )

    raw_directory.mkdir()

    (
        raw_directory / "stm32.txt"
    ).write_text(
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
    ) as mocked_embeddings:
        result = (
            synchronize_knowledge_base(
                raw_data_directory=raw_directory,
                database_path=database_path,
            )
        )

    assert result.new_files == (
        "stm32.txt",
    )

    assert (
        result.modified_files
        == ()
    )

    assert (
        result.deleted_files
        == ()
    )

    assert (
        result.unchanged_files
        == ()
    )

    assert result.inserted_chunks == 1
    assert result.deleted_chunks == 0
    assert result.has_changes is True

    mocked_embeddings.assert_called_once()

    documents = get_all_documents(
        database_path
    )

    assert len(documents) == 1

    assert (
        documents[0]["source"]
        == "stm32.txt"
    )

    manifests = get_all_source_files(
        database_path
    )

    assert len(manifests) == 1

    assert (
        manifests[0]["source"]
        == "stm32.txt"
    )


def test_unchanged_file_is_not_embedded_again(
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
        synchronize_knowledge_base(
            raw_data_directory=raw_directory,
            database_path=database_path,
        )

    with patch(
        "src.ingestion.generate_embeddings"
    ) as mocked_embeddings:
        result = (
            synchronize_knowledge_base(
                raw_data_directory=raw_directory,
                database_path=database_path,
            )
        )

    assert result.new_files == ()
    assert result.modified_files == ()
    assert result.deleted_files == ()

    assert result.unchanged_files == (
        "stm32.txt",
    )

    assert result.inserted_chunks == 0
    assert result.deleted_chunks == 0
    assert result.has_changes is False

    mocked_embeddings.assert_not_called()


def test_modified_file_is_reindexed(
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
        "STM32 version one",
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
        synchronize_knowledge_base(
            raw_data_directory=raw_directory,
            database_path=database_path,
        )

    original_documents = (
        get_all_documents(
            database_path
        )
    )

    assert len(
        original_documents
    ) == 1

    assert (
        original_documents[0]["content"]
        == "STM32 version one"
    )

    file_path.write_text(
        "STM32 version two with PWM",
        encoding="utf-8",
    )

    with patch(
        "src.ingestion.generate_embeddings",
        return_value=[
            [0.8, 0.9],
        ],
    ) as mocked_embeddings:
        result = (
            synchronize_knowledge_base(
                raw_data_directory=raw_directory,
                database_path=database_path,
            )
        )

    assert result.new_files == ()

    assert result.modified_files == (
        "stm32.txt",
    )

    assert result.deleted_files == ()
    assert result.unchanged_files == ()

    assert result.deleted_chunks == 1
    assert result.inserted_chunks == 1

    mocked_embeddings.assert_called_once()

    documents = get_all_documents(
        database_path
    )

    assert len(documents) == 1

    assert (
        documents[0]["content"]
        == "STM32 version two with PWM"
    )

    assert (
        documents[0]["embedding"]
        == [0.8, 0.9]
    )


def test_deleted_file_is_removed_from_database(
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
        "STM32 test document",
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
        synchronize_knowledge_base(
            raw_data_directory=raw_directory,
            database_path=database_path,
        )

    file_path.unlink()

    result = synchronize_knowledge_base(
        raw_data_directory=raw_directory,
        database_path=database_path,
    )

    assert result.new_files == ()
    assert result.modified_files == ()

    assert result.deleted_files == (
        "stm32.txt",
    )

    assert result.unchanged_files == ()

    assert result.inserted_chunks == 0
    assert result.deleted_chunks == 1

    assert get_all_documents(
        database_path
    ) == []

    assert get_all_source_files(
        database_path
    ) == []


def test_sync_handles_multiple_file_states(
    tmp_path: Path,
) -> None:
    raw_directory = (
        tmp_path / "raw"
    )

    raw_directory.mkdir()

    unchanged_file = (
        raw_directory
        / "unchanged.txt"
    )

    modified_file = (
        raw_directory
        / "modified.txt"
    )

    deleted_file = (
        raw_directory
        / "deleted.txt"
    )

    unchanged_file.write_text(
        "Unchanged content",
        encoding="utf-8",
    )

    modified_file.write_text(
        "Old modified content",
        encoding="utf-8",
    )

    deleted_file.write_text(
        "Deleted content",
        encoding="utf-8",
    )

    database_path = (
        tmp_path / "test.db"
    )

    def initial_fake_embeddings(
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [0.1, 0.2]
            for _ in texts
        ]

    with patch(
        "src.ingestion.generate_embeddings",
        side_effect=initial_fake_embeddings,
    ):
        synchronize_knowledge_base(
            raw_data_directory=raw_directory,
            database_path=database_path,
        )

    modified_file.write_text(
        "New modified content",
        encoding="utf-8",
    )

    deleted_file.unlink()

    new_file = (
        raw_directory
        / "new.txt"
    )

    new_file.write_text(
        "Brand new content",
        encoding="utf-8",
    )

    def updated_fake_embeddings(
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [0.9, 0.8]
            for _ in texts
        ]

    with patch(
        "src.ingestion.generate_embeddings",
        side_effect=updated_fake_embeddings,
    ):
        result = (
            synchronize_knowledge_base(
                raw_data_directory=raw_directory,
                database_path=database_path,
            )
        )

    assert result.new_files == (
        "new.txt",
    )

    assert result.modified_files == (
        "modified.txt",
    )

    assert result.deleted_files == (
        "deleted.txt",
    )

    assert result.unchanged_files == (
        "unchanged.txt",
    )

    assert result.has_changes is True


def test_sync_result_reports_change_count() -> None:
    result = SyncResult(
        new_files=(
            "new.txt",
        ),
        modified_files=(
            "modified.txt",
        ),
        deleted_files=(
            "deleted.txt",
        ),
        unchanged_files=(
            "unchanged.txt",
        ),
        inserted_chunks=2,
        deleted_chunks=2,
    )

    assert (
        result.changed_file_count
        == 3
    )

    assert result.has_changes is True