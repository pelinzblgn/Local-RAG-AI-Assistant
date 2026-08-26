import tempfile
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

from src.database import (
    delete_all_documents,
    delete_documents_by_source,
    delete_source_file,
    get_all_documents,
    get_all_source_files,
    get_document_count,
    get_source_file,
    initialize_database,
    insert_document,
    upsert_source_file,
)

@contextmanager
def temporary_database() -> Iterator[Path]:
    """
    Create a temporary SQLite database path for one test.

    The directory remains available throughout the context and
    is removed automatically when the test finishes.
    """

    with tempfile.TemporaryDirectory() as temp_directory:
        database_path = Path(temp_directory) / "test.db"
        yield database_path


def test_database_initializes() -> None:
    with temporary_database() as database_path:
        initialize_database(database_path)

        assert database_path.exists()


def test_insert_document_returns_first_id() -> None:
    with temporary_database() as database_path:
        document_id = insert_document(
            content="STM32 Timer",
            source="stm32.txt",
            embedding=[0.1, 0.2, 0.3],
            database_path=database_path,
        )

        assert document_id == 1


def test_duplicate_document_returns_same_id() -> None:
    with temporary_database() as database_path:
        first_id = insert_document(
            content="PID",
            source="pid.txt",
            embedding=[0.5, 0.2],
            database_path=database_path,
        )

        second_id = insert_document(
            content="PID",
            source="pid.txt",
            embedding=[0.5, 0.2],
            database_path=database_path,
        )

        assert first_id == second_id
        assert get_document_count(database_path) == 1


def test_document_count() -> None:
    with temporary_database() as database_path:
        insert_document(
            content="A",
            source="a.txt",
            embedding=[1.0],
            database_path=database_path,
        )

        insert_document(
            content="B",
            source="b.txt",
            embedding=[2.0],
            database_path=database_path,
        )

        assert get_document_count(database_path) == 2


def test_delete_documents() -> None:
    with temporary_database() as database_path:
        insert_document(
            content="STM32",
            source="stm32.txt",
            embedding=[0.1],
            database_path=database_path,
        )

        delete_all_documents(database_path)

        assert get_document_count(database_path) == 0


def test_delete_resets_id_sequence() -> None:
    with temporary_database() as database_path:
        insert_document(
            content="İlk belge",
            source="first.txt",
            embedding=[0.1],
            database_path=database_path,
        )

        delete_all_documents(database_path)

        new_id = insert_document(
            content="Yeni belge",
            source="new.txt",
            embedding=[0.2],
            database_path=database_path,
        )

        assert new_id == 1


def test_get_all_documents_returns_complete_record() -> None:
    with temporary_database() as database_path:
        insert_document(
            content="PWM controls motor speed",
            source="motor.txt",
            embedding=[0.4, 0.5],
            database_path=database_path,
        )

        documents = get_all_documents(database_path)

        assert len(documents) == 1

        document = documents[0]

        assert document["id"] == 1
        assert document["source"] == "motor.txt"
        assert document["content"] == "PWM controls motor speed"
        assert document["embedding"] == [0.4, 0.5]
        assert document["created_at"]


def test_empty_content_raises_error() -> None:
    with temporary_database() as database_path:
        try:
            insert_document(
                content="   ",
                source="test.txt",
                embedding=[0.1],
                database_path=database_path,
            )
        except ValueError as error:
            assert "content cannot be empty" in str(error)
            return

    raise AssertionError(
        "Expected ValueError for empty document content."
    )


def test_empty_source_raises_error() -> None:
    with temporary_database() as database_path:
        try:
            insert_document(
                content="Test content",
                source="   ",
                embedding=[0.1],
                database_path=database_path,
            )
        except ValueError as error:
            assert "source cannot be empty" in str(error)
            return

    raise AssertionError(
        "Expected ValueError for empty document source."
    )


def test_invalid_embedding_value_raises_error() -> None:
    with temporary_database() as database_path:
        try:
            insert_document(
                content="Test content",
                source="test.txt",
                embedding=[0.1, float("nan")],
                database_path=database_path,
            )
        except ValueError as error:
            assert "NaN or infinite" in str(error)
            return

    raise AssertionError(
        "Expected ValueError for invalid embedding value."
    )


def run_tests() -> None:
    tests = [
        test_database_initializes,
        test_insert_document_returns_first_id,
        test_duplicate_document_returns_same_id,
        test_document_count,
        test_delete_documents,
        test_delete_resets_id_sequence,
        test_get_all_documents_returns_complete_record,
        test_empty_content_raises_error,
        test_empty_source_raises_error,
        test_invalid_embedding_value_raises_error,
    ]

    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print(f"PASS: {test.__name__}")
        except Exception as error:
            print(f"FAIL: {test.__name__}")
            print(f"      {error}")

    print("-" * 50)
    print(f"Sonuç: {passed}/{len(tests)} test başarılı.")


if __name__ == "__main__":
    run_tests()
    
    
from pathlib import Path


def test_source_file_can_be_inserted(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    upsert_source_file(
        source="stm32.txt",
        file_hash="abc123",
        modified_at=123.0,
        database_path=database_path,
    )

    record = get_source_file(
        "stm32.txt",
        database_path,
    )

    assert record is not None
    assert record["source"] == "stm32.txt"
    assert record["file_hash"] == "abc123"
    assert record["modified_at"] == 123.0


def test_source_file_can_be_updated(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    upsert_source_file(
        "stm32.txt",
        "old_hash",
        100.0,
        database_path,
    )

    upsert_source_file(
        "stm32.txt",
        "new_hash",
        200.0,
        database_path,
    )

    record = get_source_file(
        "stm32.txt",
        database_path,
    )

    assert record is not None
    assert record["file_hash"] == "new_hash"
    assert record["modified_at"] == 200.0


def test_get_all_source_files(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    upsert_source_file(
        "b.txt",
        "hash-b",
        2.0,
        database_path,
    )

    upsert_source_file(
        "a.txt",
        "hash-a",
        1.0,
        database_path,
    )

    records = get_all_source_files(
        database_path
    )

    assert len(records) == 2
    assert records[0]["source"] == "a.txt"
    assert records[1]["source"] == "b.txt"


def test_delete_documents_by_source(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    insert_document(
        "STM32 chunk 1",
        "stm32.txt",
        [0.1, 0.2],
        database_path,
    )

    insert_document(
        "STM32 chunk 2",
        "stm32.txt",
        [0.2, 0.3],
        database_path,
    )

    insert_document(
        "PID chunk",
        "pid.txt",
        [0.3, 0.4],
        database_path,
    )

    deleted_count = delete_documents_by_source(
        "stm32.txt",
        database_path,
    )

    documents = get_all_documents(
        database_path
    )

    assert deleted_count == 2
    assert len(documents) == 1
    assert documents[0]["source"] == "pid.txt"


def test_delete_source_file(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    upsert_source_file(
        "stm32.txt",
        "abc123",
        10.0,
        database_path,
    )

    deleted = delete_source_file(
        "stm32.txt",
        database_path,
    )

    assert deleted is True

    assert get_source_file(
        "stm32.txt",
        database_path,
    ) is None


def test_delete_all_documents_also_clears_source_metadata(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    insert_document(
        "STM32",
        "stm32.txt",
        [0.1],
        database_path,
    )

    upsert_source_file(
        "stm32.txt",
        "abc123",
        10.0,
        database_path,
    )

    delete_all_documents(
        database_path
    )

    assert get_document_count(
        database_path
    ) == 0

    assert get_all_source_files(
        database_path
    ) == []