import tempfile
from pathlib import Path
from unittest.mock import patch

from src.chunker import split_into_chunks
from src.document_loader import read_text_file
from src.ingestion import ingest_text_files

def test_read_text_file_returns_clean_content() -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
        file_path = Path(temp_directory) / "notes.txt"
        file_path.write_text(
            "  STM32 PWM motor hızını kontrol eder.  ",
            encoding="utf-8",
        )

        content = read_text_file(file_path)

        assert content == "STM32 PWM motor hızını kontrol eder."


def test_read_text_file_rejects_missing_file() -> None:
    missing_file = Path("missing_file.txt")

    try:
        read_text_file(missing_file)
    except FileNotFoundError as error:
        assert "File does not exist" in str(error)
        return

    raise AssertionError(
        "Expected FileNotFoundError for missing file."
    )


def test_read_text_file_rejects_non_txt_file() -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
        file_path = Path(temp_directory) / "notes.md"
        file_path.write_text(
            "Markdown content",
            encoding="utf-8",
        )

        try:
            read_text_file(file_path)
        except ValueError as error:
            assert "Unsupported file type" in str(error)
            return

    raise AssertionError(
        "Expected ValueError for unsupported file type."
    )


def test_read_text_file_rejects_empty_file() -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
        file_path = Path(temp_directory) / "empty.txt"
        file_path.write_text(
            "   ",
            encoding="utf-8",
        )

        try:
            read_text_file(file_path)
        except ValueError as error:
            assert "Text file is empty" in str(error)
            return

    raise AssertionError(
        "Expected ValueError for empty text file."
    )


def test_split_into_chunks_uses_configured_limits() -> None:
    text = " ".join(
        f"kelime{i}"
        for i in range(200)
    )

    chunks = split_into_chunks(
        text=text,
        max_characters=120,
        overlap_characters=20,
    )

    assert len(chunks) > 1
    assert all(
        0 < len(chunk) <= 120
        for chunk in chunks
    )


def test_ingestion_stores_generated_chunks() -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
        raw_directory = Path(temp_directory)

        first_file = raw_directory / "pid.txt"
        second_file = raw_directory / "stm32.txt"

        first_file.write_text(
            "PID kontrolcü P, I ve D bileşenlerinden oluşur.",
            encoding="utf-8",
        )

        second_file.write_text(
            "PWM duty cycle motor hızını kontrol eder.",
            encoding="utf-8",
        )

        fake_embeddings = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

        inserted_documents: list[dict[str, object]] = []

        def fake_insert_document(
            content: str,
            source: str,
            embedding: list[float],
        ) -> int:
            inserted_documents.append(
                {
                    "content": content,
                    "source": source,
                    "embedding": embedding,
                }
            )

            return len(inserted_documents)

        with (
            patch(
                "src.ingestion.initialize_database"
            ),
            patch(
                "src.ingestion.delete_all_documents"
            ),
            patch(
                "src.ingestion.get_document_count",
                side_effect=[0, 2],
            ),
            patch(
                "src.ingestion.generate_embeddings",
                return_value=fake_embeddings,
            ) as mocked_embeddings,
            patch(
                "src.ingestion.insert_document",
                side_effect=fake_insert_document,
            ),
        ):
            inserted_count = ingest_text_files(
                reset_database=True,
                raw_data_directory=raw_directory,
            )

        assert inserted_count == 2
        assert len(inserted_documents) == 2

        assert inserted_documents[0]["source"] == "pid.txt"
        assert inserted_documents[1]["source"] == "stm32.txt"

        mocked_embeddings.assert_called_once()

        generated_contents = mocked_embeddings.call_args.args[0]

        assert len(generated_contents) == 2


def test_ingestion_rejects_missing_directory() -> None:
    missing_directory = Path(
        "directory_that_does_not_exist"
    )

    with patch(
        "src.ingestion.initialize_database"
    ):
        try:
            ingest_text_files(
                raw_data_directory=missing_directory,
            )
        except FileNotFoundError as error:
            assert "Data directory does not exist" in str(error)
            return

    raise AssertionError(
        "Expected FileNotFoundError for missing directory."
    )


def test_ingestion_rejects_directory_without_txt_files() -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
        raw_directory = Path(temp_directory)

        markdown_file = raw_directory / "notes.md"
        markdown_file.write_text(
            "Markdown content",
            encoding="utf-8",
        )

        with patch(
            "src.ingestion.initialize_database"
        ):
            try:
                ingest_text_files(
                    raw_data_directory=raw_directory,
                )
            except RuntimeError as error:
                assert "No TXT files were found" in str(error)
                return

    raise AssertionError(
        "Expected RuntimeError when no TXT files exist."
    )


def run_tests() -> None:
    tests = [
        test_read_text_file_returns_clean_content,
        test_read_text_file_rejects_missing_file,
        test_read_text_file_rejects_non_txt_file,
        test_read_text_file_rejects_empty_file,
        test_split_into_chunks_uses_configured_limits,
        test_ingestion_stores_generated_chunks,
        test_ingestion_rejects_missing_directory,
        test_ingestion_rejects_directory_without_txt_files,
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
    print(
        f"Sonuç: {passed}/{len(tests)} "
        "test başarılı."
    )


if __name__ == "__main__":
    run_tests()