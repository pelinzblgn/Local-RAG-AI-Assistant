from pathlib import Path


SUPPORTED_TEXT_EXTENSIONS = {".txt"}

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
}


def _validate_path_object(
    path: Path,
    *,
    argument_name: str,
) -> None:
    """
    Validate that a path argument is a pathlib.Path instance.
    """

    if not isinstance(path, Path):
        raise TypeError(
            f"{argument_name} must be a pathlib.Path object."
        )


def is_supported_document_file(
    file_path: Path,
) -> bool:
    """
    Return whether a path has a supported document extension.

    Supported formats:
        - .txt
        - .pdf
        - .docx

    This function checks only the extension and does not require
    the file to exist.

    Args:
        file_path: Path to inspect.

    Returns:
        True if the extension is supported, otherwise False.

    Raises:
        TypeError: If file_path is not a pathlib.Path.
    """

    _validate_path_object(
        file_path,
        argument_name="file_path",
    )

    return (
        file_path.suffix.lower()
        in SUPPORTED_DOCUMENT_EXTENSIONS
    )


def is_supported_text_file(
    file_path: Path,
) -> bool:
    """
    Return whether a path is a supported plain-text file.

    This function is retained for backward compatibility with
    the existing TXT ingestion pipeline.

    Args:
        file_path: Path to inspect.

    Returns:
        True only for supported plain-text extensions.

    Raises:
        TypeError: If file_path is not a pathlib.Path.
    """

    _validate_path_object(
        file_path,
        argument_name="file_path",
    )

    return (
        file_path.suffix.lower()
        in SUPPORTED_TEXT_EXTENSIONS
    )


def _validate_existing_file(
    file_path: Path,
) -> None:
    """
    Validate that a path exists and points to a supported file.
    """

    _validate_path_object(
        file_path,
        argument_name="file_path",
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    if not is_supported_document_file(file_path):
        raise ValueError(
            f"Unsupported file type: "
            f"{file_path.suffix or '<no extension>'}"
        )


def _read_txt_file(
    file_path: Path,
) -> str:
    """
    Extract text from a UTF-8 TXT file.
    """

    try:
        content = file_path.read_text(
            encoding="utf-8"
        ).strip()
    except UnicodeDecodeError as error:
        raise RuntimeError(
            f"File is not valid UTF-8: {file_path}"
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"Unable to read file: {file_path}"
        ) from error

    if not content:
        raise ValueError(
            f"Text file is empty: {file_path.name}"
        )

    return content


def _read_pdf_file(
    file_path: Path,
) -> str:
    """
    Extract text from a text-based PDF document.

    Scanned/image-only PDFs are intentionally not OCR-processed.
    If no extractable text exists, the document is rejected.
    """

    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "PDF support requires the 'pypdf' package."
        ) from error

    try:
        reader = PdfReader(str(file_path))

        page_texts: list[str] = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                cleaned_text = text.strip()

                if cleaned_text:
                    page_texts.append(cleaned_text)

        content = "\n\n".join(page_texts).strip()

    except Exception as error:
        raise RuntimeError(
            f"Unable to read PDF file: {file_path}"
        ) from error

    if not content:
        raise ValueError(
            "PDF contains no extractable text. "
            "Scanned or image-only PDFs are not supported: "
            f"{file_path.name}"
        )

    return content


def _read_docx_file(
    file_path: Path,
) -> str:
    """
    Extract paragraph text from a DOCX document.
    """

    try:
        from docx import Document
    except ImportError as error:
        raise RuntimeError(
            "DOCX support requires the 'python-docx' package."
        ) from error

    try:
        document = Document(str(file_path))

        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        content = "\n\n".join(paragraphs).strip()

    except Exception as error:
        raise RuntimeError(
            f"Unable to read DOCX file: {file_path}"
        ) from error

    if not content:
        raise ValueError(
            f"DOCX file contains no readable text: "
            f"{file_path.name}"
        )

    return content


def read_document_file(
    file_path: Path,
) -> str:
    """
    Read a supported document and return normalized plain text.

    Supported formats:
        - UTF-8 TXT
        - text-based PDF
        - DOCX

    The function only extracts text. Chunking, embedding,
    persistence, and retrieval remain responsibilities of the
    existing RAG pipeline.

    Args:
        file_path: Path of the document.

    Returns:
        Extracted and whitespace-trimmed text.

    Raises:
        TypeError: If file_path is not a pathlib.Path.
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is invalid, unsupported, or
            contains no usable text.
        RuntimeError: If the document cannot be decoded,
            parsed, or its required parser is unavailable.
    """

    _validate_existing_file(file_path)

    extension = file_path.suffix.lower()

    if extension == ".txt":
        return _read_txt_file(file_path)

    if extension == ".pdf":
        return _read_pdf_file(file_path)

    if extension == ".docx":
        return _read_docx_file(file_path)

    raise ValueError(
        f"Unsupported file type: "
        f"{file_path.suffix or '<no extension>'}"
    )


def read_text_file(
    file_path: Path,
) -> str:
    """
    Read and return the contents of a UTF-8 TXT file.

    This function is retained for backward compatibility with
    the existing ingestion pipeline.

    Args:
        file_path: Path of the TXT file.

    Returns:
        Whitespace-trimmed file content.

    Raises:
        TypeError: If file_path is not a pathlib.Path.
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a TXT file or is empty.
        RuntimeError: If the file cannot be decoded as UTF-8.
    """

    _validate_path_object(
        file_path,
        argument_name="file_path",
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    if not is_supported_text_file(file_path):
        raise ValueError(
            f"Unsupported text file type: "
            f"{file_path.suffix or '<no extension>'}"
        )

    return _read_txt_file(file_path)


def find_document_files(
    directory: Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    """
    Find supported documents inside a directory.

    Supported formats:
        - .txt
        - .pdf
        - .docx

    By default, only files directly inside the selected
    directory are returned. Recursive scanning must be
    explicitly enabled.

    Args:
        directory: Directory to scan.
        recursive: Whether subdirectories should be scanned.

    Returns:
        Alphabetically ordered supported document paths.

    Raises:
        TypeError: If directory is not a pathlib.Path or
            recursive is not a bool.
        FileNotFoundError: If the directory does not exist.
        ValueError: If the path is not a directory.
    """

    _validate_path_object(
        directory,
        argument_name="directory",
    )

    if not isinstance(recursive, bool):
        raise TypeError(
            "recursive must be a bool."
        )

    if not directory.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Data path is not a directory: {directory}"
        )

    if recursive:
        candidates = directory.rglob("*")
    else:
        candidates = directory.iterdir()

    files = [
        path
        for path in candidates
        if path.is_file()
        and is_supported_document_file(path)
    ]

    return sorted(
        files,
        key=lambda path: str(path).lower(),
    )


def find_text_files(
    directory: Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    """
    Find supported TXT files inside a directory.

    This function is retained for backward compatibility with
    the existing managed TXT knowledge-base workflow.

    Args:
        directory: Directory to scan.
        recursive: Whether subdirectories should be scanned.

    Returns:
        Alphabetically ordered TXT file paths.

    Raises:
        TypeError: If directory is not a pathlib.Path or
            recursive is not a bool.
        FileNotFoundError: If the directory does not exist.
        ValueError: If the path is not a directory.
    """

    _validate_path_object(
        directory,
        argument_name="directory",
    )

    if not isinstance(recursive, bool):
        raise TypeError(
            "recursive must be a bool."
        )

    if not directory.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Data path is not a directory: {directory}"
        )

    if recursive:
        candidates = directory.rglob("*")
    else:
        candidates = directory.iterdir()

    files = [
        path
        for path in candidates
        if path.is_file()
        and is_supported_text_file(path)
    ]

    return sorted(
        files,
        key=lambda path: str(path).lower(),
    )


def collect_document_files(
    source_path: Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    """
    Resolve a user-selected file or directory into supported
    document files.

    Supported formats:
        - .txt
        - .pdf
        - .docx

    If source_path points to a file, that one supported document
    is returned. If it points to a directory, supported documents
    inside that directory are returned.

    Args:
        source_path: User-selected file or directory.
        recursive: Whether directory scanning should include
            subdirectories.

    Returns:
        List of supported document paths.

    Raises:
        TypeError: If source_path is not a pathlib.Path.
        FileNotFoundError: If source_path does not exist.
        ValueError: If the selected path or file type is
            unsupported.
    """

    _validate_path_object(
        source_path,
        argument_name="source_path",
    )

    if not isinstance(recursive, bool):
        raise TypeError(
            "recursive must be a bool."
        )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Selected path does not exist: {source_path}"
        )

    if source_path.is_file():
        if not is_supported_document_file(source_path):
            raise ValueError(
                f"Unsupported file type: "
                f"{source_path.suffix or '<no extension>'}"
            )

        return [source_path]

    if source_path.is_dir():
        return find_document_files(
            source_path,
            recursive=recursive,
        )

    raise ValueError(
        f"Unsupported path type: {source_path}"
    )


def collect_text_files(
    source_path: Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    """
    Resolve a user-selected file or directory into TXT files.

    This function is retained for backward compatibility with
    the existing TXT ingestion pipeline.

    Args:
        source_path: User-selected file or directory.
        recursive: Whether directory scanning should include
            subdirectories.

    Returns:
        List of supported TXT file paths.

    Raises:
        TypeError: If source_path is not a pathlib.Path.
        FileNotFoundError: If source_path does not exist.
        ValueError: If the selected path or file type is
            unsupported.
    """

    _validate_path_object(
        source_path,
        argument_name="source_path",
    )

    if not isinstance(recursive, bool):
        raise TypeError(
            "recursive must be a bool."
        )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Selected path does not exist: {source_path}"
        )

    if source_path.is_file():
        if not is_supported_text_file(source_path):
            raise ValueError(
                f"Unsupported text file type: "
                f"{source_path.suffix or '<no extension>'}"
            )

        return [source_path]

    if source_path.is_dir():
        return find_text_files(
            source_path,
            recursive=recursive,
        )

    raise ValueError(
        f"Unsupported path type: {source_path}"
    )