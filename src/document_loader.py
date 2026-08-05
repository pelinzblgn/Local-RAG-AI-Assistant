from pathlib import Path


def read_text_file(file_path: Path) -> str:
    """
    Read and return the contents of a UTF-8 TXT file.

    Args:
        file_path: Path of the text file.

    Returns:
        Whitespace-trimmed file content.

    Raises:
        TypeError: If file_path is not a pathlib.Path.
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file, the extension is
            unsupported, or the file is empty.
        RuntimeError: If the file cannot be decoded as UTF-8.
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

    if file_path.suffix.lower() != ".txt":
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}"
        )

    try:
        content = file_path.read_text(
            encoding="utf-8"
        ).strip()
    except UnicodeDecodeError as error:
        raise RuntimeError(
            f"File is not valid UTF-8: {file_path}"
        ) from error

    if not content:
        raise ValueError(
            f"Text file is empty: {file_path.name}"
        )

    return content


def find_text_files(
    directory: Path,
) -> list[Path]:
    """
    Find TXT files directly inside a directory.

    Args:
        directory: Directory to scan.

    Returns:
        Alphabetically ordered TXT file paths.

    Raises:
        TypeError: If directory is not a pathlib.Path.
        FileNotFoundError: If the directory does not exist.
        ValueError: If the path is not a directory.
    """

    if not isinstance(directory, Path):
        raise TypeError(
            "directory must be a pathlib.Path object."
        )

    if not directory.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Data path is not a directory: {directory}"
        )

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".txt"
    )