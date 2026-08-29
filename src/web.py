from pathlib import Path

from fastapi.responses import (
    FileResponse,
)


WEB_DIRECTORY = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    / "web"
)


INDEX_FILE = (
    WEB_DIRECTORY
    / "index.html"
)


def get_index_file() -> FileResponse:
    """
    Return the Local RAG web application entry page.

    Raises:
        FileNotFoundError:
            If web/index.html does not exist.
    """

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            "Web interface index file was not found."
        )

    return FileResponse(
        INDEX_FILE
    )