from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)


def _validate_chunk_settings(
    max_characters: int,
    overlap_characters: int,
) -> None:
    """
    Validate chunking configuration values.
    """

    if max_characters <= 0:
        raise ValueError(
            "max_characters must be greater than zero."
        )

    if overlap_characters < 0:
        raise ValueError(
            "overlap_characters cannot be negative."
        )

    if overlap_characters >= max_characters:
        raise ValueError(
            "overlap_characters must be smaller "
            "than max_characters."
        )


def _split_long_text(
    text: str,
    max_characters: int,
    overlap_characters: int,
) -> list[str]:
    """
    Split long text by character limit with overlap.

    Word boundaries are preserved whenever possible.
    """

    clean_text = " ".join(
        text.split()
    )

    if not clean_text:
        return []

    chunks: list[str] = []

    start = 0
    text_length = len(clean_text)

    while start < text_length:
        end = min(
            start + max_characters,
            text_length,
        )

        if end < text_length:
            last_space = clean_text.rfind(
                " ",
                start,
                end,
            )

            if last_space > start:
                end = last_space

        chunk = clean_text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = (
            end - overlap_characters
        )

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def _extract_paragraphs(
    text: str,
) -> list[str]:
    """
    Extract normalized non-empty paragraphs.

    Blank lines are treated as paragraph boundaries.
    """

    normalized_text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    raw_paragraphs = normalized_text.split(
        "\n\n"
    )

    paragraphs: list[str] = []

    for paragraph in raw_paragraphs:
        clean_paragraph = " ".join(
            paragraph.split()
        )

        if clean_paragraph:
            paragraphs.append(
                clean_paragraph
            )

    return paragraphs


def split_into_chunks(
    text: str,
    max_characters: int = CHUNK_SIZE,
    overlap_characters: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into paragraph-aware chunks.

    Paragraph boundaries are preserved whenever possible.
    Multiple short paragraphs may be grouped into one chunk.

    If a single paragraph is larger than the configured maximum,
    character-based chunking is used as a fallback.

    Args:
        text: Source text.
        max_characters: Maximum characters per chunk.
        overlap_characters: Overlap used for oversized paragraphs.

    Returns:
        Ordered text chunks.

    Raises:
        TypeError: If text is not a string.
        ValueError: If chunk settings are invalid.
    """

    if not isinstance(text, str):
        raise TypeError(
            "Text must be a string."
        )

    _validate_chunk_settings(
        max_characters=max_characters,
        overlap_characters=overlap_characters,
    )

    if not text.strip():
        return []

    paragraphs = _extract_paragraphs(
        text
    )

    if not paragraphs:
        return []

    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_characters:
            if current_paragraphs:
                chunks.append(
                    "\n\n".join(
                        current_paragraphs
                    )
                )

                current_paragraphs = []
                current_length = 0

            long_chunks = _split_long_text(
                text=paragraph,
                max_characters=max_characters,
                overlap_characters=overlap_characters,
            )

            chunks.extend(
                long_chunks
            )

            continue

        separator_length = (
            2
            if current_paragraphs
            else 0
        )

        projected_length = (
            current_length
            + separator_length
            + len(paragraph)
        )

        if (
            current_paragraphs
            and projected_length > max_characters
        ):
            chunks.append(
                "\n\n".join(
                    current_paragraphs
                )
            )

            current_paragraphs = [
                paragraph
            ]

            current_length = len(
                paragraph
            )

            continue

        current_paragraphs.append(
            paragraph
        )

        current_length = (
            projected_length
        )

    if current_paragraphs:
        chunks.append(
            "\n\n".join(
                current_paragraphs
            )
        )

    return chunks