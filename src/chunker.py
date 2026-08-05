from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_into_chunks(
    text: str,
    max_characters: int = CHUNK_SIZE,
    overlap_characters: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into size-limited chunks with overlap.

    Words are not divided in the middle when a suitable
    whitespace boundary is available.

    Args:
        text: Source text.
        max_characters: Maximum number of characters per chunk.
        overlap_characters: Approximate overlap between chunks.

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

        chunk = clean_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - overlap_characters

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks