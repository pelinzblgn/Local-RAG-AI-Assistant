import logging
import math
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

from src.config import (
    EMBEDDING_MODEL_ALIAS,
    get_foundry_manager,
)


_embedding_model: Any | None = None
_embedding_client: Any | None = None


def _clean_texts(texts: Sequence[str]) -> list[str]:
    """
    Validate and normalize texts before embedding generation.

    Args:
        texts: Text values to normalize.

    Returns:
        Cleaned, non-empty texts.

    Raises:
        TypeError: If an item is not a string.
        ValueError: If no usable text remains.
    """

    if len(texts) == 0:
        raise ValueError("At least one text is required.")

    cleaned_texts: list[str] = []

    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(
                f"Text at index {index} must be a string."
            )

        clean_text = text.strip()

        if not clean_text:
            raise ValueError(
                f"Text at index {index} cannot be empty."
            )

        cleaned_texts.append(clean_text)

    return cleaned_texts


def _validate_embedding(
    embedding: Sequence[float],
) -> list[float]:
    """
    Validate and normalize one embedding vector.

    Args:
        embedding: Embedding returned by the model.

    Returns:
        Embedding values converted to floats.

    Raises:
        ValueError: If the embedding is empty or invalid.
    """

    if len(embedding) == 0:
        raise ValueError(
            "The embedding model returned an empty vector."
        )

    normalized_embedding: list[float] = []

    for value in embedding:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                "Embedding values must be numerical."
            )

        normalized_value = float(value)

        if not math.isfinite(normalized_value):
            raise ValueError(
                "Embedding values must be finite."
            )

        normalized_embedding.append(normalized_value)

    return normalized_embedding


def _get_embedding_client() -> Any:
    """
    Load the embedding model once and return its client.

    The initialized model and client are reused for later requests.
    """

    global _embedding_model
    global _embedding_client

    if _embedding_client is not None:
        return _embedding_client

    manager = get_foundry_manager()
    model = manager.catalog.get_model(
        EMBEDDING_MODEL_ALIAS
    )

    logger.info("Embedding modeli kontrol ediliyor.")
    
    model.download(
        lambda progress: print(
            f"\rEmbedding modeli: %{progress:.1f}",
            end="",
            flush=True,
        )
    )

    print()
    logger.info("Embedding modeli belleğe yükleniyor.")

    if not model.is_loaded:
        model.load()

    _embedding_model = model
    _embedding_client = model.get_embedding_client()

    return _embedding_client


def generate_embeddings(
    texts: Sequence[str],
) -> list[list[float]]:
    """
    Generate embeddings for one or more texts using Foundry Local.

    Args:
        texts: Texts for which embeddings will be generated.

    Returns:
        One validated embedding vector for each input text.

    Raises:
        RuntimeError: If the model returns an unexpected result count.
        TypeError: If an input value is not a string.
        ValueError: If an input text or returned embedding is invalid.
    """

    cleaned_texts = _clean_texts(texts)
    client = _get_embedding_client()

    logger.info(
    "%d metin için embedding üretiliyor.",
    len(cleaned_texts),
)

    response = client.generate_embeddings(
        cleaned_texts
    )

    embeddings = [
        _validate_embedding(item.embedding)
        for item in response.data
    ]

    if len(embeddings) != len(cleaned_texts):
        raise RuntimeError(
            "The embedding model returned an unexpected "
            "number of vectors."
        )

    embedding_dimensions = {
        len(embedding)
        for embedding in embeddings
    }

    if len(embedding_dimensions) != 1:
        raise RuntimeError(
            "The embedding model returned vectors with "
            "different dimensions."
        )

    return embeddings


def generate_embedding(text: str) -> list[float]:
    """
    Generate one embedding vector for a single text.

    Args:
        text: Text for which an embedding will be generated.

    Returns:
        Validated embedding vector.
    """

    return generate_embeddings([text])[0]

def warm_up_embedding_model() -> None:
    """
    Load and prepare the embedding model before the first query.
    """

    _get_embedding_client()

    logger.info(
        "Embedding modeli warm-up tamamlandı."
    )

def unload_embedding_model() -> None:
    """
    Unload the cached embedding model and clear its client.

    This function may be called when the application is closing.
    """

    global _embedding_model
    global _embedding_client

    if (
        _embedding_model is not None
        and _embedding_model.is_loaded
    ):
        _embedding_model.unload()

    _embedding_model = None
    _embedding_client = None