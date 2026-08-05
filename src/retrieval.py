import logging
from typing import TypedDict

from src.config import (
    MIN_SIMILARITY_SCORE,
    TOP_K,
)
from src.database import get_all_documents
from src.embeddings import generate_embedding
from src.similarity import cosine_similarity


logger = logging.getLogger(__name__)


class RetrievedDocument(TypedDict):
    """Structure of a document returned by semantic retrieval."""

    id: int
    content: str
    source: str
    score: float


def get_top_documents(
    query: str,
    top_k: int = TOP_K,
    minimum_score: float | None = MIN_SIMILARITY_SCORE,
) -> list[RetrievedDocument]:
    """
    Return the most relevant stored documents for a query.

    Args:
        query: User question or semantic search query.
        top_k: Maximum number of documents to return.
        minimum_score: Optional minimum cosine similarity score.

    Returns:
        Documents ordered from highest to lowest similarity.
    """

    if not isinstance(query, str):
        raise TypeError(
            "Query must be a string."
        )

    clean_query = query.strip()

    if not clean_query:
        raise ValueError(
            "Query cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if minimum_score is not None:
        if not -1.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between -1.0 and 1.0."
            )

    stored_documents = get_all_documents()

    if not stored_documents:
        raise RuntimeError(
            "The database does not contain any documents."
        )

    logger.info(
        "Retrieval başladı. Belge sayısı: %d, top_k: %d",
        len(stored_documents),
        top_k,
    )

    query_embedding = generate_embedding(
        clean_query
    )

    scored_documents: list[
        RetrievedDocument
    ] = []

    for document in stored_documents:
        score = cosine_similarity(
            query_embedding,
            document["embedding"],
        )

        if (
            minimum_score is not None
            and score < minimum_score
        ):
            continue

        scored_documents.append(
            {
                "id": document["id"],
                "content": document["content"],
                "source": document["source"],
                "score": score,
            }
        )

    scored_documents.sort(
        key=lambda document: document["score"],
        reverse=True,
    )

    results = scored_documents[
        :top_k
    ]

    logger.info(
        "Retrieval tamamlandı. Dönen sonuç sayısı: %d",
        len(results),
    )

    for index, result in enumerate(
        results,
        start=1,
    ):
        logger.debug(
            "Sonuç %d | Kaynak: %s | Skor: %.4f",
            index,
            result["source"],
            result["score"],
        )

    return results