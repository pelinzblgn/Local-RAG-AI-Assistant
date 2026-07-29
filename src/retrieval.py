from src.database import get_all_documents
from src.embeddings import cosine_similarity, generate_embeddings


def get_top_documents(
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """Return the most relevant stored documents for a query."""

    clean_query = query.strip()

    if not clean_query:
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    stored_documents = get_all_documents()

    if not stored_documents:
        raise RuntimeError(
            "The database does not contain any documents."
        )

    print("Sorgu embedding'i oluşturuluyor...")

    query_embedding = generate_embeddings(
        [clean_query]
    )[0]

    scored_documents = []

    for document in stored_documents:
        score = cosine_similarity(
            query_embedding,
            document["embedding"],
        )

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

    return scored_documents[:top_k]