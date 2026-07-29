from src.embeddings import cosine_similarity, generate_embeddings


def find_relevant_documents(
    query: str,
    documents: list[str],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Return the most relevant documents for a query."""

    clean_query = query.strip()
    clean_documents = [
        document.strip()
        for document in documents
        if document.strip()
    ]

    if not clean_query:
        raise ValueError("Query cannot be empty.")

    if not clean_documents:
        raise ValueError("At least one document is required.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    all_texts = clean_documents + [clean_query]
    embeddings = generate_embeddings(all_texts)

    document_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    scored_documents = []

    for document, document_embedding in zip(
        clean_documents,
        document_embeddings,
    ):
        score = cosine_similarity(
            query_embedding,
            document_embedding,
        )

        scored_documents.append((document, score))

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored_documents[:top_k]