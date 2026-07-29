from src.embeddings import cosine_similarity, generate_embeddings


def main() -> None:
    documents = [
        "PWM, motor hızını duty cycle ile kontrol eder.",
        "SQLite hafif ve yerel bir veritabanıdır.",
        "RAG, ilgili belgeleri bulup dil modeline bağlam olarak verir.",
        "PID kontrolcü P, I ve D bileşenlerinden oluşur.",
    ]

    query = "Motorun hızını nasıl ayarlayabilirim?"

    print("Local RAG AI Assistant")
    print("-" * 40)

    all_texts = documents + [query]
    embeddings = generate_embeddings(all_texts)

    document_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    scored_documents = []

    for document, document_embedding in zip(
        documents,
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

    print(f"\nSorgu: {query}")
    print("\nBenzerlik sonuçları:")

    for document, score in scored_documents:
        print(f"{score:.4f} -> {document}")

    best_document, best_score = scored_documents[0]

    print("\nEn alakalı metin:")
    print(best_document)
    print(f"Benzerlik skoru: {best_score:.4f}")


if __name__ == "__main__":
    main()