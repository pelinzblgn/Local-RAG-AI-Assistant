from src.database import initialize_database
from src.retrieval import get_top_documents


def main() -> None:
    print("Local RAG AI Assistant")
    print("-" * 40)

    initialize_database()

    query = (
        "Çizgi takip aracında hata nasıl hesaplanır?"
    )

    print(f"\nSorgu: {query}")

    try:
        results = get_top_documents(
            query=query,
            top_k=3,
        )

        print("\nEn alakalı chunk'lar:")

        for index, document in enumerate(
            results,
            start=1,
        ):
            print(
                f"\n{index}. Sonuç"
                f"\nSkor: {document['score']:.4f}"
                f"\nKaynak: {document['source']}"
                f"\nİçerik: {document['content']}"
            )

    except Exception as error:
        print(f"\nBir hata oluştu: {error}")


if __name__ == "__main__":
    main()