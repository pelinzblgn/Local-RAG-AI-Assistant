from src.database import get_all_documents
from src.ingestion import ingest_text_files


def main() -> None:
    print("Local RAG AI Assistant")
    print("-" * 40)

    try:
        inserted_count = ingest_text_files(
            reset_database=True
        )

        print(
            f"\n{inserted_count} chunk "
            "veritabanına kaydedildi."
        )

        stored_documents = get_all_documents()

        print(
            f"SQLite toplam kayıt: "
            f"{len(stored_documents)}"
        )

        for document in stored_documents:
            print(
                f"\nID: {document['id']}"
                f"\nKaynak: {document['source']}"
                f"\nİçerik: {document['content']}"
                f"\nEmbedding boyutu: "
                f"{len(document['embedding'])}"
            )

    except Exception as error:
        print(f"\nBir hata oluştu: {error}")


if __name__ == "__main__":
    main()