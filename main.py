from src.database import (
    delete_all_documents,
    get_all_documents,
    initialize_database,
    insert_document,
)
from src.embeddings import generate_embeddings


def main() -> None:
    documents = [
        {
            "content": "PWM, motor hızını duty cycle ile kontrol eder.",
            "source": "stm32_notes.txt",
        },
        {
            "content": "SQLite hafif ve yerel bir veritabanıdır.",
            "source": "database_notes.txt",
        },
        {
            "content": (
                "RAG, ilgili belgeleri bulup dil modeline "
                "bağlam olarak verir."
            ),
            "source": "rag_notes.txt",
        },
        {
            "content": "PID kontrolcü P, I ve D bileşenlerinden oluşur.",
            "source": "pid_notes.txt",
        },
    ]

    print("Local RAG AI Assistant")
    print("-" * 40)

    initialize_database()
    delete_all_documents()

    contents = [
        document["content"]
        for document in documents
    ]

    embeddings = generate_embeddings(contents)

    for document, embedding in zip(documents, embeddings):
        document_id = insert_document(
            content=document["content"],
            source=document["source"],
            embedding=embedding,
        )

        print(f"Belge kaydedildi. ID: {document_id}")

    stored_documents = get_all_documents()

    print(f"\nToplam kayıt: {len(stored_documents)}")

    for document in stored_documents:
        print(
            f"\nID: {document['id']}"
            f"\nKaynak: {document['source']}"
            f"\nİçerik: {document['content']}"
            f"\nEmbedding boyutu: {len(document['embedding'])}"
        )


if __name__ == "__main__":
    main()