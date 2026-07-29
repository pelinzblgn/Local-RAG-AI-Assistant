from src.database import initialize_database
from src.llm import generate_response
from src.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.retrieval import get_top_documents


def main() -> None:
    print("Local RAG AI Assistant")
    print("-" * 50)

    initialize_database()

    question = (
        "Çizgi takip sisteminde hata nasıl hesaplanır?"
    )

    print(f"\nSoru: {question}")

    try:
        retrieved_documents = get_top_documents(
            query=question,
            top_k=3,
        )

        print("\nBulunan bağlamlar:")

        for index, document in enumerate(
            retrieved_documents,
            start=1,
        ):
            print(
                f"\n{index}. sonuç"
                f"\nSkor: {document['score']:.4f}"
                f"\nKaynak: {document['source']}"
                f"\nİçerik: {document['content']}"
            )

        rag_prompt = build_rag_prompt(
            question=question,
            retrieved_documents=retrieved_documents,
        )

        answer = generate_response(
            prompt=rag_prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        print("\n" + "=" * 50)
        print("RAG CEVABI")
        print("=" * 50)
        print(answer)

    except Exception as error:
        print(f"\nBir hata oluştu: {error}")


if __name__ == "__main__":
    main()