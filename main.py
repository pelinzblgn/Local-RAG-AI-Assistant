from src.llm import generate_response


def main() -> None:
    print("Local RAG AI Assistant")
    print("-" * 40)

    question = "RAG nedir? Türkçe ve iki cümleyle açıkla."

    print(f"Soru: {question}")
    print("\nModel hazırlanıyor...")

    try:
        answer = generate_response(question)
        print(f"\nCevap:\n{answer}")
    except Exception as error:
        print(f"\nBir hata oluştu: {error}")


if __name__ == "__main__":
    main()