SYSTEM_PROMPT = """
You are a local document question-answering assistant.

Answer the user's question using only the supplied context.

Rules:
1. Do not use information outside the context.
2. If the answer is not available in the context, say:
   "Bu bilgi mevcut belgelerde bulunamadı."
3. Answer clearly and concisely in Turkish.
4. Mention the source files used in the answer.
""".strip()


def build_rag_prompt(
    question: str,
    retrieved_documents: list[dict],
) -> str:
    """Build a grounded prompt from retrieved document chunks."""

    clean_question = question.strip()

    if not clean_question:
        raise ValueError("Question cannot be empty.")

    if not retrieved_documents:
        raise ValueError(
            "At least one retrieved document is required."
        )

    context_parts: list[str] = []

    for index, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        context_parts.append(
            f"[Belge {index}]\n"
            f"Kaynak: {document['source']}\n"
            f"İçerik: {document['content']}"
        )

    context = "\n\n".join(context_parts)

    return (
        f"BAĞLAM:\n{context}\n\n"
        f"KULLANICI SORUSU:\n{clean_question}\n\n"
        "CEVAP:"
    )