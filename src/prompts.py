from collections.abc import Mapping, Sequence


DEFAULT_FALLBACK_MESSAGE = (
    "Bu bilgi mevcut yerel belgelerde bulunamadı."
)


class PromptBuilder:
    """
    Build grounded prompts for the local RAG assistant.

    The builder keeps system instructions, retrieved context,
    the user question, and answer-format rules separate.
    """

    def __init__(
        self,
        fallback_message: str = DEFAULT_FALLBACK_MESSAGE,
    ) -> None:
        """
        Initialize the prompt builder.

        Args:
            fallback_message: Exact response to use when the
                supplied context does not contain the answer.

        Raises:
            ValueError: If the fallback message is empty.
        """

        clean_fallback_message = fallback_message.strip()

        if not clean_fallback_message:
            raise ValueError(
                "Fallback message cannot be empty."
            )

        self._fallback_message = clean_fallback_message

    @property
    def fallback_message(self) -> str:
        """Return the configured fallback message."""

        return self._fallback_message

    def build_system_prompt(self) -> str:
        """
        Build the system instructions for grounded Q&A.

        Returns:
            System prompt that restricts the model to the
            retrieved local context.
        """

        return (
            "You are a local document question-answering assistant.\n\n"
            "Follow these rules strictly:\n"
            "1. Answer using only the supplied local document context.\n"
            "2. Do not use outside knowledge or make assumptions.\n"
            "3. If the context does not contain enough information, "
            f'reply exactly: "{self._fallback_message}"\n'
            "4. Answer clearly and concisely in Turkish.\n"
            "5. Do not invent facts, file names, or source references.\n"
            "6. At the end of a supported answer, list the source "
            "file names under a 'Kaynaklar:' heading.\n"
            "7. Use only source names that appear in the supplied context."
        )

    def build_context(
        self,
        retrieved_documents: Sequence[Mapping[str, object]],
    ) -> str:
        """
        Format retrieved documents as numbered context blocks.

        Each document must contain non-empty ``source`` and
        ``content`` string values.

        Args:
            retrieved_documents: Retrieved document records.

        Returns:
            Formatted context text.

        Raises:
            ValueError: If no documents are supplied or required
                values are missing.
            TypeError: If a document is not a mapping.
        """

        if len(retrieved_documents) == 0:
            raise ValueError(
                "At least one retrieved document is required."
            )

        context_parts: list[str] = []

        for index, document in enumerate(
            retrieved_documents,
            start=1,
        ):
            if not isinstance(document, Mapping):
                raise TypeError(
                    f"Document at index {index - 1} "
                    "must be a mapping."
                )

            source = document.get("source")
            content = document.get("content")

            if not isinstance(source, str):
                raise ValueError(
                    f"Document at index {index - 1} "
                    "must contain a string source."
                )

            if not isinstance(content, str):
                raise ValueError(
                    f"Document at index {index - 1} "
                    "must contain string content."
                )

            clean_source = source.strip()
            clean_content = content.strip()

            if not clean_source:
                raise ValueError(
                    f"Document at index {index - 1} "
                    "has an empty source."
                )

            if not clean_content:
                raise ValueError(
                    f"Document at index {index - 1} "
                    "has empty content."
                )

            context_parts.append(
                f"[Belge {index}]\n"
                f"Kaynak: {clean_source}\n"
                f"İçerik:\n{clean_content}"
            )

        return "\n\n".join(context_parts)

    def build_user_prompt(
        self,
        question: str,
        retrieved_documents: Sequence[Mapping[str, object]],
    ) -> str:
        """
        Build the user prompt from a question and retrieved context.

        Args:
            question: User question.
            retrieved_documents: Documents selected by retrieval.

        Returns:
            Grounded RAG user prompt.

        Raises:
            TypeError: If the question is not a string.
            ValueError: If the question is empty.
        """

        if not isinstance(question, str):
            raise TypeError("Question must be a string.")

        clean_question = question.strip()

        if not clean_question:
            raise ValueError("Question cannot be empty.")

        context = self.build_context(
            retrieved_documents
        )

        return (
            "YEREL BELGE BAĞLAMI\n"
            "===================\n"
            f"{context}\n\n"
            "KULLANICI SORUSU\n"
            "=================\n"
            f"{clean_question}\n\n"
            "YANIT TALİMATI\n"
            "===============\n"
            "Soruyu yalnızca yukarıdaki belge bağlamına göre "
            "yanıtla. Bağlam yeterli değilse sistem mesajındaki "
            "sabit bulunamadı yanıtını kullan."
        )

    def build(
        self,
        question: str,
        retrieved_documents: Sequence[Mapping[str, object]],
    ) -> tuple[str, str]:
        """
        Build both system and user prompts.

        Returns:
            Tuple containing ``system_prompt`` and ``user_prompt``.
        """

        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(
            question=question,
            retrieved_documents=retrieved_documents,
        )

        return system_prompt, user_prompt