import logging
from typing import TypedDict

from src.config import (
    MIN_SIMILARITY_SCORE,
    TOP_K,
)
from src.llm import LocalLLM
from src.memory import ConversationMemory
from src.prompts import (
    DEFAULT_FALLBACK_MESSAGE,
    PromptBuilder,
)
from src.retrieval import (
    RetrievedDocument,
    get_top_documents,
)


logger = logging.getLogger(__name__)


class AssistantResponse(TypedDict):
    """Structured response returned by the RAG assistant."""

    answer: str
    sources: list[str]
    retrieved_documents: list[RetrievedDocument]


class RAGAssistant:
    """
    Orchestrate retrieval, prompt construction, local generation,
    and session-level conversation memory.
    """

    def __init__(
        self,
        local_llm: LocalLLM | None = None,
        prompt_builder: PromptBuilder | None = None,
        memory: ConversationMemory | None = None,
        top_k: int = TOP_K,
        minimum_score: float = MIN_SIMILARITY_SCORE,
    ) -> None:
        """
        Initialize the RAG assistant.

        Args:
            local_llm: Optional local LLM instance.
            prompt_builder: Optional prompt builder instance.
            memory: Optional conversation memory instance.
            top_k: Maximum number of retrieved documents.
            minimum_score: Minimum accepted similarity score.

        Raises:
            ValueError: If retrieval settings are invalid.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not -1.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between -1.0 and 1.0."
            )

        self._local_llm = (
            local_llm
            if local_llm is not None
            else LocalLLM()
        )

        self._prompt_builder = (
            prompt_builder
            if prompt_builder is not None
            else PromptBuilder()
        )

        self._memory = (
            memory
            if memory is not None
            else ConversationMemory()
        )

        self._top_k = top_k
        self._minimum_score = minimum_score

    @property
    def memory_size(self) -> int:
        """Return the number of stored conversation turns."""

        return self._memory.size
    def warm_up(self) -> None:
        """Prepare the local chat model before user interaction."""

        self._local_llm.warm_up()
    def answer(
        self,
        question: str,
    ) -> AssistantResponse:
        """
        Answer a question using the local RAG pipeline.

        Recent conversation history is used to enrich the retrieval
        query. A successful answer is stored in session memory.

        Args:
            question: User question.

        Returns:
            Answer text, source names, and retrieved documents.

        Raises:
            TypeError: If question is not a string.
            ValueError: If question is empty.
        """

        if not isinstance(question, str):
            raise TypeError(
                "Question must be a string."
            )

        clean_question = question.strip()

        if not clean_question:
            raise ValueError(
                "Question cannot be empty."
            )

        retrieval_query = (
            self._memory.build_retrieval_query(
                clean_question
            )
        )

        logger.debug(
            "Retrieval sorgusu oluşturuldu: %s",
            retrieval_query,
        )

        retrieved_documents = get_top_documents(
            query=retrieval_query,
            top_k=self._top_k,
            minimum_score=self._minimum_score,
        )

        if not retrieved_documents:
            logger.warning(
                "Sorgu için yeterli benzerlikte belge bulunamadı."
            )

            return {
                "answer": DEFAULT_FALLBACK_MESSAGE,
                "sources": [],
                "retrieved_documents": [],
            }

        logger.info(
            "%d belge bağlam olarak seçildi.",
            len(retrieved_documents),
        )

        conversation_history = (
            self._memory.build_history_text()
        )

        system_prompt, user_prompt = (
            self._prompt_builder.build(
                question=clean_question,
                retrieved_documents=retrieved_documents,
                conversation_history=conversation_history,
            )
        )

        generated_answer = self._local_llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        self._memory.add_turn(
            question=clean_question,
            answer=generated_answer,
        )

        sources = list(
            dict.fromkeys(
                document["source"]
                for document in retrieved_documents
            )
        )

        return {
            "answer": generated_answer,
            "sources": sources,
            "retrieved_documents": retrieved_documents,
        }

    def clear_memory(self) -> None:
        """Clear the current conversation history."""

        self._memory.clear()

        logger.info(
            "Konuşma hafızası temizlendi."
        )

    def get_conversation_history(self) -> str:
        """Return formatted conversation history."""

        return self._memory.build_history_text()

    def close(self) -> None:
        """Release the local chat model."""

        self._local_llm.unload()

    def __enter__(self) -> "RAGAssistant":
        """Return the assistant for context-manager usage."""

        return self

    def __exit__(
        self,
        exception_type: object,
        exception_value: object,
        traceback: object,
    ) -> None:
        """Release resources when leaving the context block."""

        self.close()