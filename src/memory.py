from typing import TypedDict

from src.config import (
    MEMORY_MAX_TURNS,
    RETRIEVAL_MEMORY_TURNS,
)


class ConversationTurn(TypedDict):
    """One completed user-assistant conversation turn."""

    question: str
    answer: str


class ConversationMemory:
    """
    Store successful conversation turns and latest user-query context.

    Two different memory concepts are intentionally separated:

    1. Completed conversation history
       Only successful user-assistant turns are stored here.
       This history may later be supplied to the language model.

    2. Latest user query
       The most recent user question is stored independently,
       even if retrieval confidence was too low to answer it.

       This allows follow-up questions such as:
           "Bunun avantajı nedir?"

       to refer to the immediately preceding user question without
       treating a failed answer as factual conversation history.

    Memory exists only for the current application session.
    """

    def __init__(
        self,
        max_turns: int = MEMORY_MAX_TURNS,
        retrieval_turns: int = RETRIEVAL_MEMORY_TURNS,
    ) -> None:
        """
        Initialize conversation memory.

        Args:
            max_turns:
                Maximum number of successful conversation turns
                retained in memory.

            retrieval_turns:
                Number of recent successful turns used by the
                legacy retrieval-query builder.

        Raises:
            ValueError:
                If configuration values are invalid.
        """

        if max_turns <= 0:
            raise ValueError(
                "max_turns must be greater than zero."
            )

        if retrieval_turns < 0:
            raise ValueError(
                "retrieval_turns cannot be negative."
            )

        if retrieval_turns > max_turns:
            raise ValueError(
                "retrieval_turns cannot be greater than max_turns."
            )

        self._max_turns = max_turns
        self._retrieval_turns = retrieval_turns

        self._turns: list[ConversationTurn] = []

        self._last_user_query: str | None = None

    @property
    def size(self) -> int:
        """Return number of stored successful conversation turns."""

        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        """Return whether successful conversation history is empty."""

        return not self._turns

    @property
    def last_user_query(self) -> str | None:
        """
        Return the most recently observed user question.

        This value may exist even when the previous question
        did not produce a successful assistant response.
        """

        return self._last_user_query

    def record_user_query(
        self,
        question: str,
    ) -> None:
        """
        Record the latest user query independently of answer success.

        This state is used by conversational query rewriting.
        """

        if not isinstance(
            question,
            str,
        ):
            raise TypeError(
                "Question must be a string."
            )

        clean_question = (
            question.strip()
        )

        if not clean_question:
            raise ValueError(
                "Question cannot be empty."
            )

        self._last_user_query = (
            clean_question
        )

    def add_turn(
        self,
        question: str,
        answer: str,
    ) -> None:
        """
        Store one completed successful conversation turn.

        The corresponding question also becomes the latest user query.
        """

        if not isinstance(
            question,
            str,
        ):
            raise TypeError(
                "Question must be a string."
            )

        if not isinstance(
            answer,
            str,
        ):
            raise TypeError(
                "Answer must be a string."
            )

        clean_question = (
            question.strip()
        )

        clean_answer = (
            answer.strip()
        )

        if not clean_question:
            raise ValueError(
                "Question cannot be empty."
            )

        if not clean_answer:
            raise ValueError(
                "Answer cannot be empty."
            )

        self._turns.append(
            {
                "question": clean_question,
                "answer": clean_answer,
            }
        )

        self._last_user_query = (
            clean_question
        )

        if len(self._turns) > self._max_turns:
            self._turns = self._turns[
                -self._max_turns:
            ]

    def get_turns(
        self,
    ) -> list[ConversationTurn]:
        """
        Return a copy of successful conversation history.
        """

        return [
            turn.copy()
            for turn in self._turns
        ]

    def build_history_text(
        self,
    ) -> str:
        """
        Format successful turns for inclusion in the model prompt.

        Low-confidence or unanswered questions are intentionally
        excluded from this factual conversation history.
        """

        if not self._turns:
            return ""

        parts: list[str] = []

        for index, turn in enumerate(
            self._turns,
            start=1,
        ):
            parts.append(
                f"[Konuşma {index}]\n"
                f"Kullanıcı: {turn['question']}\n"
                f"Asistan: {turn['answer']}"
            )

        return "\n\n".join(
            parts
        )

    def build_rewrite_context(
        self,
    ) -> str:
        """
        Build minimal context for conversational query rewriting.

        The latest user question is included even if that question
        produced a low-confidence fallback.

        Returns:
            A minimal formatted user-query context or an empty string.
        """

        if self._last_user_query is None:
            return ""

        return (
            "Kullanıcı: "
            f"{self._last_user_query}"
        )

    def build_retrieval_query(
        self,
        question: str,
    ) -> str:
        """
        Build the legacy retrieval query using recent successful turns.

        This method is retained for compatibility with existing code
        and tests. The current RAG pipeline uses the dedicated
        conversational query rewriter instead.
        """

        if not isinstance(
            question,
            str,
        ):
            raise TypeError(
                "Question must be a string."
            )

        clean_question = (
            question.strip()
        )

        if not clean_question:
            raise ValueError(
                "Question cannot be empty."
            )

        if (
            self._retrieval_turns == 0
            or not self._turns
        ):
            return clean_question

        recent_turns = self._turns[
            -self._retrieval_turns:
        ]

        previous_questions = "\n".join(
            f"- {turn['question']}"
            for turn in recent_turns
        )

        return (
            "Önceki ilgili kullanıcı soruları:\n"
            f"{previous_questions}\n\n"
            "Güncel kullanıcı sorusu:\n"
            f"{clean_question}"
        )

    def clear(
        self,
    ) -> None:
        """
        Clear successful conversation history and latest-query state.
        """

        self._turns.clear()

        self._last_user_query = None