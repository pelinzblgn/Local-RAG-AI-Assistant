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
    Store a bounded number of conversation turns.

    The memory is kept only for the current application session.
    """

    def __init__(
        self,
        max_turns: int = MEMORY_MAX_TURNS,
        retrieval_turns: int = RETRIEVAL_MEMORY_TURNS,
    ) -> None:
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

    @property
    def size(self) -> int:
        """Return the number of stored conversation turns."""

        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        """Return whether the memory contains no turns."""

        return not self._turns

    def add_turn(
        self,
        question: str,
        answer: str,
    ) -> None:
        """Store one completed conversation turn."""

        if not isinstance(question, str):
            raise TypeError(
                "Question must be a string."
            )

        if not isinstance(answer, str):
            raise TypeError(
                "Answer must be a string."
            )

        clean_question = question.strip()
        clean_answer = answer.strip()

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

        if len(self._turns) > self._max_turns:
            self._turns = self._turns[
                -self._max_turns:
            ]

    def get_turns(self) -> list[ConversationTurn]:
        """Return a copy of the stored conversation history."""

        return [
            turn.copy()
            for turn in self._turns
        ]

    def build_history_text(self) -> str:
        """Format stored turns for inclusion in a model prompt."""

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

        return "\n\n".join(parts)

    def build_retrieval_query(
        self,
        question: str,
    ) -> str:
        """
        Build a retrieval query using recent user questions.

        Recent questions provide context for follow-up queries
        such as 'Peki bunun avantajı nedir?'.
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

    def clear(self) -> None:
        """Remove all conversation history."""

        self._turns.clear()