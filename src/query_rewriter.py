import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRewriteResult:
    """
    Result of conversational query rewriting.
    """

    original_query: str
    rewritten_query: str
    was_rewritten: bool
    reason: str


_FOLLOW_UP_PREFIXES = (
    "peki",
    "peki ya",
    "bunun",
    "buna",
    "bunu",
    "bundan",
    "şunun",
    "şuna",
    "şunu",
    "şundan",
    "onun",
    "ona",
    "onu",
    "ondan",
    "orada",
    "burada",
    "aynı",
    "then",
    "what about",
    "how about",
)


_REFERENCE_TERMS = {
    "bu",
    "buna",
    "bunu",
    "bunun",
    "bundan",
    "şu",
    "şuna",
    "şunu",
    "şunun",
    "şundan",
    "o",
    "ona",
    "onu",
    "onun",
    "ondan",
    "orada",
    "burada",
    "aynı",
    "it",
    "this",
    "that",
}


_GENERIC_FOLLOW_UP_STEMS = (
    "avantaj",
    "dezavantaj",
    "özellik",
    "bileşen",
    "fark",
    "amaç",
    "amac",
    "kullan",
    "çalış",
    "calis",
    "işe",
    "ise",
    "yar",
    "nasıl",
    "nasil",
    "neden",
    "nel",
    "ne",
)


def _clean_text(
    text: str,
) -> str:
    """
    Normalize whitespace without changing semantic content.
    """

    return " ".join(
        text.strip().split()
    )


def _tokenize(
    text: str,
) -> list[str]:
    """
    Extract lowercase word tokens.
    """

    return re.findall(
        r"\b[\wçğıöşüÇĞİÖŞÜ]+\b",
        text.lower(),
        flags=re.UNICODE,
    )


def _matches_generic_follow_up_term(
    token: str,
) -> bool:
    """
    Return whether a token looks like generic follow-up wording.
    """

    return any(
        token.startswith(stem)
        for stem in _GENERIC_FOLLOW_UP_STEMS
    )


def _is_generic_short_follow_up(
    query: str,
) -> bool:
    """
    Detect short context-dependent questions without blindly
    treating every short question as a follow-up.

    Examples that should be follow-ups:
        Avantajları neler?
        Nasıl çalışır?

    Examples that should remain standalone:
        STM32 nedir?
        PWM nedir?
        PID kontrol nedir?
    """

    tokens = _tokenize(
        query
    )

    if not tokens:
        return False

    if len(tokens) > 4:
        return False

    return all(
        _matches_generic_follow_up_term(
            token
        )
        for token in tokens
    )


def _looks_like_follow_up(
    query: str,
) -> bool:
    """
    Determine whether a query appears to depend on previous
    conversation context.
    """

    clean_query = _clean_text(
        query
    )

    lowered = clean_query.lower()

    if any(
        lowered.startswith(prefix)
        for prefix in _FOLLOW_UP_PREFIXES
    ):
        return True

    tokens = set(
        _tokenize(
            clean_query
        )
    )

    if tokens.intersection(
        _REFERENCE_TERMS
    ):
        return True

    if _is_generic_short_follow_up(
        clean_query
    ):
        return True

    return False


def _extract_last_user_question(
    conversation_history: str,
) -> str | None:
    """
    Extract the most recent user question from formatted
    conversation context.

    Supported lines:

        Kullanıcı: ...
        User: ...
    """

    if not conversation_history.strip():
        return None

    candidate: str | None = None

    for raw_line in conversation_history.splitlines():
        line = raw_line.strip()
        lowered = line.lower()

        if lowered.startswith(
            "kullanıcı:"
        ):
            value = line.split(
                ":",
                maxsplit=1,
            )[1].strip()

            if value:
                candidate = value

        elif lowered.startswith(
            "user:"
        ):
            value = line.split(
                ":",
                maxsplit=1,
            )[1].strip()

            if value:
                candidate = value

    return candidate


def rewrite_query(
    query: str,
    conversation_history: str = "",
) -> QueryRewriteResult:
    """
    Rewrite a context-dependent follow-up into a more
    self-contained retrieval query.

    The implementation is deterministic and does not require
    another LLM call.

    The rewritten query is intended for retrieval only.
    """

    if not isinstance(
        query,
        str,
    ):
        raise TypeError(
            "Query must be a string."
        )

    if not isinstance(
        conversation_history,
        str,
    ):
        raise TypeError(
            "Conversation history must be a string."
        )

    clean_query = _clean_text(
        query
    )

    if not clean_query:
        raise ValueError(
            "Query cannot be empty."
        )

    clean_history = (
        conversation_history.strip()
    )

    if not clean_history:
        return QueryRewriteResult(
            original_query=clean_query,
            rewritten_query=clean_query,
            was_rewritten=False,
            reason=(
                "No conversation history is available."
            ),
        )

    if not _looks_like_follow_up(
        clean_query
    ):
        return QueryRewriteResult(
            original_query=clean_query,
            rewritten_query=clean_query,
            was_rewritten=False,
            reason=(
                "The query appears to be self-contained."
            ),
        )

    previous_question = (
        _extract_last_user_question(
            clean_history
        )
    )

    if previous_question is None:
        return QueryRewriteResult(
            original_query=clean_query,
            rewritten_query=clean_query,
            was_rewritten=False,
            reason=(
                "No previous user question could be extracted."
            ),
        )

    if (
        previous_question.strip().lower()
        == clean_query.lower()
    ):
        return QueryRewriteResult(
            original_query=clean_query,
            rewritten_query=clean_query,
            was_rewritten=False,
            reason=(
                "The previous question is identical "
                "to the current query."
            ),
        )

    rewritten_query = (
        f"{previous_question} {clean_query}"
    )

    return QueryRewriteResult(
        original_query=clean_query,
        rewritten_query=rewritten_query,
        was_rewritten=True,
        reason=(
            "The current query appears to be a follow-up, "
            "so the previous user question was added "
            "to retrieval context."
        ),
    )