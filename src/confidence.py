import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol


class ScoredDocument(Protocol):
    """
    Minimal interface required by the confidence engine.
    """

    def __getitem__(
        self,
        key: str,
    ) -> object:
        ...


class ConfidenceLevel(str, Enum):
    """Retrieval confidence categories."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class ConfidenceResult:
    """Structured retrieval-confidence result."""

    level: ConfidenceLevel

    top_score: float
    second_score: float | None
    score_gap: float | None

    selected_count: int
    total_count: int

    evidence_coverage: float

    threshold: float
    reason: str

    @property
    def is_confident(self) -> bool:
        """
        Return whether context is reliable enough
        for grounded answer generation.
        """

        return self.level in {
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
        }

    @property
    def filtered_count(self) -> int:
        """Return number of filtered retrieval candidates."""

        return max(
            0,
            self.total_count - self.selected_count,
        )


@dataclass(frozen=True)
class ConfidencePolicy:
    """
    Configuration for confidence-aware retrieval.

    Confidence combines:
    - semantic similarity
    - ranking separation
    - adaptive context filtering
    - direct lexical evidence
    - strong-semantic override protection
    """

    low_score_threshold: float = 0.30
    medium_score_threshold: float = 0.45
    high_score_threshold: float = 0.60

    strong_gap_threshold: float = 0.15

    context_score_ratio: float = 0.75
    minimum_context_score: float = 0.30

    minimum_evidence_coverage: float = 0.25
    high_evidence_coverage: float = 0.45

    semantic_override_threshold: float = 0.70
    semantic_override_gap: float = 0.20
    semantic_override_minimum_evidence: float = 0.20

    max_context_documents: int = 3

    def __post_init__(self) -> None:
        """Validate policy configuration."""

        thresholds = (
            self.low_score_threshold,
            self.medium_score_threshold,
            self.high_score_threshold,
            self.semantic_override_threshold,
        )

        for threshold in thresholds:
            if not isfinite(threshold):
                raise ValueError(
                    "Confidence thresholds must be finite."
                )

            if not -1.0 <= threshold <= 1.0:
                raise ValueError(
                    "Confidence thresholds must be between "
                    "-1.0 and 1.0."
                )

        if not (
            self.low_score_threshold
            <= self.medium_score_threshold
            <= self.high_score_threshold
        ):
            raise ValueError(
                "Confidence thresholds must be ordered."
            )

        if not 0.0 <= self.context_score_ratio <= 1.0:
            raise ValueError(
                "context_score_ratio must be between 0 and 1."
            )

        if not -1.0 <= self.minimum_context_score <= 1.0:
            raise ValueError(
                "minimum_context_score must be between "
                "-1.0 and 1.0."
            )

        if self.strong_gap_threshold < 0.0:
            raise ValueError(
                "strong_gap_threshold cannot be negative."
            )

        if self.semantic_override_gap < 0.0:
            raise ValueError(
                "semantic_override_gap cannot be negative."
            )

        evidence_values = (
            self.minimum_evidence_coverage,
            self.high_evidence_coverage,
            self.semantic_override_minimum_evidence,
        )

        for value in evidence_values:
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "Evidence coverage thresholds must be "
                    "between 0 and 1."
                )

        if (
            self.minimum_evidence_coverage
            > self.high_evidence_coverage
        ):
            raise ValueError(
                "Evidence coverage thresholds must be ordered."
            )

        if self.max_context_documents <= 0:
            raise ValueError(
                "max_context_documents must be greater than zero."
            )


DEFAULT_CONFIDENCE_POLICY = ConfidencePolicy()


_STOP_WORDS = {
    "acaba",
    "ama",
    "ancak",
    "bana",
    "bir",
    "biri",
    "bunun",
    "bu",
    "da",
    "de",
    "daha",
    "gibi",
    "hangi",
    "ile",
    "icin",
    "için",
    "ise",
    "işe",
    "mi",
    "mı",
    "mu",
    "mü",
    "nasıl",
    "nasil",
    "neden",
    "ne",
    "nedir",
    "nelerdir",
    "peki",
    "şey",
    "su",
    "şu",
    "ve",
    "veya",
    "yarar",
    "yapar",
    "nin",
    "nın",
    "nun",
    "nün",
    "in",
    "ın",
    "un",
    "ün",
    "a",
    "an",
    "and",
    "are",
    "for",
    "how",
    "is",
    "of",
    "or",
    "the",
    "to",
    "what",
    "why",
}


def _normalize_text(
    text: str,
) -> str:
    """Normalize text for lexical evidence matching."""

    normalized = unicodedata.normalize(
        "NFKC",
        text.lower(),
    )

    normalized = normalized.replace(
        "ı",
        "i",
    )

    normalized = re.sub(
        r"[^\wçğıöşü]+",
        " ",
        normalized,
        flags=re.UNICODE,
    )

    return " ".join(
        normalized.split()
    )


def _tokenize_meaningful_terms(
    text: str,
) -> set[str]:
    """Extract meaningful evidence terms."""

    normalized = _normalize_text(
        text
    )

    return {
        token
        for token in normalized.split()
        if (
            len(token) >= 3
            and token not in _STOP_WORDS
        )
    }


def _terms_match(
    query_term: str,
    context_term: str,
) -> bool:
    """
    Compare query/context terms with lightweight
    Turkish suffix tolerance.
    """

    if query_term == context_term:
        return True

    minimum_prefix_length = 5

    if (
        len(query_term) < minimum_prefix_length
        or len(context_term) < minimum_prefix_length
    ):
        return False

    prefix_length = min(
        len(query_term),
        len(context_term),
        6,
    )

    return (
        query_term[:prefix_length]
        == context_term[:prefix_length]
    )


def calculate_evidence_coverage(
    query: str,
    documents: list[ScoredDocument],
) -> float:
    """
    Measure direct lexical evidence between question
    and selected document context.
    """

    if not isinstance(
        query,
        str,
    ):
        raise TypeError(
            "Query must be a string."
        )

    clean_query = query.strip()

    if not clean_query:
        raise ValueError(
            "Query cannot be empty."
        )

    query_terms = (
        _tokenize_meaningful_terms(
            clean_query
        )
    )

    if not query_terms:
        return 0.0

    context_terms: set[str] = set()

    for document in documents:
        raw_content = document["content"]

        if not isinstance(
            raw_content,
            str,
        ):
            raise ValueError(
                "Retrieved document content must be a string."
            )

        context_terms.update(
            _tokenize_meaningful_terms(
                raw_content
            )
        )

    if not context_terms:
        return 0.0

    matched_terms = 0

    for query_term in query_terms:
        if any(
            _terms_match(
                query_term,
                context_term,
            )
            for context_term in context_terms
        ):
            matched_terms += 1

    return (
        matched_terms
        / len(query_terms)
    )


def _extract_score(
    document: ScoredDocument,
) -> float:
    """Read and validate one similarity score."""

    raw_score = document["score"]

    if (
        isinstance(
            raw_score,
            bool,
        )
        or not isinstance(
            raw_score,
            (int, float),
        )
    ):
        raise ValueError(
            "Retrieved document score must be numerical."
        )

    score = float(
        raw_score
    )

    if not isfinite(score):
        raise ValueError(
            "Retrieved document score must be finite."
        )

    if not -1.0 <= score <= 1.0:
        raise ValueError(
            "Retrieved document score must be between "
            "-1.0 and 1.0."
        )

    return score


def select_confident_documents(
    documents: list[ScoredDocument],
    policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY,
) -> list[ScoredDocument]:
    """
    Select context using adaptive similarity filtering.
    """

    if not documents:
        return []

    scored_documents = [
        (
            document,
            _extract_score(
                document
            ),
        )
        for document in documents
    ]

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    top_score = (
        scored_documents[0][1]
    )

    adaptive_threshold = max(
        policy.minimum_context_score,
        top_score
        * policy.context_score_ratio,
    )

    selected = [
        document
        for document, score
        in scored_documents
        if score >= adaptive_threshold
    ]

    return selected[
        :policy.max_context_documents
    ]


def evaluate_retrieval_confidence(
    documents: list[ScoredDocument],
    query: str | None = None,
    policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY,
) -> ConfidenceResult:
    """
    Evaluate retrieval confidence.

    A strong-semantic override protects against false negatives
    caused by lexical wording differences.

    The override NEVER accepts zero-evidence retrieval.
    """

    if not documents:
        return ConfidenceResult(
            level=ConfidenceLevel.LOW,
            top_score=0.0,
            second_score=None,
            score_gap=None,
            selected_count=0,
            total_count=0,
            evidence_coverage=0.0,
            threshold=policy.low_score_threshold,
            reason=(
                "No documents passed retrieval."
            ),
        )

    scores = sorted(
        (
            _extract_score(
                document
            )
            for document in documents
        ),
        reverse=True,
    )

    top_score = scores[0]

    second_score = (
        scores[1]
        if len(scores) > 1
        else None
    )

    score_gap = (
        top_score - second_score
        if second_score is not None
        else None
    )

    selected_documents = (
        select_confident_documents(
            documents=documents,
            policy=policy,
        )
    )

    selected_count = len(
        selected_documents
    )

    if query is None:
        evidence_coverage = 1.0
    else:
        evidence_coverage = (
            calculate_evidence_coverage(
                query=query,
                documents=selected_documents,
            )
        )

    strong_semantic_override = (
        query is not None
        and top_score
        >= policy.semantic_override_threshold
        and evidence_coverage
        >= policy.semantic_override_minimum_evidence
        and (
            score_gap is None
            or score_gap
            >= policy.semantic_override_gap
        )
    )

    if (
        top_score
        < policy.low_score_threshold
        or selected_count == 0
    ):
        level = ConfidenceLevel.LOW

        reason = (
            "Semantic retrieval evidence is below the "
            "minimum confidence threshold."
        )

    elif (
        query is not None
        and evidence_coverage
        < policy.minimum_evidence_coverage
    ):
        if strong_semantic_override:
            level = ConfidenceLevel.MEDIUM

            reason = (
                "Very strong semantic retrieval with clear "
                "ranking separation compensates for limited "
                "direct lexical evidence."
            )

        else:
            level = ConfidenceLevel.LOW

            reason = (
                "Retrieved documents are semantically similar, "
                "but contain insufficient direct evidence for "
                "the current question."
            )

    elif (
        top_score
        >= policy.high_score_threshold
        and evidence_coverage
        >= policy.high_evidence_coverage
        and (
            score_gap is None
            or score_gap
            >= policy.strong_gap_threshold
        )
    ):
        level = ConfidenceLevel.HIGH

        reason = (
            "Strong semantic match, clear ranking separation, "
            "and sufficient query-context evidence."
        )

    elif (
        top_score
        >= policy.medium_score_threshold
        and evidence_coverage
        >= policy.minimum_evidence_coverage
    ):
        level = ConfidenceLevel.MEDIUM

        reason = (
            "Relevant context was found with sufficient "
            "evidence, but confidence is not strong enough "
            "for a high-confidence classification."
        )

    else:
        level = ConfidenceLevel.LOW

        reason = (
            "Retrieval evidence is too weak for grounded "
            "answer generation."
        )

    return ConfidenceResult(
        level=level,
        top_score=top_score,
        second_score=second_score,
        score_gap=score_gap,
        selected_count=selected_count,
        total_count=len(documents),
        evidence_coverage=evidence_coverage,
        threshold=policy.low_score_threshold,
        reason=reason,
    )