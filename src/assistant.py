import logging
from typing import TypedDict

from src.confidence import (
    DEFAULT_CONFIDENCE_POLICY,
    ConfidencePolicy,
    ConfidenceResult,
    evaluate_retrieval_confidence,
    select_confident_documents,
)
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
from src.query_rewriter import (
    QueryRewriteResult,
    rewrite_query,
)
from src.retrieval import (
    RetrievedDocument,
    get_top_documents,
)


logger = logging.getLogger(__name__)


class ConfidenceMetadata(TypedDict):
    """Serializable retrieval-confidence metadata."""

    level: str
    is_confident: bool

    top_score: float
    second_score: float | None
    score_gap: float | None

    evidence_coverage: float

    selected_count: int
    total_count: int
    filtered_count: int

    reason: str


class QueryRewriteMetadata(TypedDict):
    """Serializable conversational query-rewrite metadata."""

    original_query: str
    retrieval_query: str
    was_rewritten: bool
    reason: str


class AssistantResponse(TypedDict):
    """Structured response returned by the RAG assistant."""

    answer: str
    sources: list[str]

    retrieved_documents: list[
        RetrievedDocument
    ]

    confidence: ConfidenceMetadata
    query_rewrite: QueryRewriteMetadata


def _build_confidence_metadata(
    confidence: ConfidenceResult,
) -> ConfidenceMetadata:
    """
    Convert internal confidence state into response metadata.
    """

    return {
        "level": confidence.level.value,
        "is_confident": confidence.is_confident,
        "top_score": confidence.top_score,
        "second_score": confidence.second_score,
        "score_gap": confidence.score_gap,
        "evidence_coverage": confidence.evidence_coverage,
        "selected_count": confidence.selected_count,
        "total_count": confidence.total_count,
        "filtered_count": confidence.filtered_count,
        "reason": confidence.reason,
    }


def _build_query_rewrite_metadata(
    result: QueryRewriteResult,
) -> QueryRewriteMetadata:
    """
    Convert query-rewrite result into response metadata.
    """

    return {
        "original_query": result.original_query,
        "retrieval_query": result.rewritten_query,
        "was_rewritten": result.was_rewritten,
        "reason": result.reason,
    }


def _normalize_answer_text(
    answer: str,
) -> str:
    """
    Normalize generated text for reliable comparison.

    Only whitespace is normalized. Semantic content is not altered.
    """

    return " ".join(
        answer.strip().split()
    )


def _is_fallback_answer(
    answer: str,
) -> bool:
    """
    Determine whether an LLM response represents the configured
    document-not-found fallback.

    The local model may occasionally prepend the question or add
    whitespace before returning the fallback. Therefore the canonical
    fallback is detected inside normalized generated text.

    Example:

        SQLite neden ...?

        Bu bilgi mevcut yerel belgelerde bulunamadı.

    is still considered a fallback response.
    """

    normalized_answer = (
        _normalize_answer_text(
            answer
        )
    )

    normalized_fallback = (
        _normalize_answer_text(
            DEFAULT_FALLBACK_MESSAGE
        )
    )

    return (
        normalized_fallback
        in normalized_answer
    )


class RAGAssistant:
    """
    Orchestrate the local conversational confidence-aware RAG pipeline.

    Responsibilities:
        - latest-user-query context tracking
        - conversational query rewriting
        - semantic retrieval
        - confidence evaluation
        - evidence coverage validation
        - adaptive context selection
        - grounded prompt construction
        - local generation
        - generated-fallback detection
        - successful conversation memory
        - trusted source attribution
    """

    def __init__(
        self,
        local_llm: LocalLLM | None = None,
        prompt_builder: PromptBuilder | None = None,
        memory: ConversationMemory | None = None,
        confidence_policy: ConfidencePolicy | None = None,
        top_k: int = TOP_K,
        minimum_score: float = MIN_SIMILARITY_SCORE,
    ) -> None:
        """Initialize the RAG assistant."""

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

        self._confidence_policy = (
            confidence_policy
            if confidence_policy is not None
            else DEFAULT_CONFIDENCE_POLICY
        )

        self._top_k = top_k
        self._minimum_score = minimum_score

    @property
    def memory_size(self) -> int:
        """Return number of successful conversation turns."""

        return self._memory.size

    @property
    def confidence_policy(
        self,
    ) -> ConfidencePolicy:
        """Return active confidence policy."""

        return self._confidence_policy

    def warm_up(
        self,
    ) -> None:
        """Prepare the local language model."""

        self._local_llm.warm_up()

    def answer(
        self,
        question: str,
    ) -> AssistantResponse:
        """
        Answer a question using the conversational local RAG pipeline.

        Important memory behavior:

        - Every valid user question becomes latest-query context.
        - Only successful grounded answers become conversation history.
        - Confidence-level fallbacks never become factual history.
        - LLM-generated fallback responses never become factual history.
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

        # ==================================================
        # Successful conversation history
        # ==================================================

        conversation_history = (
            self._memory.build_history_text()
        )

        # ==================================================
        # Previous user-query context
        #
        # Read this BEFORE storing the current question.
        # ==================================================

        rewrite_context = (
            self._memory.build_rewrite_context()
        )

        # ==================================================
        # Conversational query rewriting
        # ==================================================

        rewrite_result = rewrite_query(
            query=clean_question,
            conversation_history=rewrite_context,
        )

        retrieval_query = (
            rewrite_result.rewritten_query
        )

        query_rewrite_metadata = (
            _build_query_rewrite_metadata(
                rewrite_result
            )
        )

        if rewrite_result.was_rewritten:
            logger.info(
                (
                    "Conversational query rewritten | "
                    "Original: %s | Retrieval: %s"
                ),
                rewrite_result.original_query,
                rewrite_result.rewritten_query,
            )

        else:
            logger.debug(
                (
                    "Query rewrite uygulanmadı | "
                    "Query: %s | Reason: %s"
                ),
                rewrite_result.original_query,
                rewrite_result.reason,
            )

        # ==================================================
        # Record current user query
        #
        # Even unanswered queries remain available to the NEXT
        # follow-up question.
        # ==================================================

        self._memory.record_user_query(
            clean_question
        )

        # ==================================================
        # Semantic retrieval
        # ==================================================

        retrieved_documents = (
            get_top_documents(
                query=retrieval_query,
                top_k=self._top_k,
                minimum_score=self._minimum_score,
            )
        )

        # ==================================================
        # Confidence evaluation
        #
        # Evidence coverage uses the ORIGINAL current question.
        # The rewritten query is retrieval-only.
        # ==================================================

        confidence = (
            evaluate_retrieval_confidence(
                documents=retrieved_documents,
                query=clean_question,
                policy=self._confidence_policy,
            )
        )

        confidence_metadata = (
            _build_confidence_metadata(
                confidence
            )
        )

        logger.info(
            (
                "Retrieval confidence | "
                "Level: %s | "
                "Top score: %.4f | "
                "Evidence: %.2f%% | "
                "Selected: %d/%d | "
                "Filtered: %d"
            ),
            confidence.level.value.upper(),
            confidence.top_score,
            confidence.evidence_coverage * 100,
            confidence.selected_count,
            confidence.total_count,
            confidence.filtered_count,
        )

        # ==================================================
        # Confidence-level fallback
        # ==================================================

        if not confidence.is_confident:
            logger.warning(
                (
                    "Retrieval güveni yetersiz. "
                    "LLM çağrısı yapılmayacak. "
                    "Sebep: %s"
                ),
                confidence.reason,
            )

            return {
                "answer": DEFAULT_FALLBACK_MESSAGE,
                "sources": [],
                "retrieved_documents": retrieved_documents,
                "confidence": confidence_metadata,
                "query_rewrite": query_rewrite_metadata,
            }

        # ==================================================
        # Adaptive context filtering
        # ==================================================

        selected_documents = (
            select_confident_documents(
                documents=retrieved_documents,
                policy=self._confidence_policy,
            )
        )

        if not selected_documents:
            logger.warning(
                (
                    "Confidence filtresinden sonra "
                    "kullanılabilir bağlam kalmadı."
                )
            )

            return {
                "answer": DEFAULT_FALLBACK_MESSAGE,
                "sources": [],
                "retrieved_documents": retrieved_documents,
                "confidence": confidence_metadata,
                "query_rewrite": query_rewrite_metadata,
            }

        logger.info(
            (
                "%d/%d retrieval sonucu "
                "LLM bağlamı olarak seçildi."
            ),
            len(selected_documents),
            len(retrieved_documents),
        )

        # ==================================================
        # Grounded prompt
        #
        # LLM receives the ORIGINAL user question.
        # ==================================================

        system_prompt, user_prompt = (
            self._prompt_builder.build(
                question=clean_question,
                retrieved_documents=selected_documents,
                conversation_history=conversation_history,
            )
        )

        # ==================================================
        # Local generation
        # ==================================================

        generated_answer = (
            self._local_llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
        )

        if not isinstance(
            generated_answer,
            str,
        ):
            raise TypeError(
                "Local LLM response must be a string."
            )

        clean_generated_answer = (
            generated_answer.strip()
        )

        if not clean_generated_answer:
            raise ValueError(
                "Local LLM returned an empty response."
            )

        # ==================================================
        # LLM-level fallback detection
        #
        # Confidence may allow generation while the LLM still
        # determines that the retrieved context is insufficient.
        #
        # In that case:
        # - canonical fallback is returned,
        # - no source is exposed,
        # - no successful conversation turn is stored.
        # ==================================================

        if _is_fallback_answer(
            clean_generated_answer
        ):
            logger.info(
                (
                    "Yerel model belge bağlamını yetersiz "
                    "buldu ve fallback üretti. "
                    "Kaynak ve başarılı konuşma kaydı "
                    "oluşturulmayacak."
                )
            )

            return {
                "answer": DEFAULT_FALLBACK_MESSAGE,
                "sources": [],
                "retrieved_documents": retrieved_documents,
                "confidence": confidence_metadata,
                "query_rewrite": query_rewrite_metadata,
            }

        # ==================================================
        # Successful conversation history
        # ==================================================

        self._memory.add_turn(
            question=clean_question,
            answer=clean_generated_answer,
        )

        # ==================================================
        # Trusted source attribution
        # ==================================================

        sources = list(
            dict.fromkeys(
                document["source"]
                for document
                in selected_documents
            )
        )

        return {
            "answer": clean_generated_answer,
            "sources": sources,
            "retrieved_documents": retrieved_documents,
            "confidence": confidence_metadata,
            "query_rewrite": query_rewrite_metadata,
        }

    def clear_memory(
        self,
    ) -> None:
        """
        Clear successful history and latest user-query context.
        """

        self._memory.clear()

        logger.info(
            "Konuşma hafızası temizlendi."
        )

    def get_conversation_history(
        self,
    ) -> str:
        """Return formatted successful conversation history."""

        return (
            self._memory.build_history_text()
        )

    def close(
        self,
    ) -> None:
        """Release the local chat model."""

        self._local_llm.unload()

    def __enter__(
        self,
    ) -> "RAGAssistant":
        """Enter context-manager session."""

        return self

    def __exit__(
        self,
        exception_type: object,
        exception_value: object,
        traceback: object,
    ) -> None:
        """Release resources."""

        self.close()