from collections.abc import Callable

from src.assistant import RAGAssistant
from src.file_sync import (
    SyncResult,
    synchronize_knowledge_base,
)


EXIT_COMMANDS = {
    "exit",
    "quit",
    "q",
    "çık",
    "çıkış",
}

CLEAR_COMMANDS = {
    "/clear",
    "/temizle",
}

HISTORY_COMMANDS = {
    "/history",
    "/geçmiş",
}

SYNC_COMMANDS = {
    "/sync",
    "/senkronize",
}


def format_sync_result(
    result: SyncResult,
) -> list[str]:
    """Format a Smart Folder Sync result."""

    lines = [
        "",
        "=" * 50,
        "KNOWLEDGE BASE SYNC",
        "=" * 50,
        f"New files      : {len(result.new_files)}",
        f"Modified files : {len(result.modified_files)}",
        f"Deleted files  : {len(result.deleted_files)}",
        f"Unchanged      : {len(result.unchanged_files)}",
        f"Inserted chunks: {result.inserted_chunks}",
        f"Deleted chunks : {result.deleted_chunks}",
    ]

    if result.new_files:
        lines.append(
            "\nNew:"
        )

        lines.extend(
            f"  + {source}"
            for source in result.new_files
        )

    if result.modified_files:
        lines.append(
            "\nModified:"
        )

        lines.extend(
            f"  ~ {source}"
            for source in result.modified_files
        )

    if result.deleted_files:
        lines.append(
            "\nDeleted:"
        )

        lines.extend(
            f"  - {source}"
            for source in result.deleted_files
        )

    if result.unchanged_files:
        lines.append(
            "\nUnchanged:"
        )

        lines.extend(
            f"  = {source}"
            for source in result.unchanged_files
        )

    if not result.has_changes:
        lines.append(
            "\nKnowledge base is already up to date."
        )

    return lines


def _format_query_rewrite(
    query_rewrite: dict,
) -> list[str]:
    """
    Format conversational query-rewrite metadata.
    """

    if not query_rewrite:
        return []

    original_query = str(
        query_rewrite.get(
            "original_query",
            "",
        )
    )

    retrieval_query = str(
        query_rewrite.get(
            "retrieval_query",
            original_query,
        )
    )

    was_rewritten = bool(
        query_rewrite.get(
            "was_rewritten",
            False,
        )
    )

    reason = str(
        query_rewrite.get(
            "reason",
            "",
        )
    )

    lines = [
        "",
        "=" * 50,
        "QUERY REWRITE",
        "=" * 50,
        (
            "Rewritten       : "
            f"{'YES' if was_rewritten else 'NO'}"
        ),
        f"Original Query  : {original_query}",
        f"Retrieval Query : {retrieval_query}",
    ]

    if reason:
        lines.append(
            f"Reason          : {reason}"
        )

    return lines


def _format_confidence(
    confidence: dict,
) -> list[str]:
    """
    Format retrieval-confidence metadata.
    """

    if not confidence:
        return []

    level = str(
        confidence.get(
            "level",
            "unknown",
        )
    ).upper()

    top_score = float(
        confidence.get(
            "top_score",
            0.0,
        )
    )

    second_score = confidence.get(
        "second_score"
    )

    score_gap = confidence.get(
        "score_gap"
    )

    evidence_coverage = float(
        confidence.get(
            "evidence_coverage",
            0.0,
        )
    )

    selected_count = int(
        confidence.get(
            "selected_count",
            0,
        )
    )

    total_count = int(
        confidence.get(
            "total_count",
            0,
        )
    )

    filtered_count = int(
        confidence.get(
            "filtered_count",
            0,
        )
    )

    reason = str(
        confidence.get(
            "reason",
            "",
        )
    )

    lines = [
        "",
        "=" * 50,
        "RETRIEVAL CONFIDENCE",
        "=" * 50,
        f"Level            : {level}",
        f"Top Score        : {top_score:.4f}",
    ]

    if second_score is None:
        lines.append(
            "Second Score     : N/A"
        )
    else:
        lines.append(
            f"Second Score     : "
            f"{float(second_score):.4f}"
        )

    if score_gap is None:
        lines.append(
            "Score Gap        : N/A"
        )
    else:
        lines.append(
            f"Score Gap        : "
            f"{float(score_gap):.4f}"
        )

    lines.extend(
        [
            (
                "Evidence Coverage: "
                f"{evidence_coverage:.2%}"
            ),
            (
                "Selected Context  : "
                f"{selected_count}/{total_count}"
            ),
            (
                "Filtered Noise    : "
                f"{filtered_count}"
            ),
        ]
    )

    if reason:
        lines.append(
            f"Reason           : {reason}"
        )

    return lines


def _format_retrieved_documents(
    retrieved_documents: list[dict],
) -> list[str]:
    """
    Format retrieval metadata.
    """

    if not retrieved_documents:
        return [
            "Retrieved document bulunamadı."
        ]

    lines: list[str] = [
        "",
        "=" * 50,
        "RETRIEVED DOCUMENTS",
        "=" * 50,
    ]

    for index, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        source = document.get(
            "source",
            "Unknown",
        )

        score = document.get(
            "score",
            0.0,
        )

        lines.extend(
            [
                f"[{index}]",
                f"Source : {source}",
                f"Score  : {float(score):.4f}",
                "",
            ]
        )

    return lines


def run_chat_session(
    assistant: RAGAssistant,
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], None] = print,
) -> None:
    """
    Run the interactive terminal assistant.
    """

    output_function(
        "Local RAG AI Assistant"
    )

    output_function(
        "-" * 50
    )

    output_function(
        "Çıkmak için 'exit', 'quit', 'q' veya 'çıkış' yaz."
    )

    output_function(
        "Komutlar: /clear, /history, /sync"
    )

    while True:
        try:
            question = input_function(
                "\nSoru: "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
            StopIteration,
        ):
            output_function(
                "\nSohbet sonlandırıldı."
            )
            break

        if not question:
            output_function(
                "Lütfen boş olmayan bir soru gir."
            )
            continue

        normalized_question = (
            question.lower()
        )

        if normalized_question in EXIT_COMMANDS:
            output_function(
                "Sohbet sonlandırıldı."
            )
            break

        if normalized_question in CLEAR_COMMANDS:
            assistant.clear_memory()

            output_function(
                "Konuşma hafızası temizlendi."
            )

            continue

        if normalized_question in HISTORY_COMMANDS:
            history = (
                assistant.get_conversation_history()
            )

            if not history:
                output_function(
                    "Konuşma geçmişi boş."
                )

            else:
                output_function(
                    "\n" + "=" * 50
                )

                output_function(
                    "KONUŞMA GEÇMİŞİ"
                )

                output_function(
                    "=" * 50
                )

                output_function(
                    history
                )

            continue

        if normalized_question in SYNC_COMMANDS:
            try:
                sync_result = (
                    synchronize_knowledge_base()
                )

            except Exception as error:
                output_function(
                    f"Senkronizasyon hatası: {error}"
                )

                continue

            for line in format_sync_result(
                sync_result
            ):
                output_function(
                    line
                )

            continue

        try:
            response = assistant.answer(
                question
            )

        except Exception as error:
            output_function(
                f"Bir hata oluştu: {error}"
            )

            continue

        # ==================================================
        # Answer
        # ==================================================

        output_function(
            "\n" + "=" * 50
        )

        output_function(
            "RAG CEVABI"
        )

        output_function(
            "=" * 50
        )

        output_function(
            response["answer"]
        )

        # ==================================================
        # Trusted sources
        # ==================================================

        sources = response.get(
            "sources",
            [],
        )

        if sources:
            output_function(
                "\nKaynaklar:"
            )

            for source in sources:
                output_function(
                    f"- {source}"
                )

        # ==================================================
        # Conversational query rewrite
        # ==================================================

        query_rewrite = response.get(
            "query_rewrite",
            {},
        )

        for line in _format_query_rewrite(
            query_rewrite
        ):
            output_function(
                line
            )

        # ==================================================
        # Retrieval confidence
        # ==================================================

        confidence = response.get(
            "confidence",
            {},
        )

        for line in _format_confidence(
            confidence
        ):
            output_function(
                line
            )

        # ==================================================
        # Retrieval diagnostics
        # ==================================================

        retrieved_documents = response.get(
            "retrieved_documents",
            [],
        )

        for line in _format_retrieved_documents(
            retrieved_documents
        ):
            output_function(
                line
            )