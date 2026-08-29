from __future__ import annotations

from typing import Any

from src.agent_models import ToolResult
from src.agent_tools import (
    AgentTool,
    ToolContext,
)
from src.assistant import RAGAssistant

from src.database import get_all_documents
from src.file_sync import synchronize_knowledge_base


class KnowledgeSearchTool(AgentTool):
    """
    Agent adapter for the existing RAG assistant.

    This tool does not implement retrieval itself.
    It delegates knowledge-grounded answering to the
    existing RAGAssistant so retrieval, confidence,
    memory and generation logic remain centralized.
    """

    def __init__(
        self,
        assistant: RAGAssistant,
    ) -> None:
        if not isinstance(
            assistant,
            RAGAssistant,
        ):
            raise TypeError(
                "assistant must be a RAGAssistant."
            )

        self._assistant = assistant

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return (
            "Search the indexed local knowledge base "
            "and generate an evidence-grounded answer."
        )

    def execute(
        self,
        context: ToolContext,
    ) -> ToolResult:
        """
        Answer a knowledge question using the existing
        local RAG pipeline.
        """

        response = self._assistant.answer(
            context.user_input
        )

        answer = response.get(
            "answer"
        )

        if not isinstance(
            answer,
            str,
        ):
            return ToolResult(
                success=False,
                content=(
                    "Knowledge search returned "
                    "an invalid response."
                ),
                error=(
                    "RAG response does not contain "
                    "a valid answer."
                ),
            )

        clean_answer = answer.strip()

        if not clean_answer:
            return ToolResult(
                success=False,
                content=(
                    "Knowledge search returned "
                    "an empty answer."
                ),
                error=(
                    "RAG response answer is empty."
                ),
            )

        data = self._build_result_data(
            response
        )

        return ToolResult(
            success=True,
            content=clean_answer,
            data=data,
        )

    @staticmethod
    def _build_result_data(
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Preserve RAG observability metadata.

        Agent consumers can inspect the same sources,
        confidence information, query rewrite metadata
        and retrieved documents used by the existing
        API and web workspace.
        """

        sources = response.get(
            "sources",
            [],
        )

        confidence = response.get(
            "confidence",
            {},
        )

        query_rewrite = response.get(
            "query_rewrite",
            {},
        )

        retrieved_documents = response.get(
            "retrieved_documents",
            [],
        )

        return {
            "sources": (
                list(sources)
                if isinstance(
                    sources,
                    (list, tuple),
                )
                else []
            ),
            "confidence": (
                dict(confidence)
                if isinstance(
                    confidence,
                    dict,
                )
                else {}
            ),
            "query_rewrite": (
                dict(query_rewrite)
                if isinstance(
                    query_rewrite,
                    dict,
                )
                else {}
            ),
            "retrieved_documents": (
                list(retrieved_documents)
                if isinstance(
                    retrieved_documents,
                    (list, tuple),
                )
                else []
            ),
        }
        
        
class KnowledgeStatusTool(AgentTool):
    """
    Inspect the current local knowledge base.

    This tool performs a deterministic database
    inspection and does not invoke the local LLM.
    """

    @property
    def name(self) -> str:
        return "knowledge_status"

    @property
    def description(self) -> str:
        return (
            "Inspect the indexed local knowledge base "
            "and report document and source statistics."
        )

    def execute(
        self,
        context: ToolContext,
    ) -> ToolResult:
        del context

        documents = get_all_documents()

        unique_sources = sorted(
            {
                str(document["source"])
                for document in documents
                if document.get("source")
            }
        )

        document_count = len(documents)
        source_count = len(unique_sources)

        if document_count == 0:
            content = (
                "The local knowledge base is empty."
            )
        else:
            content = (
                "The local knowledge base contains "
                f"{document_count} indexed document chunks "
                f"from {source_count} unique source files."
            )

        return ToolResult(
            success=True,
            content=content,
            data={
                "document_count": document_count,
                "source_count": source_count,
                "sources": unique_sources,
                "local_only": True,
            },
        )


class KnowledgeSyncTool(AgentTool):
    """
    Synchronize the configured local knowledge base.

    The implementation delegates to the existing
    Smart Folder Sync pipeline instead of duplicating
    ingestion or deletion logic.
    """

    @property
    def name(self) -> str:
        return "knowledge_sync"

    @property
    def description(self) -> str:
        return (
            "Synchronize the configured local knowledge "
            "base with its source folder."
        )

    def execute(
        self,
        context: ToolContext,
    ) -> ToolResult:
        del context

        result = synchronize_knowledge_base()

        new_files = list(
            result.new_files
        )

        modified_files = list(
            result.modified_files
        )

        deleted_files = list(
            result.deleted_files
        )

        unchanged_files = list(
            result.unchanged_files
        )

        if result.has_changes:
            content = (
                "Knowledge base synchronization completed. "
                f"{len(new_files)} new, "
                f"{len(modified_files)} modified, "
                f"{len(deleted_files)} deleted files were detected."
            )
        else:
            content = (
                "Knowledge base is already up to date."
            )

        return ToolResult(
            success=True,
            content=content,
            data={
                "new_files": new_files,
                "modified_files": modified_files,
                "deleted_files": deleted_files,
                "unchanged_files": unchanged_files,
                "inserted_chunks": (
                    result.inserted_chunks
                ),
                "deleted_chunks": (
                    result.deleted_chunks
                ),
                "has_changes": (
                    result.has_changes
                ),
                "local_only": True,
            },
        )