import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.staticfiles import StaticFiles
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from src.agent import LocalAgent
from src.agent_models import AgentResult
from src.agent_tools import ToolRegistry
from src.assistant import RAGAssistant
from src.database import (
    get_unique_document_sources,
)
from src.file_sync import (
    synchronize_knowledge_base,
)
from src.ingestion import (
    ingest_selected_source,
)
from src.knowledge_tools import (
    KnowledgeSearchTool,
    KnowledgeStatusTool,
    KnowledgeSyncTool,
)
from src.web import (
    WEB_DIRECTORY,
    get_index_file,
)


API_TITLE = "Local RAG AI Assistant API"
API_VERSION = "1.3.0"

MAX_UPLOAD_SIZE_BYTES = (
    5 * 1024 * 1024
)

SUPPORTED_UPLOAD_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
}


# ==========================================================
# API MODELS
# ==========================================================


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    question: str = Field(
        min_length=1,
        max_length=2000,
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: dict[str, Any]
    query_rewrite: dict[str, Any]
    retrieved_documents: list[
        dict[str, Any]
    ]


class AgentRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    message: str = Field(
        min_length=1,
        max_length=2000,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class AgentDecisionResponse(BaseModel):
    intent: str
    tool_name: str | None
    reason: str
    confidence: float


class AgentStepResponse(BaseModel):
    name: str
    status: str
    detail: str
    tool_name: str | None


class AgentToolResultResponse(BaseModel):
    success: bool
    content: str
    data: dict[str, Any]
    error: str | None


class AgentResponse(BaseModel):
    answer: str
    succeeded: bool
    decision: AgentDecisionResponse
    steps: list[
        AgentStepResponse
    ]
    tool_result: (
        AgentToolResultResponse
        | None
    )
    metadata: dict[str, Any]


class HistoryResponse(BaseModel):
    history: str
    is_empty: bool


class MessageResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    assistant_ready: bool
    agent_ready: bool
    local_only: bool


class SyncResponse(BaseModel):
    new_files: list[str]
    modified_files: list[str]
    deleted_files: list[str]
    unchanged_files: list[str]
    inserted_chunks: int
    deleted_chunks: int
    has_changes: bool


class KnowledgeImportResponse(BaseModel):
    file_count: int
    inserted_chunks: int
    sources: list[str]
    message: str


class KnowledgeSourcesResponse(BaseModel):
    source_count: int
    sources: list[str]


# ==========================================================
# APPLICATION STATE
# ==========================================================


_assistant: RAGAssistant | None = None
_agent: LocalAgent | None = None


def get_assistant() -> RAGAssistant:
    if _assistant is None:
        raise RuntimeError(
            "RAG assistant is not initialized."
        )

    return _assistant


def get_agent() -> LocalAgent:
    if _agent is None:
        raise RuntimeError(
            "Local agent is not initialized."
        )

    return _agent


def set_assistant_for_testing(
    assistant: RAGAssistant | None,
) -> None:
    global _assistant

    _assistant = assistant


def set_agent_for_testing(
    agent: LocalAgent | None,
) -> None:
    global _agent

    _agent = agent


# ==========================================================
# APPLICATION LIFESPAN
# ==========================================================


@asynccontextmanager
async def lifespan(
    application: FastAPI,
):
    del application

    global _assistant
    global _agent

    assistant = RAGAssistant()

    try:
        assistant.warm_up()

        registry = ToolRegistry(
            tools=[
                KnowledgeSearchTool(
                    assistant
                ),
                KnowledgeStatusTool(),
                KnowledgeSyncTool(),
            ]
        )

        agent = LocalAgent(
            registry=registry
        )

        _assistant = assistant
        _agent = agent

        yield

    finally:
        _agent = None
        _assistant = None

        assistant.close()


# ==========================================================
# FASTAPI APPLICATION
# ==========================================================


app = FastAPI(
    title=API_TITLE,
    description=(
        "Fully local, confidence-aware RAG and "
        "tool-using agent API powered by "
        "Microsoft Foundry Local."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


# ==========================================================
# STATIC WEB APPLICATION
# ==========================================================


app.mount(
    "/static",
    StaticFiles(
        directory=str(
            WEB_DIRECTORY
        ),
    ),
    name="static",
)


# ==========================================================
# WEB INTERFACE
# ==========================================================


@app.get(
    "/",
    include_in_schema=False,
)
def web_interface():
    return get_index_file()


# ==========================================================
# SYSTEM
# ==========================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=[
        "System",
    ],
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        assistant_ready=(
            _assistant is not None
        ),
        agent_ready=(
            _agent is not None
        ),
        local_only=True,
    )


# ==========================================================
# RAG
# ==========================================================


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=[
        "RAG",
    ],
)
def chat(
    request: ChatRequest,
) -> ChatResponse:
    try:
        assistant = get_assistant()

    except RuntimeError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Local RAG assistant is not ready."
            ),
        ) from error

    try:
        response = assistant.answer(
            request.question
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(
                error
            ),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "The local RAG assistant "
                "could not process the request."
            ),
        ) from error

    return ChatResponse(
        answer=response[
            "answer"
        ],
        sources=list(
            response[
                "sources"
            ]
        ),
        confidence=dict(
            response[
                "confidence"
            ]
        ),
        query_rewrite=dict(
            response[
                "query_rewrite"
            ]
        ),
        retrieved_documents=[
            dict(document)
            for document
            in response[
                "retrieved_documents"
            ]
        ],
    )


# ==========================================================
# AGENT
# ==========================================================


@app.post(
    "/agent",
    response_model=AgentResponse,
    tags=[
        "Agent",
    ],
)
def run_agent(
    request: AgentRequest,
) -> AgentResponse:
    try:
        agent = get_agent()

    except RuntimeError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Local agent is not ready."
            ),
        ) from error

    try:
        result = agent.run(
            user_input=request.message,
            metadata=request.metadata,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(
                error
            ),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "The local agent could not "
                "process the request."
            ),
        ) from error

    return _serialize_agent_result(
        result
    )


def _serialize_agent_result(
    result: AgentResult,
) -> AgentResponse:
    tool_result = None

    if result.tool_result is not None:
        tool_result = (
            AgentToolResultResponse(
                success=(
                    result.tool_result.success
                ),
                content=(
                    result.tool_result.content
                ),
                data=dict(
                    result.tool_result.data
                ),
                error=(
                    result.tool_result.error
                ),
            )
        )

    return AgentResponse(
        answer=result.answer,
        succeeded=result.succeeded,
        decision=AgentDecisionResponse(
            intent=(
                result.decision.intent.value
            ),
            tool_name=(
                result.decision.tool_name
            ),
            reason=(
                result.decision.reason
            ),
            confidence=(
                result.decision.confidence
            ),
        ),
        steps=[
            AgentStepResponse(
                name=step.name,
                status=step.status.value,
                detail=step.detail,
                tool_name=step.tool_name,
            )
            for step in result.steps
        ],
        tool_result=tool_result,
        metadata=dict(
            result.metadata
        ),
    )


# ==========================================================
# CONVERSATION
# ==========================================================


@app.get(
    "/history",
    response_model=HistoryResponse,
    tags=[
        "Conversation",
    ],
)
def history() -> HistoryResponse:
    try:
        assistant = get_assistant()

    except RuntimeError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Local RAG assistant is not ready."
            ),
        ) from error

    try:
        conversation_history = (
            assistant
            .get_conversation_history()
        )

    except Exception as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Conversation history "
                "could not be retrieved."
            ),
        ) from error

    return HistoryResponse(
        history=conversation_history,
        is_empty=(
            not conversation_history
        ),
    )


@app.delete(
    "/history",
    response_model=MessageResponse,
    tags=[
        "Conversation",
    ],
)
def clear_history() -> MessageResponse:
    try:
        assistant = get_assistant()

    except RuntimeError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Local RAG assistant is not ready."
            ),
        ) from error

    try:
        assistant.clear_memory()

    except Exception as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Conversation history "
                "could not be cleared."
            ),
        ) from error

    return MessageResponse(
        message=(
            "Conversation history cleared."
        )
    )


# ==========================================================
# KNOWLEDGE BASE SOURCES
# ==========================================================


@app.get(
    "/knowledge/sources",
    response_model=KnowledgeSourcesResponse,
    tags=[
        "Knowledge Base",
    ],
)
def knowledge_sources() -> KnowledgeSourcesResponse:
    try:
        sources = (
            get_unique_document_sources()
        )

    except Exception as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Knowledge-base sources "
                "could not be retrieved."
            ),
        ) from error

    return KnowledgeSourcesResponse(
        source_count=len(
            sources
        ),
        sources=list(
            sources
        ),
    )


# ==========================================================
# KNOWLEDGE BASE SYNC
# ==========================================================


@app.post(
    "/sync",
    response_model=SyncResponse,
    tags=[
        "Knowledge Base",
    ],
)
def sync_knowledge_base() -> SyncResponse:
    try:
        result = (
            synchronize_knowledge_base()
        )

    except Exception as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Knowledge base synchronization failed."
            ),
        ) from error

    return SyncResponse(
        new_files=list(
            result.new_files
        ),
        modified_files=list(
            result.modified_files
        ),
        deleted_files=list(
            result.deleted_files
        ),
        unchanged_files=list(
            result.unchanged_files
        ),
        inserted_chunks=(
            result.inserted_chunks
        ),
        deleted_chunks=(
            result.deleted_chunks
        ),
        has_changes=(
            result.has_changes
        ),
    )


# ==========================================================
# EXTERNAL KNOWLEDGE FILE UPLOAD
# ==========================================================


@app.post(
    "/knowledge/files",
    response_model=KnowledgeImportResponse,
    status_code=(
        status.HTTP_201_CREATED
    ),
    tags=[
        "Knowledge Base",
    ],
)
async def import_knowledge_file(
    file: UploadFile = File(...),
) -> KnowledgeImportResponse:
    original_name = Path(
        file.filename or ""
    ).name.strip()

    if not original_name:
        await file.close()

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                "A valid file name is required."
            ),
        )

    original_suffix = (
        Path(original_name)
        .suffix
        .lower()
    )

    if (
        original_suffix
        not in SUPPORTED_UPLOAD_EXTENSIONS
    ):
        await file.close()

        raise HTTPException(
            status_code=(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            ),
            detail=(
                "Supported document types are "
                "TXT, PDF, and DOCX."
            ),
        )

    temporary_path: Path | None = None

    try:
        content = await file.read(
            MAX_UPLOAD_SIZE_BYTES
            + 1
        )

        if not content:
            raise HTTPException(
                status_code=(
                    status.HTTP_422_UNPROCESSABLE_ENTITY
                ),
                detail=(
                    "The selected document is empty."
                ),
            )

        if (
            len(content)
            > MAX_UPLOAD_SIZE_BYTES
        ):
            raise HTTPException(
                status_code=413,
                detail=(
                    "The selected document exceeds "
                    "the 5 MB upload limit."
                ),
            )

        if original_suffix == ".txt":
            try:
                content.decode(
                    "utf-8"
                )

            except UnicodeDecodeError as error:
                raise HTTPException(
                    status_code=(
                        status.HTTP_422_UNPROCESSABLE_ENTITY
                    ),
                    detail=(
                        "TXT documents must contain "
                        "valid UTF-8 text."
                    ),
                ) from error

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=original_suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(
                content
            )

            temporary_path = Path(
                temporary_file.name
            )

        result = ingest_selected_source(
            source_path=temporary_path,
            external_source_name=original_name,
        )

    except HTTPException:
        raise

    except (
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        if original_suffix == ".pdf":
            safe_detail = (
                "The PDF document could not be read "
                "or contains no extractable text."
            )

        elif original_suffix == ".docx":
            safe_detail = (
                "The DOCX document could not be read "
                "or contains no extractable text."
            )

        else:
            safe_detail = (
                "The selected document could not be processed."
            )

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=safe_detail,
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "The selected document could not be "
                "added to the local knowledge base."
            ),
        ) from error

    finally:
        await file.close()

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            try:
                temporary_path.unlink()

            except OSError:
                pass

    return KnowledgeImportResponse(
        file_count=result[
            "file_count"
        ],
        inserted_chunks=result[
            "inserted_chunks"
        ],
        sources=list(
            result[
                "sources"
            ]
        ),
        message=(
            f"{original_name} was added to "
            "the local knowledge base."
        ),
    )