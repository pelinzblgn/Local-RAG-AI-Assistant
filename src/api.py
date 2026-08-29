from contextlib import asynccontextmanager
from typing import Any

from fastapi import (
    FastAPI,
    HTTPException,
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
from src.file_sync import (
    synchronize_knowledge_base,
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
API_VERSION = "1.1.0"


# ==========================================================
# API MODELS
# ==========================================================


class ChatRequest(BaseModel):
    """Question payload accepted by the RAG API."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    question: str = Field(
        min_length=1,
        max_length=2000,
    )


class ChatResponse(BaseModel):
    """
    Public response returned by the local RAG assistant.
    """

    answer: str

    sources: list[str]

    confidence: dict[str, Any]

    query_rewrite: dict[str, Any]

    retrieved_documents: list[
        dict[str, Any]
    ]


class AgentRequest(BaseModel):
    """
    User request accepted by the local agent.
    """

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
    """Public representation of an agent decision."""

    intent: str

    tool_name: str | None

    reason: str

    confidence: float


class AgentStepResponse(BaseModel):
    """One observable agent execution step."""

    name: str

    status: str

    detail: str

    tool_name: str | None


class AgentToolResultResponse(BaseModel):
    """Public representation of a tool result."""

    success: bool

    content: str

    data: dict[str, Any]

    error: str | None


class AgentResponse(BaseModel):
    """
    Observable result returned by the local agent.
    """

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
    """
    Current successful conversation history.
    """

    history: str

    is_empty: bool


class MessageResponse(BaseModel):
    """Generic API message response."""

    message: str


class HealthResponse(BaseModel):
    """
    Application and local-runtime health information.
    """

    status: str

    assistant_ready: bool

    agent_ready: bool

    local_only: bool


class SyncResponse(BaseModel):
    """
    Knowledge-base synchronization result.
    """

    new_files: list[str]

    modified_files: list[str]

    deleted_files: list[str]

    unchanged_files: list[str]

    inserted_chunks: int

    deleted_chunks: int

    has_changes: bool


# ==========================================================
# APPLICATION STATE
# ==========================================================


_assistant: RAGAssistant | None = None
_agent: LocalAgent | None = None


def get_assistant() -> RAGAssistant:
    """
    Return the application-level RAG assistant.
    """

    if _assistant is None:
        raise RuntimeError(
            "RAG assistant is not initialized."
        )

    return _assistant


def get_agent() -> LocalAgent:
    """
    Return the application-level local agent.
    """

    if _agent is None:
        raise RuntimeError(
            "Local agent is not initialized."
        )

    return _agent


def set_assistant_for_testing(
    assistant: RAGAssistant | None,
) -> None:
    """
    Override the application-level assistant.

    This exists for isolated API tests.
    """

    global _assistant

    _assistant = assistant


def set_agent_for_testing(
    agent: LocalAgent | None,
) -> None:
    """
    Override the application-level agent.

    This exists for isolated API tests.
    """

    global _agent

    _agent = agent


# ==========================================================
# APPLICATION LIFESPAN
# ==========================================================


@asynccontextmanager
async def lifespan(
    application: FastAPI,
):
    """
    Manage application-level local AI resources.

    One RAGAssistant instance is created and shared
    between the direct RAG API and the local agent.
    """

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
    """
    Serve the Local RAG AI Workspace.

    Swagger remains available at /docs.
    """

    return get_index_file()


# ==========================================================
# SYSTEM ENDPOINTS
# ==========================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=[
        "System",
    ],
)
def health() -> HealthResponse:
    """
    Return lightweight application health state.

    This endpoint does not trigger model inference.
    """

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
# RAG ENDPOINTS
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
    """
    Ask the local RAG assistant a grounded question.
    """

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
            dict(
                document
            )
            for document
            in response[
                "retrieved_documents"
            ]
        ],
    )


# ==========================================================
# AGENT ENDPOINTS
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
    """
    Process one request through the local agent.

    The agent:

    - analyzes intent
    - selects an allowed local tool
    - executes the capability
    - returns an observable execution trace
    """

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
    """
    Convert the internal agent domain model into
    the stable public API representation.
    """

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
# CONVERSATION ENDPOINTS
# ==========================================================


@app.get(
    "/history",
    response_model=HistoryResponse,
    tags=[
        "Conversation",
    ],
)
def history() -> HistoryResponse:
    """
    Return successful conversation history.
    """

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
    """
    Clear current conversational memory.
    """

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
# KNOWLEDGE BASE ENDPOINTS
# ==========================================================


@app.post(
    "/sync",
    response_model=SyncResponse,
    tags=[
        "Knowledge Base",
    ],
)
def sync_knowledge_base() -> SyncResponse:
    """
    Synchronize local source files with the SQLite
    knowledge base.
    """

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