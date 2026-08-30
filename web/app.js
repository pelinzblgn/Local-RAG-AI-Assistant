/* =========================================================
   DOM REFERENCES
   ========================================================= */


const chatForm =
    document.getElementById(
        "chat-form"
    );


const questionInput =
    document.getElementById(
        "question-input"
    );


const sendButton =
    document.getElementById(
        "send-button"
    );


const chatStream =
    document.getElementById(
        "chat-stream"
    );


const syncButton =
    document.getElementById(
        "sync-button"
    );


const clearButton =
    document.getElementById(
        "clear-button"
    );


const sourceList =
    document.getElementById(
        "source-list"
    );


const sourceCount =
    document.getElementById(
        "source-count"
    );


const healthDot =
    document.getElementById(
        "health-dot"
    );


const healthTitle =
    document.getElementById(
        "health-title"
    );


const healthDetail =
    document.getElementById(
        "health-detail"
    );


const workspaceTitle =
    document.getElementById(
        "workspace-title"
    );


const workspaceDescription =
    document.getElementById(
        "workspace-description"
    );


const assistantModeButton =
    document.getElementById(
        "assistant-mode-button"
    );


const agentModeButton =
    document.getElementById(
        "agent-mode-button"
    );


const modeStatusDot =
    document.getElementById(
        "mode-status-dot"
    );


const modeStatusTitle =
    document.getElementById(
        "mode-status-title"
    );


const modeStatusDetail =
    document.getElementById(
        "mode-status-detail"
    );


const composerModeLabel =
    document.getElementById(
        "composer-mode-label"
    );


const openDiagnostics =
    document.getElementById(
        "open-diagnostics"
    );


const closeDiagnostics =
    document.getElementById(
        "close-diagnostics"
    );


const diagnosticsPanel =
    document.getElementById(
        "diagnostics-panel"
    );


const openAgentTrace =
    document.getElementById(
        "open-agent-trace"
    );


const closeAgentPanel =
    document.getElementById(
        "close-agent-panel"
    );


const agentPanel =
    document.getElementById(
        "agent-panel"
    );


const sourceModal =
    document.getElementById(
        "source-modal"
    );


const sourceModalTitle =
    document.getElementById(
        "source-modal-title"
    );


const sourceModalScore =
    document.getElementById(
        "source-modal-score"
    );


const sourceModalContent =
    document.getElementById(
        "source-modal-content"
    );


const closeSourceModal =
    document.getElementById(
        "close-source-modal"
    );


const sourceModalBackdrop =
    sourceModal.querySelector(
        ".source-modal-backdrop"
    );


const toast =
    document.getElementById(
        "toast"
    );


const knowledgeFileInput =
    document.getElementById(
        "knowledge-file-input"
    );


const selectKnowledgeFileButton =
    document.getElementById(
        "select-knowledge-file-button"
    );


const selectedFileInfo =
    document.getElementById(
        "selected-file-info"
    );


const selectedFileName =
    document.getElementById(
        "selected-file-name"
    );


const uploadKnowledgeFileButton =
    document.getElementById(
        "upload-knowledge-file-button"
    );


const knowledgeUploadStatus =
    document.getElementById(
        "knowledge-upload-status"
    );


/* =========================================================
   APPLICATION STATE
   ========================================================= */


let currentMode =
    "assistant";


let latestResponse =
    null;


let latestAgentResponse =
    null;


let latestHealth =
    null;


let diagnosticsAvailable =
    false;


let selectedKnowledgeFile =
    null;


const MAX_KNOWLEDGE_FILE_SIZE =
    5 * 1024 * 1024;


const SUPPORTED_KNOWLEDGE_FILE_EXTENSIONS = [
    ".txt",
    ".pdf",
    ".docx"
];

/* =========================================================
   GENERAL HELPERS
   ========================================================= */


function showToast(
    message
) {
    toast.textContent =
        message;


    toast.classList.add(
        "visible"
    );


    window.setTimeout(
        () => {
            toast.classList.remove(
                "visible"
            );
        },
        2200
    );
}


function escapeHtml(
    value
) {
    const element =
        document.createElement(
            "div"
        );


    element.textContent =
        String(
            value ?? ""
        );


    return element.innerHTML;
}


function scrollToBottom() {
    chatStream.scrollTop =
        chatStream.scrollHeight;
}


function getConfidenceClass(
    level
) {
    const normalized =
        String(
            level || ""
        ).toLowerCase();


    if (
        normalized
        === "high"
    ) {
        return "confidence-high";
    }


    if (
        normalized
        === "medium"
    ) {
        return "confidence-medium";
    }


    return "confidence-low";
}


function formatPercentage(
    value
) {
    return `${
        (
            Number(
                value || 0
            )
            * 100
        ).toFixed(0)
    }%`;
}


function formatExecutionTime(
    value
) {
    const executionMs =
        Number(
            value
        );


    if (
        !Number.isFinite(
            executionMs
        )
    ) {
        return "-";
    }


    if (
        executionMs
        < 1000
    ) {
        return `${
            executionMs.toFixed(1)
        } ms`;
    }


    return `${
        (
            executionMs
            / 1000
        ).toFixed(2)
    } s`;
}


function normalizeStatusClass(
    status
) {
    const normalized =
        String(
            status || ""
        ).toLowerCase();


    if (
        normalized
        === "completed"
        ||
        normalized
        === "success"
        ||
        normalized
        === "succeeded"
    ) {
        return "completed";
    }


    if (
        normalized
        === "failed"
        ||
        normalized
        === "error"
    ) {
        return "failed";
    }


    return "pending";
}


/* =========================================================
   WORKSPACE MODE
   ========================================================= */


function setMode(
    mode
) {
    if (
        mode
        !== "assistant"
        &&
        mode
        !== "agent"
    ) {
        return;
    }


    currentMode =
        mode;


    const isAssistant =
        currentMode
        === "assistant";


    assistantModeButton
        .classList
        .toggle(
            "active",
            isAssistant
        );


    agentModeButton
        .classList
        .toggle(
            "active",
            !isAssistant
        );


    if (
        isAssistant
    ) {
        workspaceTitle.textContent =
            "What are you working on today?";


        workspaceDescription.textContent =
            "Ask questions, review concepts, and explore your own documents privately.";


        modeStatusTitle.textContent =
            "Study Mode";


        modeStatusDetail.textContent =
            "Your document library";


        composerModeLabel.textContent =
            "Private · On-device";


        questionInput.placeholder =
            "Ask something from your documents...";


        sendButton.textContent =
            "Ask";


        openAgentTrace
            .classList
            .add(
                "hidden-control"
            );


        openDiagnostics
            .classList
            .remove(
                "hidden-control"
            );


        modeStatusDot.style.background =
            "var(--accent)";


        modeStatusDot.style.boxShadow =
            "none";


        agentPanel
            .classList
            .add(
                "hidden"
            );
    }

    else {
        workspaceTitle.textContent =
            "Work with your documents";


        workspaceDescription.textContent =
            "Use LocalMind's controlled local tools to search, check, and manage your knowledge library.";


        modeStatusTitle.textContent =
            "Agent Mode";


        modeStatusDetail.textContent =
            "Controlled local actions";


        composerModeLabel.textContent =
            "Private · Local tools";


        questionInput.placeholder =
            "Tell LocalMind what you want to do...";


        sendButton.textContent =
            "Run";


        openAgentTrace
            .classList
            .remove(
                "hidden-control"
            );


        updateDiagnosticsButtonVisibility();


        modeStatusDot.style.background =
            "var(--accent-2)";


        modeStatusDot.style.boxShadow =
            "none";
    }


    updateHealthDisplay();


    questionInput.focus();
}


/* =========================================================
   HEALTH
   ========================================================= */


function updateHealthDisplay() {
    if (
        !latestHealth
    ) {
        return;
    }


    const assistantReady =
        Boolean(
            latestHealth
                .assistant_ready
        );


    const agentReady =
        Boolean(
            latestHealth
                .agent_ready
        );


    const ready =
        currentMode
        === "agent"
            ? agentReady
            : assistantReady;


    if (
        ready
    ) {
        healthDot.style.background =
            "var(--success)";


        healthDot.style.boxShadow =
            "none";


        healthTitle.textContent =
            "Ready";


        healthDetail.textContent =
            currentMode
            === "agent"
                ? "Local tools available"
                : "LocalMind is ready";


        return;
    }


    healthDot.style.background =
        "var(--danger)";


    healthDot.style.boxShadow =
        "none";


    healthTitle.textContent =
        "Unavailable";


    healthDetail.textContent =
        currentMode
        === "agent"
            ? "Local tools unavailable"
            : "LocalMind is unavailable";
}


async function loadHealth() {
    try {
        const response =
            await fetch(
                "/health"
            );


        const payload =
            await response.json();


        if (
            !response.ok
        ) {
            throw new Error(
                "Health check failed."
            );
        }


        latestHealth =
            payload;


        updateHealthDisplay();
    }

    catch {
        latestHealth =
            null;


        healthDot.style.background =
            "var(--danger)";


        healthDot.style.boxShadow =
            "none";


        healthTitle.textContent =
            "Unavailable";


        healthDetail.textContent =
            "Local service not ready";
    }
}


/* =========================================================
   CHAT MESSAGE RENDERING
   ========================================================= */


function addUserMessage(
    question
) {
    const wrapper =
        document.createElement(
            "div"
        );


    wrapper.className =
        "message user";


    wrapper.innerHTML = `
        <div class="message-bubble">
            ${escapeHtml(question)}
        </div>
    `;


    chatStream.appendChild(
        wrapper
    );


    scrollToBottom();
}


function addAssistantMessage(
    response
) {
    const wrapper =
        document.createElement(
            "div"
        );


    wrapper.className =
        "message assistant-message";


    const confidence =
        response.confidence || {};


    const confidenceLevel =
        String(
            confidence.level
            || "unknown"
        ).toUpperCase();


    const evidenceCoverage =
        Number(
            confidence
                .evidence_coverage
            || 0
        );


    const sources =
        Array.isArray(
            response.sources
        )
            ? response.sources
            : [];


    const sourceButtons =
        sources
            .map(
                source => `
                    <button
                        type="button"
                        class="source-chip"
                        data-source="${escapeHtml(source)}"
                    >
                        ${escapeHtml(source)}
                    </button>
                `
            )
            .join("");


    wrapper.innerHTML = `
        <div class="assistant-card">

            <p>
                ${escapeHtml(
                    response.answer
                    || "No answer returned."
                )}
            </p>

            <div class="answer-meta">

                <span
                    class="
                        meta-chip
                        ${getConfidenceClass(
                            confidenceLevel
                        )}
                    "
                >
                    Confidence
                    ${escapeHtml(
                        confidenceLevel
                    )}
                </span>

                <span class="meta-chip">
                    Evidence
                    ${(
                        evidenceCoverage
                        * 100
                    ).toFixed(0)}%
                </span>

            </div>

            ${
                sourceButtons
                    ? `
                        <div class="source-chips">
                            ${sourceButtons}
                        </div>
                    `
                    : ""
            }

        </div>
    `;


    chatStream.appendChild(
        wrapper
    );


    const buttons =
        wrapper.querySelectorAll(
            ".source-chip"
        );


    for (
        const button
        of buttons
    ) {
        button.addEventListener(
            "click",
            () => {
                openSourceByName(
                    button.dataset.source
                );
            }
        );
    }


    scrollToBottom();
}


function addAgentMessage(
    response
) {
    const wrapper =
        document.createElement(
            "div"
        );


    wrapper.className =
        "message assistant-message";


    const decision =
        response.decision || {};


    const metadata =
        response.metadata || {};


    const toolName =
        decision.tool_name
        || metadata.selected_tool
        || "unknown";


    const intent =
        decision.intent
        || "unknown";


    const succeeded =
        Boolean(
            response.succeeded
        );


    const resultClass =
        succeeded
            ? "confidence-high"
            : "confidence-low";


    wrapper.innerHTML = `
        <div class="assistant-card">

            <p>
                ${escapeHtml(
                    response.answer
                    || "LocalMind completed the action without a response."
                )}
            </p>

            <div class="answer-meta">

                <span
                    class="
                        meta-chip
                        ${resultClass}
                    "
                >
                    ${
                        succeeded
                            ? "Completed"
                            : "Failed"
                    }
                </span>

                <span
                    class="
                        meta-chip
                        agent-chip
                    "
                >
                    ${escapeHtml(
                        intent
                    )}
                </span>

                <span
                    class="
                        meta-chip
                        tool-chip
                    "
                >
                    ${escapeHtml(
                        toolName
                    )}
                </span>

                <span class="meta-chip">
                    ${escapeHtml(
                        formatExecutionTime(
                            metadata.execution_ms
                        )
                    )}
                </span>

            </div>

        </div>
    `;


    chatStream.appendChild(
        wrapper
    );


    scrollToBottom();
}


function addErrorMessage(
    message
) {
    const wrapper =
        document.createElement(
            "div"
        );


    wrapper.className =
        "message assistant-message";


    wrapper.innerHTML = `
        <div class="assistant-card">

            <p>
                ${escapeHtml(message)}
            </p>

            <div class="answer-meta">

                <span
                    class="
                        meta-chip
                        confidence-low
                    "
                >
                    Something went wrong
                </span>

            </div>

        </div>
    `;


    chatStream.appendChild(
        wrapper
    );


    scrollToBottom();
}


/* =========================================================
   SOURCE INSPECTOR
   ========================================================= */


function openSourceInspector(
    retrievedDocument
) {
    if (
        !retrievedDocument
    ) {
        showToast(
            "Source details are not available."
        );


        return;
    }


    sourceModalTitle.textContent =
        retrievedDocument.source
        || "Unknown source";


    sourceModalScore.textContent =
        Number(
            retrievedDocument.score
            || 0
        ).toFixed(4);


    sourceModalContent.textContent =
        retrievedDocument.content
        || "No source content is available.";


    sourceModal
        .classList
        .remove(
            "hidden"
        );
}


function closeSourceInspector() {
    sourceModal
        .classList
        .add(
            "hidden"
        );
}


function openSourceByName(
    sourceName
) {
    if (
        !latestResponse
        ||
        !Array.isArray(
            latestResponse
                .retrieved_documents
        )
    ) {
        showToast(
            "Answer details are not available for this response."
        );


        return;
    }


    const documentMatch =
        latestResponse
            .retrieved_documents
            .find(
                item =>
                    item.source
                    === sourceName
            );


    openSourceInspector(
        documentMatch
    );
}


/* =========================================================
   RETRIEVAL DIAGNOSTICS
   ========================================================= */


function updateDiagnosticsButtonVisibility() {
    if (
        currentMode
        === "assistant"
    ) {
        openDiagnostics
            .classList
            .remove(
                "hidden-control"
            );


        return;
    }


    openDiagnostics
        .classList
        .toggle(
            "hidden-control",
            !diagnosticsAvailable
        );
}


function updateDiagnostics(
    response
) {
    latestResponse =
        response;


    diagnosticsAvailable =
        true;


    updateDiagnosticsButtonVisibility();


    const confidence =
        response.confidence || {};


    const rewrite =
        response.query_rewrite || {};


    document.getElementById(
        "metric-confidence"
    ).textContent =
        String(
            confidence.level
            || "-"
        ).toUpperCase();


    document.getElementById(
        "metric-evidence"
    ).textContent =
        formatPercentage(
            confidence
                .evidence_coverage
        );


    document.getElementById(
        "metric-top-score"
    ).textContent =
        confidence.top_score
        === undefined
        ||
        confidence.top_score
        === null
            ? "-"
            : Number(
                confidence.top_score
            ).toFixed(4);


    document.getElementById(
        "metric-score-gap"
    ).textContent =
        confidence.score_gap
        === null
        ||
        confidence.score_gap
        === undefined
            ? "N/A"
            : Number(
                confidence.score_gap
            ).toFixed(4);


    document.getElementById(
        "metric-selected"
    ).textContent =
        `${
            confidence.selected_count
            || 0
        }/${
            confidence.total_count
            || 0
        }`;


    document.getElementById(
        "metric-filtered"
    ).textContent =
        String(
            confidence.filtered_count
            || 0
        );


    document.getElementById(
        "rewrite-status"
    ).textContent =
        rewrite.was_rewritten
            ? "YES"
            : "NO";


    document.getElementById(
        "original-query"
    ).textContent =
        rewrite.original_query
        || "-";


    document.getElementById(
        "retrieval-query"
    ).textContent =
        rewrite.retrieval_query
        || "-";


    const retrievedList =
        document.getElementById(
            "retrieved-list"
        );


    retrievedList.innerHTML =
        "";


    const documents =
        Array.isArray(
            response
                .retrieved_documents
        )
            ? response
                .retrieved_documents
            : [];


    if (
        documents.length
        === 0
    ) {
        retrievedList.innerHTML = `
            <p class="muted">
                No supporting documents were retrieved.
            </p>
        `;


        return;
    }


    documents.forEach(
        (
            retrievedDocument,
            index
        ) => {
            const item =
                document.createElement(
                    "button"
                );


            item.type =
                "button";


            item.className =
                "retrieved-item";


            item.innerHTML = `
                <div
                    class="
                        retrieved-item-header
                    "
                >

                    <strong>
                        ${escapeHtml(
                            retrievedDocument
                                .source
                        )}
                    </strong>

                    <span
                        class="
                            retrieved-rank
                        "
                    >
                        #${index + 1}
                    </span>

                </div>

                <span
                    class="
                        retrieved-score
                    "
                >
                    Relevance:
                    ${Number(
                        retrievedDocument
                            .score
                        || 0
                    ).toFixed(4)}
                </span>
            `;


            item.addEventListener(
                "click",
                () => {
                    openSourceInspector(
                        retrievedDocument
                    );
                }
            );


            retrievedList.appendChild(
                item
            );
        }
    );
}


function resetDiagnostics() {
    latestResponse =
        null;


    diagnosticsAvailable =
        false;


    document.getElementById(
        "metric-confidence"
    ).textContent = "-";


    document.getElementById(
        "metric-evidence"
    ).textContent = "-";


    document.getElementById(
        "metric-top-score"
    ).textContent = "-";


    document.getElementById(
        "metric-score-gap"
    ).textContent = "-";


    document.getElementById(
        "metric-selected"
    ).textContent = "-";


    document.getElementById(
        "metric-filtered"
    ).textContent = "-";


    document.getElementById(
        "rewrite-status"
    ).textContent = "-";


    document.getElementById(
        "original-query"
    ).textContent = "-";


    document.getElementById(
        "retrieval-query"
    ).textContent = "-";


    document.getElementById(
        "retrieved-list"
    ).innerHTML = `
        <p class="muted">
            No answer details yet.
        </p>
    `;


    diagnosticsPanel
        .classList
        .add(
            "hidden"
        );


    updateDiagnosticsButtonVisibility();
}


/* =========================================================
   AGENT TRACE
   ========================================================= */


function updateAgentTrace(
    response
) {
    latestAgentResponse =
        response;


    const decision =
        response.decision || {};


    const metadata =
        response.metadata || {};


    const toolResult =
        response.tool_result || {};


    document.getElementById(
        "agent-intent"
    ).textContent =
        decision.intent
        || "-";


    document.getElementById(
        "agent-confidence"
    ).textContent =
        decision.confidence
        === undefined
        ||
        decision.confidence
        === null
            ? "-"
            : `${
                (
                    Number(
                        decision.confidence
                    )
                    * 100
                ).toFixed(0)
            }%`;


    document.getElementById(
        "agent-tool"
    ).textContent =
        decision.tool_name
        || metadata.selected_tool
        || "-";


    document.getElementById(
        "agent-execution"
    ).textContent =
        formatExecutionTime(
            metadata.execution_ms
        );


    document.getElementById(
        "agent-reason"
    ).textContent =
        decision.reason
        || "-";


    document.getElementById(
        "agent-local-only"
    ).textContent =
        metadata.local_only
            ? "YES"
            : "NO";


    document.getElementById(
        "agent-result-status"
    ).textContent =
        response.succeeded
            ? "COMPLETED"
            : "FAILED";


    document.getElementById(
        "agent-tool-result-status"
    ).textContent =
        toolResult.success
            ? "SUCCESS"
            : "FAILED";


    document.getElementById(
        "agent-tool-output"
    ).textContent =
        toolResult.content
        || toolResult.error
        || "-";


    renderAgentSteps(
        response.steps
    );
}


function renderAgentSteps(
    steps
) {
    const container =
        document.getElementById(
            "agent-steps"
        );


    container.innerHTML =
        "";


    if (
        !Array.isArray(
            steps
        )
        ||
        steps.length
        === 0
    ) {
        container.innerHTML = `
            <p class="muted">
                No execution steps were returned.
            </p>
        `;


        return;
    }


    steps.forEach(
        step => {
            const statusClass =
                normalizeStatusClass(
                    step.status
                );


            const item =
                document.createElement(
                    "div"
                );


            item.className =
                `agent-step ${statusClass}`;


            item.innerHTML = `
                <div
                    class="
                        agent-step-header
                    "
                >

                    <strong>
                        ${escapeHtml(
                            step.name
                            || "Agent step"
                        )}
                    </strong>

                    <span
                        class="
                            agent-step-status
                        "
                    >
                        ${escapeHtml(
                            step.status
                            || "unknown"
                        )}
                    </span>

                </div>

                <p>
                    ${escapeHtml(
                        step.detail
                        || "-"
                    )}
                </p>

                ${
                    step.tool_name
                        ? `
                            <span
                                class="
                                    agent-step-tool
                                "
                            >
                                Tool:
                                ${escapeHtml(
                                    step.tool_name
                                )}
                            </span>
                        `
                        : ""
                }
            `;


            container.appendChild(
                item
            );
        }
    );
}


function resetAgentTrace() {
    latestAgentResponse =
        null;


    document.getElementById(
        "agent-intent"
    ).textContent = "-";


    document.getElementById(
        "agent-confidence"
    ).textContent = "-";


    document.getElementById(
        "agent-tool"
    ).textContent = "-";


    document.getElementById(
        "agent-execution"
    ).textContent = "-";


    document.getElementById(
        "agent-reason"
    ).textContent =
        "No agent action yet.";


    document.getElementById(
        "agent-local-only"
    ).textContent = "-";


    document.getElementById(
        "agent-result-status"
    ).textContent = "-";


    document.getElementById(
        "agent-tool-result-status"
    ).textContent = "-";


    document.getElementById(
        "agent-tool-output"
    ).textContent =
        "No tool result yet.";


    document.getElementById(
        "agent-steps"
    ).innerHTML = `
        <p class="muted">
            No agent action yet.
        </p>
    `;


    agentPanel
        .classList
        .add(
            "hidden"
        );
}


/* =========================================================
   AGENT → RAG PAYLOAD EXTRACTION
   ========================================================= */


function extractRagPayloadFromAgent(
    agentResponse
) {
    const toolResult =
        agentResponse.tool_result || {};


    const data =
        toolResult.data || {};


    const candidates = [
        data,
        data.response,
        data.result,
        data.answer_result,
        data.assistant_response
    ];


    for (
        const candidate
        of candidates
    ) {
        if (
            !candidate
            ||
            typeof candidate
            !== "object"
        ) {
            continue;
        }


        const hasRagMetadata =
            candidate.confidence
            ||
            candidate.query_rewrite
            ||
            candidate.retrieved_documents
            ||
            candidate.sources;


        if (
            hasRagMetadata
        ) {
            return {
                answer:
                    candidate.answer
                    || agentResponse.answer
                    || "",

                sources:
                    Array.isArray(
                        candidate.sources
                    )
                        ? candidate.sources
                        : [],

                confidence:
                    candidate.confidence
                    || {},

                query_rewrite:
                    candidate.query_rewrite
                    || {},

                retrieved_documents:
                    Array.isArray(
                        candidate
                            .retrieved_documents
                    )
                        ? candidate
                            .retrieved_documents
                        : []
            };
        }
    }


    return null;
}


/* =========================================================
   CHAT REQUEST
   ========================================================= */


async function askAssistant(
    question
) {
    const response =
        await fetch(
            "/chat",
            {
                method:
                    "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify(
                        {
                            question
                        }
                    )
            }
        );


    const payload =
        await response.json();


    if (
        !response.ok
    ) {
        throw new Error(
            payload.detail
            || "LocalMind could not answer this question."
        );
    }


    addAssistantMessage(
        payload
    );


    updateDiagnostics(
        payload
    );


    return payload;
}


/* =========================================================
   AGENT REQUEST
   ========================================================= */


async function runAgent(
    message
) {
    const response =
        await fetch(
            "/agent",
            {
                method:
                    "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify(
                        {
                            message
                        }
                    )
            }
        );


    const payload =
        await response.json();


    if (
        !response.ok
    ) {
        throw new Error(
            payload.detail
            || "LocalMind could not complete this action."
        );
    }


    addAgentMessage(
        payload
    );


    updateAgentTrace(
        payload
    );


    const selectedTool =
        payload.decision
            ?.tool_name
        ||
        payload.metadata
            ?.selected_tool
        ||
        "";


    if (
        selectedTool
        === "knowledge_search"
    ) {
        const ragPayload =
            extractRagPayloadFromAgent(
                payload
            );


        if (
            ragPayload
        ) {
            updateDiagnostics(
                ragPayload
            );
        }

        else {
            resetDiagnostics();
        }
    }

    else {
        resetDiagnostics();
    }


    updateDiagnosticsButtonVisibility();


    return payload;
}


/* =========================================================
   REQUEST ROUTER
   ========================================================= */


async function submitRequest(
    question
) {
    sendButton.disabled =
        true;


    sendButton.textContent =
        currentMode
        === "agent"
            ? "Working..."
            : "Thinking...";


    try {
        if (
            currentMode
            === "agent"
        ) {
            await runAgent(
                question
            );
        }

        else {
            await askAssistant(
                question
            );
        }
    }

    catch (
        error
    ) {
        addErrorMessage(
            error.message
            || "Something went wrong."
        );
    }

    finally {
        sendButton.disabled =
            false;


        sendButton.textContent =
            currentMode
            === "agent"
                ? "Run"
                : "Ask";


        questionInput.focus();
    }
}


/* =========================================================
   KNOWLEDGE LIBRARY
   ========================================================= */


function setKnowledgeUploadStatus(
    message,
    state = ""
) {
    knowledgeUploadStatus.textContent =
        message || "";


    knowledgeUploadStatus.classList.remove(
        "success",
        "error",
        "loading"
    );


    if (
        state
    ) {
        knowledgeUploadStatus
            .classList
            .add(
                state
            );
    }
}


function resetKnowledgeFileSelection() {
    selectedKnowledgeFile =
        null;


    knowledgeFileInput.value =
        "";


    selectedFileName.textContent =
        "-";


    selectedFileInfo
        .classList
        .add(
            "hidden"
        );


    uploadKnowledgeFileButton
        .classList
        .add(
            "hidden"
        );


    uploadKnowledgeFileButton.disabled =
        true;
}


function validateKnowledgeFile(
    file
) {
    if (
        !file
    ) {
        throw new Error(
            "Choose a TXT, PDF, or DOCX document first."
        );
    }


    const fileName =
        String(
            file.name || ""
        ).trim();


    const normalizedFileName =
        fileName.toLowerCase();


    const isSupported =
        SUPPORTED_KNOWLEDGE_FILE_EXTENSIONS
            .some(
                extension =>
                    normalizedFileName
                        .endsWith(
                            extension
                        )
            );


    if (
        !isSupported
    ) {
        throw new Error(
            "Supported document types are TXT, PDF, and DOCX."
        );
    }


    if (
        file.size
        === 0
    ) {
        throw new Error(
            "The selected document is empty."
        );
    }


    if (
        file.size
        > MAX_KNOWLEDGE_FILE_SIZE
    ) {
        throw new Error(
            "The document exceeds the 5 MB limit."
        );
    }
}


function handleKnowledgeFileSelection() {
    const file =
        knowledgeFileInput
            .files?.[0]
        || null;


    try {
        validateKnowledgeFile(
            file
        );


        selectedKnowledgeFile =
            file;


        selectedFileName.textContent =
            file.name;


        selectedFileInfo
            .classList
            .remove(
                "hidden"
            );


        uploadKnowledgeFileButton
            .classList
            .remove(
                "hidden"
            );


        uploadKnowledgeFileButton.disabled =
            false;


        setKnowledgeUploadStatus(
            "Ready to add this document."
        );
    }

    catch (
        error
    ) {
        resetKnowledgeFileSelection();


        setKnowledgeUploadStatus(
            error.message
            || "The document could not be selected.",
            "error"
        );


        showToast(
            error.message
            || "The document could not be selected."
        );
    }
}


function renderSources(
    sources
) {
    const normalizedSources =
        Array.isArray(
            sources
        )
            ? [
                ...new Set(
                    sources
                        .map(
                            source =>
                                String(
                                    source || ""
                                ).trim()
                        )
                        .filter(
                            Boolean
                        )
                )
            ].sort(
                (
                    first,
                    second
                ) =>
                    first.localeCompare(
                        second,
                        undefined,
                        {
                            sensitivity:
                                "base"
                        }
                    )
            )
            : [];


    sourceCount.textContent =
        String(
            normalizedSources.length
        );


    if (
        normalizedSources.length
        === 0
    ) {
        sourceList.innerHTML = `
            <p class="muted">
                Your library is empty.
            </p>
        `;


        return;
    }


    sourceList.innerHTML =
        normalizedSources
            .map(
                source => `
                    <div
                        class="source-item"
                        title="${escapeHtml(source)}"
                    >
                        ${escapeHtml(source)}
                    </div>
                `
            )
            .join("");
}


/*
 * Load the complete knowledge library directly from SQLite
 * through the API.
 *
 * This endpoint is the canonical source for the Library UI.
 * It includes both:
 *
 * - managed documents from data/raw
 * - explicitly uploaded external documents
 */
async function loadKnowledgeSources() {
    const response =
        await fetch(
            "/knowledge/sources"
        );


    let payload =
        {};


    try {
        payload =
            await response.json();
    }

    catch {
        payload =
            {};
    }


    if (
        !response.ok
    ) {
        throw new Error(
            payload.detail
            || "Your library could not be loaded."
        );
    }


    const sources =
        Array.isArray(
            payload.sources
        )
            ? payload.sources
            : [];


    renderSources(
        sources
    );


    return sources;
}


async function uploadKnowledgeFile() {
    if (
    !selectedKnowledgeFile
) {
    setKnowledgeUploadStatus(
        "Choose a TXT, PDF, or DOCX document first.",
        "error"
    );


    return;
}


    try {
        validateKnowledgeFile(
            selectedKnowledgeFile
        );
    }

    catch (
        error
    ) {
        setKnowledgeUploadStatus(
            error.message,
            "error"
        );


        return;
    }


    const fileToUpload =
        selectedKnowledgeFile;


    const formData =
        new FormData();


    formData.append(
        "file",
        fileToUpload
    );


    selectKnowledgeFileButton.disabled =
        true;


    uploadKnowledgeFileButton.disabled =
        true;


    uploadKnowledgeFileButton.textContent =
        "Adding...";


    setKnowledgeUploadStatus(
        "Adding this document to your private library...",
        "loading"
    );


    try {
        const response =
            await fetch(
                "/knowledge/files",
                {
                    method:
                        "POST",

                    body:
                        formData
                }
            );


        let payload =
            {};


        try {
            payload =
                await response.json();
        }

        catch {
            payload =
                {};
        }


        if (
            !response.ok
        ) {
            throw new Error(
                payload.detail
                || "The document could not be added."
            );
        }


        /*
         * Do not manually merge the uploaded source into
         * the sidebar. Reload the canonical Library state
         * from the backend instead.
         */
        await loadKnowledgeSources();


        const insertedChunks =
            Number(
                payload.inserted_chunks
                || 0
            );


        setKnowledgeUploadStatus(
            `${
                fileToUpload.name
            } added successfully · ${
                insertedChunks
            } chunk${
                insertedChunks === 1
                    ? ""
                    : "s"
            }`,
            "success"
        );


        showToast(
            "Document added to your library."
        );


        resetKnowledgeFileSelection();
    }

    catch (
        error
    ) {
        setKnowledgeUploadStatus(
            error.message
            || "The document could not be added.",
            "error"
        );


        showToast(
            error.message
            || "The document could not be added."
        );
    }

    finally {
        selectKnowledgeFileButton.disabled =
            false;


        uploadKnowledgeFileButton.textContent =
            "Add to Library";


        if (
            selectedKnowledgeFile
        ) {
            uploadKnowledgeFileButton.disabled =
                false;
        }
    }
}


async function syncKnowledgeBase(
    showSuccessToast = true
) {
    syncButton.disabled =
        true;


    syncButton.textContent =
        "Updating...";


    try {
        const response =
            await fetch(
                "/sync",
                {
                    method:
                        "POST"
                }
            );


        let payload =
            {};


        try {
            payload =
                await response.json();
        }

        catch {
            payload =
                {};
        }


        if (
            !response.ok
        ) {
            throw new Error(
                payload.detail
                || "Your library could not be updated."
            );
        }


        /*
         * /sync manages only the application's managed
         * data/raw knowledge folder.
         *
         * After synchronization, reload the complete
         * Library from SQLite so uploaded external
         * documents remain visible.
         */
        await loadKnowledgeSources();


        if (
            showSuccessToast
        ) {
            if (
                payload.has_changes
            ) {
                showToast(
                    "Your library has been updated."
                );
            }

            else {
                showToast(
                    "Your library is already up to date."
                );
            }
        }


        return payload;
    }

    catch (
        error
    ) {
        showToast(
            error.message
            || "Your library could not be updated."
        );


        return null;
    }

    finally {
        syncButton.disabled =
            false;


        syncButton.textContent =
            "Update Library";
    }
}


/* =========================================================
   CONVERSATION
   ========================================================= */


function renderClearedWelcome() {
    chatStream.innerHTML = `
        <div class="welcome-card">

            <div class="welcome-icon">
                LM
            </div>

            <div class="welcome-content">

                <p class="welcome-label">
                    LOCALMIND
                </p>

                <h3>
                    Ready for a new study session
                </h3>

                <p>
                    Ask a question whenever you're ready.
                    LocalMind will use your private document library
                    to help you explore the topic.
                </p>

                <div class="welcome-features">

                    <span>
                        Private
                    </span>

                    <span>
                        On-device
                    </span>

                    <span>
                        Source-based
                    </span>

                </div>

            </div>

        </div>
    `;
}


async function clearConversation() {
    try {
        const response =
            await fetch(
                "/history",
                {
                    method:
                        "DELETE"
                }
            );


        if (
            !response.ok
        ) {
            throw new Error(
                "The conversation could not be cleared."
            );
        }


        renderClearedWelcome();


        resetDiagnostics();


        resetAgentTrace();


        showToast(
            "Conversation cleared."
        );


        questionInput.focus();
    }

    catch (
        error
    ) {
        showToast(
            error.message
            || "The conversation could not be cleared."
        );
    }
}


/* =========================================================
   PANEL HELPERS
   ========================================================= */


function closeAllDrawers() {
    diagnosticsPanel
        .classList
        .add(
            "hidden"
        );


    agentPanel
        .classList
        .add(
            "hidden"
        );
}


/* =========================================================
   EVENT HANDLERS
   ========================================================= */


chatForm.addEventListener(
    "submit",
    event => {
        event.preventDefault();


        const question =
            questionInput
                .value
                .trim();


        if (
            !question
        ) {
            return;
        }


        addUserMessage(
            question
        );


        questionInput.value =
            "";


        submitRequest(
            question
        );
    }
);


questionInput.addEventListener(
    "keydown",
    event => {
        if (
            event.key
            === "Enter"
            &&
            !event.shiftKey
        ) {
            event.preventDefault();


            chatForm.requestSubmit();
        }
    }
);


assistantModeButton.addEventListener(
    "click",
    () => {
        setMode(
            "assistant"
        );
    }
);


agentModeButton.addEventListener(
    "click",
    () => {
        setMode(
            "agent"
        );
    }
);


selectKnowledgeFileButton.addEventListener(
    "click",
    () => {
        knowledgeFileInput.click();
    }
);


knowledgeFileInput.addEventListener(
    "change",
    handleKnowledgeFileSelection
);


uploadKnowledgeFileButton.addEventListener(
    "click",
    uploadKnowledgeFile
);


syncButton.addEventListener(
    "click",
    () => {
        syncKnowledgeBase(
            true
        );
    }
);


clearButton.addEventListener(
    "click",
    clearConversation
);


openDiagnostics.addEventListener(
    "click",
    () => {
        if (
            currentMode
            === "agent"
            &&
            !diagnosticsAvailable
        ) {
            showToast(
                "No supporting source search was used for this action."
            );


            return;
        }


        agentPanel
            .classList
            .add(
                "hidden"
            );


        diagnosticsPanel
            .classList
            .remove(
                "hidden"
            );
    }
);


closeDiagnostics.addEventListener(
    "click",
    () => {
        diagnosticsPanel
            .classList
            .add(
                "hidden"
            );
    }
);


openAgentTrace.addEventListener(
    "click",
    () => {
        diagnosticsPanel
            .classList
            .add(
                "hidden"
            );


        agentPanel
            .classList
            .remove(
                "hidden"
            );
    }
);


closeAgentPanel.addEventListener(
    "click",
    () => {
        agentPanel
            .classList
            .add(
                "hidden"
            );
    }
);


closeSourceModal.addEventListener(
    "click",
    closeSourceInspector
);


sourceModalBackdrop.addEventListener(
    "click",
    closeSourceInspector
);


document.addEventListener(
    "keydown",
    event => {
        if (
            event.key
            !== "Escape"
        ) {
            return;
        }


        closeAllDrawers();


        closeSourceInspector();
    }
);


/* =========================================================
   INITIALIZATION
   ========================================================= */


async function initializeApplication() {
    resetDiagnostics();


    resetAgentTrace();


    resetKnowledgeFileSelection();


    setKnowledgeUploadStatus(
        ""
    );


    setMode(
        "assistant"
    );


    /*
     * Health loading and managed-folder synchronization
     * are independent.
     *
     * syncKnowledgeBase() reloads the complete Library
     * from /knowledge/sources after synchronization.
     */
    await Promise.all([
        loadHealth(),

        syncKnowledgeBase(
            false
        )
    ]);


    questionInput.focus();
}


initializeApplication();