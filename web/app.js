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
            "Ask your local knowledge base";


        workspaceDescription.textContent =
            "Answers are grounded only in your indexed local documents.";


        modeStatusTitle.textContent =
            "Assistant Mode";


        modeStatusDetail.textContent =
            "Direct RAG pipeline";


        composerModeLabel.textContent =
            "Local inference · Direct RAG";


        questionInput.placeholder =
            "Yerel belgeleriniz hakkında bir soru sorun...";


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
            "0 0 14px rgba(124, 140, 255, 0.55)";


        agentPanel
            .classList
            .add(
                "hidden"
            );
    }

    else {
        workspaceTitle.textContent =
            "Run your local AI agent";


        workspaceDescription.textContent =
            "The agent detects intent, selects an allowed local tool and returns a traceable result.";


        modeStatusTitle.textContent =
            "Agent Mode";


        modeStatusDetail.textContent =
            "Intent · tool · execution";


        composerModeLabel.textContent =
            "Local inference · Bounded Agent";


        questionInput.placeholder =
            "Agent'a bir görev veya bilgi isteği yazın...";


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
            "0 0 14px rgba(156, 107, 255, 0.55)";
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
            "0 0 14px rgba(74, 222, 128, 0.35)";


        healthTitle.textContent =
            "Ready";


        healthDetail.textContent =
            currentMode
            === "agent"
                ? "Local agent available"
                : "Local models available";


        return;
    }


    healthDot.style.background =
        "var(--danger)";


    healthDot.style.boxShadow =
        "0 0 14px rgba(251, 113, 133, 0.35)";


    healthTitle.textContent =
        "Unavailable";


    healthDetail.textContent =
        currentMode
        === "agent"
            ? "Agent not ready"
            : "Assistant not ready";
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
            "0 0 14px rgba(251, 113, 133, 0.35)";


        healthTitle.textContent =
            "Unavailable";


        healthDetail.textContent =
            "Backend not ready";
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
                    || "Agent completed without a response."
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
                            ? "Agent succeeded"
                            : "Agent failed"
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
                    Request failed
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
            "Kaynak detayı bulunamadı."
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
        || "İçerik bulunamadı.";


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
            "Bu cevap için retrieval bilgisi bulunamadı."
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
                No retrieved documents.
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
                    Similarity:
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
            Henüz retrieval sonucu yok.
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
            ? "SUCCEEDED"
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
                No execution steps returned.
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
        "No agent execution yet.";


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
            Henüz agent çalıştırılmadı.
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


    /*
        knowledge_search çıktısının backend sürümüne göre
        RAG alanları doğrudan data içinde veya nested bir
        response/result alanında bulunabilir.

        Bu nedenle UI katmanı birkaç güvenli biçimi destekler.
    */


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
   ASSISTANT REQUEST
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
            || "Assistant request failed."
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
            || "Agent request failed."
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


    /*
        Retrieval Diagnostics sadece agent gerçekten
        knowledge_search kullandıysa gösterilir.
    */


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
            ? "Running..."
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
            || "Unexpected error."
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
   KNOWLEDGE BASE
   ========================================================= */


function renderSources(
    sources
) {
    const normalizedSources =
        Array.isArray(
            sources
        )
            ? sources
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
                No indexed sources.
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


async function syncKnowledgeBase(
    showSuccessToast = true
) {
    syncButton.disabled =
        true;


    syncButton.textContent =
        "Syncing...";


    try {
        const response =
            await fetch(
                "/sync",
                {
                    method:
                        "POST"
                }
            );


        const payload =
            await response.json();


        if (
            !response.ok
        ) {
            throw new Error(
                payload.detail
                || "Sync failed."
            );
        }


        const combinedSources = [
            ...(
                Array.isArray(
                    payload.unchanged_files
                )
                    ? payload.unchanged_files
                    : []
            ),

            ...(
                Array.isArray(
                    payload.new_files
                )
                    ? payload.new_files
                    : []
            ),

            ...(
                Array.isArray(
                    payload.modified_files
                )
                    ? payload.modified_files
                    : []
            )
        ];


        const uniqueSources =
            [
                ...new Set(
                    combinedSources
                )
            ]
            .sort();


        renderSources(
            uniqueSources
        );


        if (
            showSuccessToast
        ) {
            if (
                payload.has_changes
            ) {
                showToast(
                    "Knowledge base synchronized."
                );
            }

            else {
                showToast(
                    "Knowledge base already up to date."
                );
            }
        }
    }

    catch (
        error
    ) {
        showToast(
            error.message
            || "Sync failed."
        );
    }

    finally {
        syncButton.disabled =
            false;


        syncButton.textContent =
            "Sync Knowledge Base";
    }
}


/* =========================================================
   CONVERSATION
   ========================================================= */


function renderClearedWelcome() {
    chatStream.innerHTML = `
        <div class="welcome-card">

            <div class="welcome-icon">
                AI
            </div>

            <div>

                <h3>
                    Conversation cleared
                </h3>

                <p>
                    Yeni bir yerel AI oturumu
                    başlatabilirsiniz.
                </p>

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
                "Conversation could not be cleared."
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
            || "Conversation could not be cleared."
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
                "Bu agent çalıştırmasında retrieval kullanılmadı."
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


    setMode(
        "assistant"
    );


    await Promise.all([
        loadHealth(),

        syncKnowledgeBase(
            false
        )
    ]);


    questionInput.focus();
}


initializeApplication();