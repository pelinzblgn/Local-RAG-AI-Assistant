# LocalMind — Private Knowledge Workspace

A privacy-focused, fully local Retrieval-Augmented Generation (RAG) workspace built with Python, Microsoft Foundry Local, Phi-4 Mini, Qwen3 Embedding 0.6B, SQLite, FastAPI, and a custom web interface.

LocalMind allows users to build a private knowledge base from selected local documents and ask grounded questions without relying on a cloud-hosted language model for the core RAG workflow.

The project combines local document ingestion, semantic retrieval, confidence-aware evidence filtering, grounded generation, conversational query rewriting, incremental synchronization, source inspection, and a bounded local agent.

## Core Principles

- Local-first AI
- Grounded and source-aware responses
- Private document processing
- Measurable RAG quality
- Controlled agent execution
- Transparent retrieval diagnostics

---

## Overview

LocalMind is a local document question-answering system based on Retrieval-Augmented Generation.

The core workflow is:

```text
Selected Documents
       |
       v
Document Loading
       |
       v
    Chunking
       |
       v
Local Embeddings
       |
       v
     SQLite
       |
       v
Semantic Retrieval
       |
       v
Evidence + Confidence Evaluation
       |
       v
Adaptive Context Filtering
       |
       +--------------------+
       |                    |
       v                    v
Insufficient Evidence   Sufficient Evidence
       |                    |
       v                    v
    Fallback          Grounded Prompt
                            |
                            v
                      Phi-4 Mini
                  Microsoft Foundry Local
                            |
                            v
                   Answer + Sources
```

Documents, embeddings, retrieval data, conversation state, and model inference are designed around local execution.

If sufficient evidence cannot be found in the indexed documents, LocalMind does not answer using unrestricted model knowledge. Instead, it returns the controlled fallback:

```text
Bu bilgi mevcut yerel belgelerde bulunamadı.
```

---

# Key Features

## Fully Local RAG Pipeline

The main RAG workflow runs locally using:

- Python
- Microsoft Foundry Local
- Phi-4 Mini
- Qwen3 Embedding 0.6B
- SQLite

This architecture reduces dependency on external AI APIs and provides greater control over private document data and inference.

---

## Multi-Format Document Upload

The web interface supports explicitly selected documents in:

- TXT
- PDF
- DOCX

Uploaded documents are processed through the same existing RAG pipeline:

```text
TXT / PDF / DOCX
        |
        v
Normalized Text
        |
        v
     Chunking
        |
        v
Local Embeddings
        |
        v
      SQLite
        |
        v
    Retrieval
```

### Format Notes

**TXT**

- UTF-8 text is supported.

**PDF**

- Text-based PDFs are supported.
- Scanned or image-only PDFs are not supported.
- OCR is intentionally not included in the current version.

**DOCX**

- Paragraph text is extracted.
- Tables, headers, footers, and other complex document structures are not guaranteed to be indexed.

### Upload Limit

Browser-selected documents are limited to:

```text
5 MB per file
```

LocalMind does not automatically scan arbitrary locations on the user's computer. External documents are processed only when explicitly selected by the user.

---

## Semantic Retrieval

Questions are converted into embeddings and compared with stored document embeddings using cosine similarity.

This allows retrieval to rank chunks according to semantic relevance rather than relying only on exact keyword matches.

---

## Confidence-Aware Retrieval

LocalMind does not automatically trust every retrieved result.

The confidence layer evaluates signals including:

- Top similarity score
- Second-best similarity score
- Score gap
- Evidence coverage
- Selected context count
- Filtered context count

Retrieval confidence is classified as:

```text
HIGH
MEDIUM
LOW
```

Low-confidence retrieval can prevent unsupported generation.

---

## Evidence Coverage

Semantic similarity alone does not guarantee that retrieved content contains enough information to answer a question.

LocalMind therefore evaluates evidence coverage in addition to vector similarity.

This helps distinguish between:

```text
Semantically related content
```

and:

```text
Evidence that directly supports an answer
```

---

## Adaptive Context Filtering

The system does not blindly send every Top-K result to the language model.

Example:

```text
Retrieved

0.80
0.74
0.31

Selected

0.80
0.74

Filtered

0.31
```

Weak or noisy retrieval results can be removed before prompt construction.

This reduces irrelevant context and improves grounding.

---

## Grounded Answer Generation

The prompt layer instructs the local model to:

- Use supplied local evidence
- Avoid unsupported outside knowledge
- Avoid unsupported assumptions
- Faithfully summarize or paraphrase supported evidence
- Avoid inventing source names
- Return the controlled fallback when evidence is insufficient

Source attribution is handled by application logic using retrieval metadata rather than trusting the language model to generate filenames.

---

## Controlled Fallback Protection

Example unsupported query:

```text
Question:
Fransa'nın başkenti nedir?

Answer:
Bu bilgi mevcut yerel belgelerde bulunamadı.

Confidence:
LOW

Evidence:
0%
```

Phi-4 Mini may know that the answer is Paris, but LocalMind intentionally prevents unrestricted general knowledge from bypassing the RAG grounding rules.

This is expected behavior, not a retrieval failure.

---

## Trusted Source Attribution

Sources are derived from structured retrieval metadata.

Example:

```text
Question:
STM32 nedir?

Answer:
STM32, STMicroelectronics tarafından geliştirilen
ARM tabanlı bir mikrodenetleyici ailesidir.

Source:
stm32_notes.txt

Confidence:
HIGH
```

The language model is not responsible for inventing source filenames.

---

## Conversational Query Rewriting

LocalMind supports contextual follow-up questions.

Example:

```text
User:
STM32 nedir?

User:
Peki PWM ne işe yarar?
```

When conversation context is useful, the second query can internally become:

```text
Original Query:
Peki PWM ne işe yarar?

Retrieval Query:
STM32 nedir? Peki PWM ne işe yarar?
```

Standalone questions remain isolated when previous context is unnecessary.

---

## Conversation Memory

The application maintains bounded in-session conversation memory.

Memory supports:

- Follow-up interpretation
- Query rewriting
- Conversation history

Conversation memory remains separate from the permanent document knowledge base.

CLI commands include:

```text
/history
/clear
```

The web interface also supports starting a new session.

---

# Smart Folder Synchronization

LocalMind includes incremental knowledge-base synchronization for the managed document folder.

The synchronization engine detects:

- New files
- Modified files
- Deleted files
- Unchanged files

Instead of rebuilding the complete knowledge base every time, only required changes are processed.

```text
NEW
 |
 v
Detect
 |
 v
Chunk + Embed + Store


MODIFIED
 |
 v
Detect Change
 |
 v
Replace Indexed Content


DELETED
 |
 v
Detect Deletion
 |
 v
Remove Stored Chunks
```

File changes are tracked using SHA-256 hashes.

## Important Sync Scope

The managed Smart Folder Sync currently operates on TXT documents in the configured local raw-data folder.

This is separate from browser-selected external document upload:

| Knowledge Source | Supported Formats |
|---|---|
| Managed Smart Folder Sync | TXT |
| Browser-selected external upload | TXT, PDF, DOCX |

This separation is intentional.

Externally uploaded sources are preserved independently and are not treated as managed Smart Folder files.

Synchronization can be triggered through:

- CLI
- REST API
- Web interface
- Agent tool

CLI:

```text
/sync
```

---

# Bounded Local Agent

LocalMind includes a controlled local agent layer in addition to direct RAG interaction.

The agent is intentionally bounded.

It does not execute arbitrary system actions and does not dynamically invent tools.

Instead, it selects from an explicitly registered local tool set.

Current tools:

```text
knowledge_search
knowledge_status
knowledge_sync
```

## knowledge_search

Delegates document questions to the RAG Assistant while preserving:

- Retrieved sources
- Confidence information
- Evidence coverage
- Query rewrite metadata
- Retrieved-document diagnostics

## knowledge_status

Returns deterministic knowledge-base information such as:

- Indexed chunk count
- Unique source count
- Source filenames
- Local-only state

This operation does not require an LLM generation call.

## knowledge_sync

Triggers managed Smart Folder synchronization and reports:

- New files
- Modified files
- Deleted files
- Unchanged files
- Inserted chunks
- Deleted chunks
- Whether changes occurred

---

# Agent Architecture

```text
                 USER
                   |
                   v
            +-------------+
            | Local Agent |
            +-------------+
                   |
                   v
            Intent Analysis
                   |
                   v
             Tool Selection
                   |
                   v
            +-------------+
            |Tool Registry|
            +-------------+
              /    |    \
             /     |     \
            v      v      v

      knowledge_ knowledge_ knowledge_
        search     status      sync
          |                    |
          v                    v
    RAG Assistant          Smart Sync
          |
          v
    Retrieval Pipeline
          |
          v
    Local Phi-4 Mini
          |
          v
      Agent Result
          |
          v
    Execution Trace
```

The agent adds controlled orchestration without replacing the core RAG architecture.

---

# Agent Execution Trace

Agent Mode exposes structured execution information such as:

- Detected intent
- Selected tool
- Decision confidence
- Decision reason
- Execution steps
- Tool result
- Execution latency
- Local-only state

Example:

```text
Intent:
knowledge_status

Selected Tool:
knowledge_status

Execution Steps:
✓ Intent analysis
✓ Tool selection
✓ Tool execution
✓ Response assembly
```

This makes agent behavior inspectable instead of operating as an opaque autonomous system.

---

# LocalMind Web Interface

LocalMind includes a custom private knowledge workspace.

The interface provides two primary modes.

## Chat Mode

Direct access to the grounded RAG pipeline:

```text
Question
   |
   v
Retrieval
   |
   v
Confidence
   |
   v
Grounded Generation
   |
   v
Answer + Sources
```

## Agent Mode

Routes requests through the bounded local agent:

```text
Request
   |
   v
Intent Analysis
   |
   v
Tool Selection
   |
   v
Tool Execution
   |
   v
Traceable Result
```

The interface includes:

- Library source list
- TXT/PDF/DOCX document upload
- Managed knowledge-base synchronization
- Session clearing
- Local runtime status
- Confidence information
- Evidence coverage
- Trusted source chips
- Answer Details
- Retrieval diagnostics
- Supporting passages
- Agent activity trace

---

# Retrieval Diagnostics

LocalMind exposes retrieval information without cluttering the primary answer interface.

Example:

```text
Confidence
HIGH

Evidence
100%

Top Match
0.6291

Match Gap
0.1934

Sources Used
1/3
```

Query rewriting can also be inspected when applicable.

Retrieved documents include source information and similarity scores.

---

# Source Inspection

Users can inspect the local evidence supporting a generated answer.

Example:

```text
Source:
stm32_notes.txt

Similarity:
0.6360

Supporting Passage:
STM32, STMicroelectronics tarafından geliştirilen
ARM tabanlı mikrodenetleyici ailesidir.
```

This provides visibility into why a response was generated.

---

# System Architecture

```text
                 LOCALMIND

                    |
       +------------+------------+
       |                         |
       v                         v

 Managed TXT Folder       Selected Documents
       |                  TXT / PDF / DOCX
       v                         |
   Smart Sync                    |
       |                         |
       +------------+------------+
                    |
                    v
             Document Loader
                    |
                    v
                Chunker
                    |
                    v
         Qwen3 Embedding 0.6B
                    |
                    v
                  SQLite
                    |
                    v
               Retrieval
                    |
                    v
             Query Rewriter
                    |
                    v
         Evidence Evaluation
                    |
                    v
          Confidence Engine
                    |
                    v
        Adaptive Context Filter
                    |
             +------+------+
             |             |
            LOW        SUFFICIENT
             |             |
             v             v
          Fallback    Grounded Prompt
                           |
                           v
                     Phi-4 Mini
                  Microsoft Foundry
                       Local
                           |
                           v
                  Answer + Sources
```

---

# Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Local AI Runtime | Microsoft Foundry Local |
| Chat Model | Phi-4 Mini |
| Embedding Model | Qwen3 Embedding 0.6B |
| Database | SQLite |
| Similarity | Cosine Similarity |
| API | FastAPI |
| Web Server | Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest |
| Interfaces | Web UI, REST API, CLI |
| Agent | Bounded local tool-based agent |

---

# Project Structure

```text
Local-RAG-AI-Assistant/
|
|-- data/
|   `-- raw/
|
|-- database/
|   `-- rag.db
|
|-- evaluation/
|
|-- src/
|   |-- agent.py
|   |-- agent_decision.py
|   |-- agent_models.py
|   |-- agent_tools.py
|   |-- api.py
|   |-- assistant.py
|   |-- chunker.py
|   |-- cli.py
|   |-- confidence.py
|   |-- config.py
|   |-- database.py
|   |-- document_loader.py
|   |-- embeddings.py
|   |-- evaluation.py
|   |-- file_sync.py
|   |-- ingestion.py
|   |-- knowledge_tools.py
|   |-- llm.py
|   |-- logging_config.py
|   |-- memory.py
|   |-- prompts.py
|   |-- query_rewriter.py
|   |-- retrieval.py
|   |-- similarity.py
|   |-- utils.py
|   `-- web.py
|
|-- tests/
|
|-- web/
|   |-- index.html
|   |-- styles.css
|   `-- app.js
|
|-- main.py
|-- run_evaluation.py
|-- run_tests.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

---

# Installation

## Requirements

The project has been validated with:

```text
Python 3.13.14
```

Create and activate a virtual environment.

Windows PowerShell example:

```powershell
python -m venv venv

.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The final dependency set includes the local Foundry runtime SDK, FastAPI stack, testing dependencies, multipart upload support, and document parsers for PDF and DOCX.

---

# Running LocalMind

## Web Application

Start the FastAPI application:

```powershell
python -m uvicorn src.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## CLI

Start the CLI assistant:

```powershell
python main.py --chat
```

Available commands include:

```text
/clear
/history
/sync
```

---

# REST API

Core endpoints include:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | LocalMind web interface |
| GET | `/health` | Local runtime/application status |
| POST | `/chat` | Direct grounded RAG query |
| POST | `/agent` | Bounded local agent request |
| GET | `/history` | Conversation history |
| DELETE | `/history` | Clear conversation memory |
| POST | `/sync` | Managed Smart Folder synchronization |
| POST | `/knowledge/files` | Import a selected TXT/PDF/DOCX document |
| GET | `/knowledge/sources` | Retrieve indexed Library sources |

---

# Privacy and Security

LocalMind follows a local-first design.

The RAG workflow is designed so that:

- Document processing occurs locally
- Embeddings are generated locally
- Phi-4 Mini inference runs locally through Microsoft Foundry Local
- SQLite knowledge-base storage remains local
- Conversation memory remains inside the application
- Agent tools operate only through explicitly registered capabilities
- External documents are processed only after explicit user selection

LocalMind does not intentionally provide the agent with unrestricted filesystem access.

## Upload Security

Browser uploads include several defensive controls:

- Filename sanitization
- Supported-extension validation
- 5 MB file-size limit
- UTF-8 validation for TXT
- Temporary-file cleanup
- Generic internal server errors
- Safe parser error responses

PDF/DOCX parser failures are mapped to safe client-facing messages so temporary operating-system paths are not exposed through API responses.

Example PDF error:

```text
The PDF document could not be read or contains no extractable text.
```

Example DOCX error:

```text
The DOCX document could not be read or contains no extractable text.
```

---

# Offline Verification

The final system was tested with internet connectivity disabled.

Validated scenarios included:

## Grounded Knowledge Query

```text
Question:
STM32 nedir?

Result:
Correct grounded answer

Confidence:
HIGH

Evidence:
100%

Source:
stm32_notes.txt
```

## Unsupported General-Knowledge Query

```text
Question:
Fransa'nın başkenti nedir?

Result:
Bu bilgi mevcut yerel belgelerde bulunamadı.

Confidence:
LOW

Evidence:
0%
```

The application did not use the model's unrestricted general knowledge to answer `Paris`.

## Agent Status

```text
Bilgi tabanında kaç kaynak var?
```

The deterministic `knowledge_status` tool successfully reported the local knowledge-base state.

## Agent Synchronization

```text
Bilgi tabanını senkronize et.
```

The `knowledge_sync` tool successfully executed the local synchronization workflow.

---

# Multi-Format Live Validation

TXT, PDF, and DOCX ingestion were validated through the running application.

## TXT

An external TXT document containing a unique test identifier was uploaded, indexed, retrieved, and preserved after managed synchronization.

## PDF

A text-based PDF was uploaded and successfully used as grounded evidence.

Validated source:

```text
external/localmind_pdf_test.pdf
```

The assistant correctly answered a question using the extracted PDF text.

## DOCX

A DOCX document was uploaded and successfully used as grounded evidence.

Validated source:

```text
external/localmind_docx_test.docx
```

The assistant correctly answered a question using extracted paragraph text.

---

# Evaluation Framework

LocalMind includes a dedicated evaluation framework for retrieval and end-to-end RAG quality.

Run:

```powershell
python run_evaluation.py
```

Evaluation covers:

- Retrieval hit rate
- Mean Reciprocal Rank
- Retrieval latency
- Source correctness
- Fallback behavior
- Query rewriting
- Confidence classification
- Grounding behavior
- Supported questions
- Unsupported questions
- Contextual follow-ups

---

# Final Evaluation Results

## Retrieval Evaluation

```text
Retrieval Cases        : 22
Hit Rate               : 100%
MRR                    : 0.9773
Warm Retrieval Average : 0.7303 s
Cold Start             : 8.8319 s
```

## End-to-End Quality Evaluation

```text
E2E Cases              : 9/9 passed
Source Accuracy        : 100%
Fallback Accuracy      : 100%
Rewrite Accuracy       : 100%
Confidence Accuracy    : 100%
Grounding Accuracy     : 100%
Overall Quality Score  : 100%

QUALITY GATE           : PASSED
```

The evaluation demonstrates that the tested system:

- Retrieves expected sources reliably
- Rejects unsupported questions
- Preserves trusted source attribution
- Rewrites contextual follow-up questions when needed
- Correctly evaluates retrieval confidence
- Prevents unsupported model knowledge from bypassing grounding rules

These metrics describe the project's defined evaluation set and should not be interpreted as universal RAG accuracy.

---

# Automated Testing

Run the complete test suite:

```powershell
python -m pytest tests -q
```

Final regression result:

```text
451 passed in 5.45s
```

The final suite includes coverage across:

- Database operations
- Embedding validation
- Similarity calculations
- TXT/PDF/DOCX document loading
- Document ingestion
- Chunking
- Retrieval
- Confidence policies
- Evidence evaluation
- Prompt construction
- Conversation memory
- Query rewriting
- Smart synchronization
- RAG assistant behavior
- Agent behavior
- Agent tools
- API behavior
- Upload validation
- Privacy/error handling
- Evaluation logic

---

# Reproducibility Check

The final project was also validated in a newly created virtual environment rather than relying only on the development environment.

The validation process included:

```text
Fresh virtual environment
        |
        v
Install requirements.txt
        |
        v
Run complete test suite
        |
        v
451 tests passed
```

This verifies that the final dependency specification is sufficient for the tested project configuration.

---

# Performance Notes

Local execution introduces a performance trade-off.

Warm semantic retrieval averaged approximately:

```text
0.7303 s
```

Cold-start retrieval during evaluation:

```text
8.8319 s
```

Full answers requiring local Phi-4 Mini generation can take longer depending on hardware, prompt size, and model state.

Observed local answer generation was commonly in the multi-second range and could exceed ten seconds.

Deterministic agent operations such as `knowledge_status` are substantially faster because they do not require language-model generation.

The project intentionally prioritizes:

```text
Privacy
Grounding
Local execution
Explainability
Control
```

over cloud-level inference latency.

---

# Reliability Design

LocalMind uses several independent mechanisms to reduce unsupported answers.

## Retrieval Thresholding

Weak semantic matches can be rejected.

## Evidence Evaluation

Vector similarity alone is not treated as proof that a document contains the answer.

## Adaptive Context Filtering

Weak Top-K results can be removed before generation.

## Confidence Gating

LOW-confidence retrieval can prevent unsupported generation.

## Grounded Prompting

The local model is instructed to remain within supplied evidence.

## Controlled Fallback

Unsupported questions receive a deterministic fallback response.

## Trusted Source Attribution

Sources are generated from retrieval metadata rather than model-generated filenames.

## Query Rewrite Isolation

Conversation context is introduced only when useful for retrieval.

## Bounded Agent Tools

The agent operates only through explicitly registered local tools.

Together, these mechanisms form a defense-in-depth approach to grounded local AI.

---

# Current Status

## Core RAG

- [x] Local LLM integration
- [x] Local embedding generation
- [x] SQLite knowledge base
- [x] Document ingestion
- [x] TXT support
- [x] Text-based PDF support
- [x] DOCX paragraph-text support
- [x] Chunking
- [x] Semantic retrieval
- [x] Cosine similarity ranking
- [x] Confidence-aware retrieval
- [x] Evidence coverage
- [x] Adaptive context filtering
- [x] Controlled fallback
- [x] Trusted source attribution
- [x] Conversation memory
- [x] Conversational query rewriting

## Knowledge Management

- [x] Managed Smart Folder Sync
- [x] New-file detection
- [x] Modified-file detection
- [x] Deleted-file detection
- [x] Incremental indexing
- [x] SHA-256 change tracking
- [x] Source provenance
- [x] External document upload
- [x] Persistent Library source listing

## Interfaces

- [x] CLI
- [x] FastAPI REST API
- [x] LocalMind web interface
- [x] Chat Mode
- [x] Agent Mode
- [x] Retrieval diagnostics
- [x] Supporting passage inspection
- [x] Agent execution trace

## Agent

- [x] Bounded local agent
- [x] Intent analysis
- [x] Tool selection
- [x] Tool registry
- [x] `knowledge_search`
- [x] `knowledge_status`
- [x] `knowledge_sync`
- [x] Structured execution trace

## Quality

- [x] Automated test suite
- [x] 451 final regression tests
- [x] Retrieval evaluation
- [x] End-to-end quality evaluation
- [x] Offline verification
- [x] Multi-format live validation
- [x] Clean-environment reproducibility check
- [x] Upload privacy/error hardening
- [x] Quality Gate

---

# Current Limitations

The current version intentionally has a defined scope.

- Scanned/image-only PDF files are not supported.
- OCR is not included.
- DOCX extraction focuses on paragraph text.
- DOCX tables, headers, footers, and complex structures are not guaranteed to be indexed.
- Browser upload is limited to 5 MB per document.
- Managed Smart Folder Sync currently supports TXT files.
- The agent does not have unrestricted filesystem or operating-system access.
- Answers are intentionally limited to evidence available in the indexed knowledge base.
- Local LLM generation is slower than typical cloud inference on some hardware.

These limitations are deliberate and help keep the current architecture controlled, testable, and aligned with the project's local RAG objective.

---

# Possible Future Improvements

Potential future work could include:

- OCR for scanned documents
- Richer DOCX structure extraction
- Metadata-aware retrieval
- Hybrid semantic and lexical retrieval
- Reranking experiments
- Larger evaluation datasets
- Additional retrieval metrics
- Advanced chunking experiments
- Local inference performance optimization
- Packaged desktop distribution

These are future extensions and are not required for the completed project.

---

# Design Philosophy

LocalMind is not designed simply as a language model connected to document search.

The system is designed to be:

**Local**

Document processing, embeddings, retrieval, and model inference are built around local execution.

**Grounded**

Answers should be supported by indexed local evidence.

**Transparent**

Retrieval confidence, evidence coverage, sources, supporting passages, query rewriting, and agent execution can be inspected.

**Controlled**

The agent operates through explicitly registered local tools.

**Private**

External documents are processed only after explicit user selection, and the core knowledge workflow remains local.

**Testable**

Core behavior is protected by automated regression tests.

**Measurable**

Retrieval and end-to-end quality are evaluated using explicit metrics.

**Maintainable**

Retrieval, generation, memory, synchronization, ingestion, prompting, confidence, API, agent logic, and evaluation are separated into dedicated modules.

---

# Example

## Supported Knowledge

```text
User:
STM32 nedir?

LocalMind:
STM32, STMicroelectronics tarafından geliştirilen
ARM tabanlı mikrodenetleyici ailesidir.

Source:
stm32_notes.txt

Confidence:
HIGH
```

## Unsupported Knowledge

```text
User:
Fransa'nın başkenti nedir?

LocalMind:
Bu bilgi mevcut yerel belgelerde bulunamadı.

Confidence:
LOW
```

## Agent Request

```text
User:
Bilgi tabanında kaç kaynak var?

Agent:
Returns deterministic local knowledge-base status.

Intent:
knowledge_status

Tool:
knowledge_status
```

These examples demonstrate the central design goal of LocalMind:

> Local model capability is constrained by retrieved evidence and explicit application logic rather than being treated as an unrestricted source of truth.

---

# Project Status

**Technical development: complete**

**Final automated regression suite: 451 passed**

**Evaluation quality gate: passed**

**Offline verification: passed**

**TXT/PDF/DOCX live ingestion validation: passed**

**Clean-environment reproducibility: passed**

The project is currently in final documentation, presentation, and submission preparation.

---

# License

This project is currently intended for educational and experimental use.

---

# Author

**Pelin Özbilgin**

Computer Engineering Student

Areas of interest:

- Artificial Intelligence
- Retrieval-Augmented Generation
- Local AI Systems
- Embedded Systems
- Software Development