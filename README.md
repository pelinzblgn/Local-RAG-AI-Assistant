# Local RAG AI Assistant

A fully local Retrieval-Augmented Generation (RAG) assistant built with Microsoft Foundry Local SDK, Phi-4 Mini, Qwen3 Embedding, SQLite, and semantic document retrieval.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Tests](https://img.shields.io/badge/Tests-126%20Passed-success)
![Status](https://img.shields.io/badge/Phase-2%20In%20Progress-yellow)

The project implements an end-to-end local RAG pipeline capable of ingesting documents, generating embeddings locally, retrieving semantically relevant context, and producing grounded answers using a locally running language model.

No external cloud inference API is required for the core RAG workflow.

---

# Features

- Fully local RAG execution
- Microsoft Foundry Local SDK
- Phi-4 Mini local chat model
- Qwen3 local embedding model
- SQLite-based document and embedding storage
- Semantic similarity retrieval
- Configurable Top-K retrieval
- Smart character-based chunking with overlap
- Prompt grounding
- Source attribution
- Retrieval scores and metadata
- Conversation memory
- `/history` command
- `/clear` command
- Local model warm-up
- Incremental document ingestion
- Knowledge-base reset
- Knowledge-base statistics
- Retrieval evaluation framework
- Hit Rate and Mean Reciprocal Rank evaluation
- Cold-start and warm retrieval latency measurement
- Structured logging
- Modular architecture
- Automated unit tests

---

# Project Structure

```text
Local-RAG-AI-Assistant/
│
├── data/
│   └── raw/
│       ├── foundry_local_notes.txt
│       ├── pid_notes.txt
│       ├── rag_notes.txt
│       ├── sqlite_notes.txt
│       └── stm32_notes.txt
│
├── database/
│   └── rag.db
│
├── evaluation/
│   └── evaluation_cases.py
│
├── src/
│   ├── __init__.py
│   ├── assistant.py
│   ├── chunker.py
│   ├── cli.py
│   ├── config.py
│   ├── database.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── ingestion.py
│   ├── llm.py
│   ├── logging_config.py
│   ├── memory.py
│   ├── prompts.py
│   ├── retrieval.py
│   ├── similarity.py
│   └── utils.py
│
├── tests/
│   ├── __init__.py
│   ├── test_assistant.py
│   ├── test_chunking.py
│   ├── test_cli.py
│   ├── test_database.py
│   ├── test_embeddings.py
│   ├── test_evaluation.py
│   ├── test_ingestion.py
│   ├── test_llm.py
│   ├── test_main.py
│   ├── test_memory.py
│   ├── test_prompts.py
│   ├── test_retrieval.py
│   └── test_similarity.py
│
├── .gitignore
├── main.py
├── requirements.txt
├── run_evaluation.py
├── run_tests.py
└── README.md
```

> Runtime-generated files such as the SQLite database, virtual environment, Python cache files, and pytest cache are excluded from version control where appropriate.

---

# Architecture

```text
Local TXT Documents
        │
        ▼
Document Loader
        │
        ▼
Smart Chunking
        │
        ▼
Qwen3 Embedding Model
        │
        ▼
SQLite Knowledge Base
        │
        ▼
Semantic Similarity Search
        │
        ▼
Top-K Relevant Documents
        │
        ▼
Conversation Context
        │
        ▼
Grounded Prompt Builder
        │
        ▼
Phi-4 Mini
        │
        ▼
Grounded Local Answer
        │
        ▼
Sources + Retrieval Metadata
```

---

# Technologies

| Technology | Purpose |
|---|---|
| Python 3.13 | Core application |
| Microsoft Foundry Local SDK | Local AI runtime |
| Phi-4 Mini | Local language model |
| Qwen3 Embedding | Local embedding generation |
| SQLite | Local knowledge-base storage |
| PyTest | Automated testing |

---

# Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd Local-RAG-AI-Assistant
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

## 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Knowledge Base

Place source `.txt` documents inside:

```text
data/raw/
```

The ingestion pipeline:

1. discovers source documents,
2. loads their content,
3. splits them into overlapping chunks,
4. generates local embeddings,
5. stores chunks and embeddings in SQLite.

---

# Application Commands

## Start Chat

```bash
python main.py --chat
```

Running without an explicit mode also starts the chat interface:

```bash
python main.py
```

---

## Incremental Ingestion

```bash
python main.py --ingest
```

Processes the source documents while preserving the existing knowledge base.

---

## Reset Knowledge Base

```bash
python main.py --reset
```

Rebuilds the local knowledge base from the source documents.

---

## Knowledge Base Statistics

```bash
python main.py --stats
```

Displays information about the current local knowledge base.

---

# Chat Commands

While the assistant is running:

```text
/history
```

Displays the current conversation history.

```text
/clear
```

Clears the current conversation memory.

Exit commands:

```text
exit
quit
q
çık
çıkış
```

---

# Conversation Memory

The assistant maintains a bounded conversation history during the active session.

Recent conversation turns can be incorporated into retrieval queries, allowing follow-up questions to retain useful context without allowing the conversation history to grow indefinitely.

Example:

```text
User:
STM32 nedir?

User:
Peki onun PWM ile ilişkisi nedir?
```

The second query can use recent conversational context to improve retrieval.

---

# Model Warm-Up

Before an interactive chat session starts, the local embedding and chat models are initialized.

Example:

```text
Yerel modeller hazırlanıyor...

Embedding modeli warm-up tamamlandı.
Chat modeli warm-up tamamlandı.

Modeller hazır.
```

This moves most model initialization cost to application startup instead of the first user query.

---

# Retrieval Metadata

The CLI displays the documents selected by semantic retrieval together with their similarity scores.

Example:

```text
RETRIEVED DOCUMENTS
==================================================

[1]
Source : stm32_notes.txt
Score  : 0.6540

[2]
Source : pid_notes.txt
Score  : 0.3887

[3]
Source : rag_notes.txt
Score  : 0.3803
```

This makes the retrieval process observable and easier to evaluate.

---

# Retrieval Evaluation

The project includes a dedicated evaluation framework for measuring retrieval quality and performance.

Run:

```bash
python run_evaluation.py
```

The current evaluation set contains seven retrieval cases covering the sample knowledge base.

Current observed result:

```text
Cases                  : 7
Hit Rate               : 100.00%
MRR                    : 1.0000
Cold Start             : 10.0476 s
Avg Retrieval Time     : 1.8100 s
Warm Avg Retrieval     : 0.4371 s
Median Retrieval       : 0.4319 s
Fastest Retrieval      : 0.4261 s
Slowest Retrieval      : 10.0476 s
```

The results show that the expected source was successfully retrieved for every current evaluation case and ranked first in each case.

The latency measurements also separate model cold-start cost from normal warm retrieval performance.

> These measurements correspond to the current small evaluation dataset and local development environment and should not be interpreted as general benchmark results.

---

# Testing

Run the complete automated test suite:

```bash
python -m pytest tests -q
```

Current status:

```text
126 passed
```

For verbose output:

```bash
python -m pytest tests -v
```

The test suite covers major components including:

- chunking
- database operations
- embedding validation
- ingestion
- semantic retrieval
- similarity calculations
- prompt construction
- local LLM integration
- conversation memory
- assistant orchestration
- CLI behavior
- application modes
- retrieval evaluation

---

# Example

```text
Soru: STM32 nedir?

RAG CEVABI
==================================================

STM32, STMicroelectronics tarafından geliştirilen
ARM tabanlı mikrodenetleyici ailesidir.

Kaynaklar:
- stm32_notes.txt

==================================================
RETRIEVED DOCUMENTS
==================================================

[1]
Source : stm32_notes.txt
Score  : 0.6540
```

---

# Current Capabilities

The current implementation supports:

- local TXT document ingestion
- overlapping text chunking
- local embedding generation
- SQLite knowledge-base storage
- semantic retrieval
- configurable Top-K selection
- grounded local generation
- source attribution
- retrieval metadata
- conversational context
- session history management
- embedding and chat model warm-up
- incremental ingestion
- full knowledge-base reset
- knowledge-base statistics
- retrieval quality evaluation
- retrieval latency evaluation
- automated unit testing

---

# Development Roadmap

## Phase 1 — Core Local RAG Engine

Completed.

- Local project architecture
- Configuration management
- SQLite storage
- Document loading
- Embedding generation
- Cosine similarity
- Semantic retrieval
- Prompt construction
- Local Phi-4 Mini integration
- RAG assistant orchestration
- Initial automated testing

## Phase 2 — Reliability, Memory and Evaluation

In progress.

Completed so far:

- Improved chunking with overlap
- Conversation memory
- Context-aware follow-up retrieval
- Conversation history commands
- Retrieval metadata visibility
- Incremental ingestion
- Knowledge-base reset
- Knowledge-base statistics
- Model warm-up
- Retrieval evaluation framework
- Hit Rate evaluation
- Mean Reciprocal Rank evaluation
- Cold-start and warm latency measurements
- Expanded automated test coverage

Remaining Phase 2 work will be completed and validated before Phase 3 begins.

## Phase 3 — Application Layer

Planned.

Potential Phase 3 work includes:

- API/backend layer
- Web user interface
- File upload workflow
- Additional document formats
- Retrieval improvements
- Deployment and packaging improvements

---

# Privacy

The core RAG pipeline is designed to operate locally.

Documents, embeddings, retrieval operations, and language-model inference can remain on the local machine without requiring a cloud inference API.

---

# Project Status

**Phase 2 — In Progress**

Core RAG functionality is operational.

Current validation status:

```text
Automated Tests : 126 passed
Evaluation Cases: 7
Hit Rate        : 100%
MRR             : 1.0000
```

---

Developed as part of the Microsoft AI Summer School.