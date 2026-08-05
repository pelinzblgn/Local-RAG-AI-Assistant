# Local RAG AI Assistant

A fully local Retrieval-Augmented Generation (RAG) assistant powered by Microsoft Foundry Local SDK, Phi-4 Mini and semantic document retrieval.
![Python](https://img.shields.io/badge/Python-3.13-blue)

![Tests](https://img.shields.io/badge/Tests-77%20Passed-success)

![License](https://img.shields.io/badge/License-MIT-green)

![Status](https://img.shields.io/badge/Phase-1%20Completed-brightgreen)

A fully local Retrieval-Augmented Generation (RAG) assistant built with Microsoft Foundry Local SDK.

The project retrieves relevant document chunks from a local SQLite database using semantic search and generates grounded answers with a local Phi-4 Mini language model.

---

# Features

- ✅ Fully local execution
- ✅ Microsoft Foundry Local SDK
- ✅ Phi-4 Mini
- ✅ Qwen3 Embedding
- ✅ SQLite Vector Store
- ✅ Semantic Retrieval
- ✅ Prompt Grounding
- ✅ Source Attribution
- ✅ Modular Architecture
- ✅ Comprehensive Unit Tests

---

# Project Structure

```
Local-RAG-AI-Assistant
│
├── data
│   └── raw
│       ├── stm32_notes.txt
│       ├── pid_notes.txt
│       ├── rag_notes.txt
│       ├── sqlite_notes.txt
│       └── foundry_local_notes.txt
│
├── database
│   └── rag.db
│
├── src
│   ├── assistant.py
│   ├── config.py
│   ├── database.py
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── llm.py
│   ├── logging_config.py
│   ├── prompts.py
│   ├── retrieval.py
│   ├── similarity.py
│   └── utils.py
│
├── tests
│
├── main.py
├── run_tests.py
├── requirements.txt
└── README.md
```

---

# Architecture

```
TXT Documents
        │
        ▼
Document Loader
        │
        ▼
Chunking
        │
        ▼
Embedding Generation
        │
        ▼
SQLite Database
        │
        ▼
Semantic Retrieval
        │
        ▼
Prompt Builder
        │
        ▼
Phi-4 Mini
        │
        ▼
Grounded Answer
```

---

# Technologies

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Foundry Local SDK | Local AI Runtime |
| Phi-4 Mini | Chat Model |
| Qwen3 Embedding | Embedding Model |
| SQLite | Vector Storage |
| PyTest | Testing |

---

# Installation

Create a virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Ingest Documents

Place your text files inside

```
data/raw
```

Then run

```bash
python -c "from src.logging_config import configure_logging; from src.ingestion import ingest_text_files; configure_logging(); ingest_text_files(reset_database=True)"
```

---

# Run

```bash
python main.py
```

---

# Run Tests

```bash
python run_tests.py
```

or

```bash
python -m pytest tests -v
```

---

# Current Capabilities

- TXT document ingestion
- Character-based chunking
- Embedding generation
- Semantic retrieval
- Prompt grounding
- Local answer generation
- Source listing
- Logging
- Unit testing

---

# Example Output

```
Question

Çizgi takip sisteminde hata nasıl hesaplanır?

↓

Retrieved Documents

pid_notes.txt

stm32_notes.txt

↓

Answer

Çizgi takip sisteminde hata, sensörlerin algıladığı çizgi
konumu ile hedef merkez arasındaki farktır.

Kaynaklar

pid_notes.txt

stm32_notes.txt
```

---

# Test Status

```
77 / 77 tests passed
```

---

# Roadmap

## Phase 1

- Local RAG Engine
- SQLite Storage
- Semantic Retrieval
- Prompt Builder
- Local LLM
- Assistant Class

Completed

## Phase 2

- Smart Chunking
- PDF Support
- DOCX Support
- Metadata
- Conversation Memory
- Streaming Responses

## Phase 3

- FastAPI Backend
- Web Interface
- Hybrid Search
- Production Deployment

---

# License

MIT License

---

Developed as part of the Microsoft AI Summer School.

Phase 1 Completed ✅