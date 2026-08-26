# Local RAG AI Assistant

A privacy-focused, fully local Retrieval-Augmented Generation (RAG) assistant built with Python, Microsoft Foundry Local, Phi-4 Mini, local embeddings, and SQLite.

The system retrieves relevant information from a local knowledge base, evaluates retrieval confidence, filters noisy context, handles conversational follow-up questions, and generates grounded answers using a locally running language model.

The project is designed around three core principles:

- Local-first AI
- Grounded and explainable responses
- Measurable RAG quality

---

## Overview

Local RAG AI Assistant allows users to query local documents without relying on a cloud-hosted language model.

Instead of sending documents and questions to an external AI service, the system performs the main RAG pipeline locally:

1. Local documents are ingested.
2. Documents are divided into chunks.
3. Embeddings are generated locally.
4. Chunks and embeddings are stored in SQLite.
5. User questions are converted into embeddings.
6. Semantically relevant chunks are retrieved.
7. Retrieval confidence is evaluated.
8. Weak or noisy context is filtered.
9. Relevant context is sent to a local language model.
10. The answer is generated using only trusted local evidence.

If sufficient evidence cannot be found, the system refuses to generate an unsupported answer and returns a controlled fallback response.

---

## Key Features

### Fully Local RAG Pipeline

The main RAG workflow runs locally using:

- Python
- Microsoft Foundry Local
- Phi-4 Mini
- Local embedding model
- SQLite

This architecture reduces dependency on external AI APIs and provides greater control over local data.

---

### Semantic Retrieval

User questions are converted into embeddings and compared against document embeddings using semantic similarity.

The retrieval layer returns the most relevant document chunks rather than relying only on keyword matching.

---

### Confidence-Aware Retrieval

The system does not blindly trust every retrieved result.

A dedicated confidence engine evaluates retrieval quality using signals such as:

- Top similarity score
- Second-best similarity score
- Score separation
- Evidence coverage
- Selected context count
- Filtered context count

Retrieval confidence is classified into:

- `HIGH`
- `MEDIUM`
- `LOW`

Low-confidence retrieval can prevent the language model from being called.

---

### Evidence Coverage

Semantic similarity alone does not guarantee that a retrieved document directly supports a question.

The system therefore evaluates how much direct evidence exists between the current question and retrieved context.

This provides an additional protection layer against semantically similar but factually irrelevant retrieval results.

---

### Adaptive Context Filtering

Instead of automatically sending every Top-K result to the language model, the assistant dynamically selects the strongest context.

For example:

```text
Retrieved scores:

0.80
0.74
0.31

Selected context:

0.80
0.74

Filtered noise:

0.31
```

This reduces irrelevant context and improves grounding.

---

### Grounded Answer Generation

The prompt layer explicitly instructs the local model to:

- Use only supplied local document evidence
- Avoid outside knowledge
- Avoid unsupported assumptions
- Avoid inventing source names
- Return a controlled fallback when evidence is insufficient

Source rendering is handled by the application rather than being trusted to the language model.

---

### Controlled Fallback Protection

When local documents do not provide enough evidence, the assistant returns:

```text
Bu bilgi mevcut yerel belgelerde bulunamadı.
```

This prevents the system from answering unrelated questions using unsupported model knowledge.

Example:

```text
Question:
Fransa'nın başkenti nedir?

Answer:
Bu bilgi mevcut yerel belgelerde bulunamadı.
```

even though the underlying language model may know the answer.

This behavior is intentional and is part of the grounding architecture.

---

### Trusted Source Attribution

Sources are derived from the application's selected retrieval context.

Example:

```text
RAG CEVABI
==================================================
STM32, STMicroelectronics tarafından geliştirilen
ARM tabanlı mikrodenetleyici ailesidir.

Kaynaklar:
- stm32_notes.txt
```

The language model is not responsible for inventing or selecting source filenames.

---

### Conversational Query Rewriting

The assistant supports contextual follow-up questions.

Example:

```text
User:
STM32 nedir?

User:
Peki PWM ne işe yarar?
```

The second question can be transformed internally into a retrieval query containing relevant conversational context.

Example:

```text
Original Query:
Peki PWM ne işe yarar?

Retrieval Query:
STM32 nedir? Peki PWM ne işe yarar?
```

This improves retrieval performance for follow-up questions while preventing unrelated standalone questions from unnecessarily inheriting previous context.

---

### Conversation Memory

The assistant maintains bounded in-session conversation memory.

Memory is used for:

- Follow-up interpretation
- Query rewriting
- Conversation history

Available CLI commands include:

```text
/history
/clear
```

Conversation memory remains separate from the permanent document knowledge base.

---

### Smart Folder Synchronization

The knowledge base supports synchronization of local document changes.

The synchronization layer can identify:

- New files
- Modified files
- Deleted files
- Unchanged files

Example:

```text
KNOWLEDGE BASE SYNC
==================================================
New files      : 1
Modified files : 1
Deleted files  : 0
Unchanged      : 3
Inserted chunks: 4
Deleted chunks : 2
```

Synchronization can be triggered from the CLI using:

```text
/sync
```

This avoids unnecessarily rebuilding the entire knowledge base when only a subset of files changes.

---

## Architecture

```text
                    LOCAL DOCUMENTS
                           |
                           v
                 +-------------------+
                 | Smart Folder Sync |
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | Document Ingestion|
                 +-------------------+
                           |
                           v
                 +-------------------+
                 |    Chunking       |
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | Local Embeddings  |
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | SQLite Vector Data|
                 +-------------------+

                           |
                           |
                    USER QUESTION
                           |
                           v
                 +-------------------+
                 | Query Rewriter    |
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | Semantic Retrieval|
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | Evidence Coverage |
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | Confidence Engine |
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | Adaptive Context  |
                 | Filtering         |
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | Grounded Prompt   |
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | Phi-4 Mini        |
                 | Foundry Local     |
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | Fallback + Source |
                 | Validation        |
                 +-------------------+
                           |
                           v
                       ANSWER
```

---

## Project Structure

```text
Local-RAG-AI-Assistant/
|
|-- data/
|   `-- ...
|
|-- database/
|   `-- rag.db
|
|-- evaluation/
|   `-- evaluation_cases.py
|
|-- src/
|   |-- assistant.py
|   |-- cli.py
|   |-- confidence.py
|   |-- config.py
|   |-- database.py
|   |-- embeddings.py
|   |-- evaluation.py
|   |-- file_sync.py
|   |-- ingestion.py
|   |-- llm.py
|   |-- logging_config.py
|   |-- memory.py
|   |-- prompts.py
|   |-- query_rewriter.py
|   `-- retrieval.py
|
|-- tests/
|   |-- test_assistant.py
|   |-- test_cli.py
|   |-- test_confidence.py
|   |-- test_database.py
|   |-- test_evaluation.py
|   |-- test_file_sync.py
|   |-- test_ingestion.py
|   |-- test_main.py
|   |-- test_memory.py
|   |-- test_prompts.py
|   |-- test_query_rewriter.py
|   `-- ...
|
|-- main.py
|-- run_evaluation.py
|-- requirements.txt
`-- README.md
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Local Model Runtime | Microsoft Foundry Local |
| Chat Model | Phi-4 Mini |
| Embedding Model | Qwen3 Embedding 0.6B |
| Database | SQLite |
| Retrieval | Cosine Similarity |
| Testing | Pytest |
| Interface | CLI |

---

## RAG Pipeline

### 1. Document Ingestion

Local documents are read and validated before being added to the knowledge base.

### 2. Chunking

Documents are divided into smaller overlapping chunks to improve retrieval precision.

### 3. Embedding Generation

Each chunk is converted into a vector representation using a locally running embedding model.

### 4. Local Storage

Document content, source information, and embeddings are stored in SQLite.

### 5. Query Processing

The user's question is cleaned and analyzed.

When necessary, conversational context is incorporated through query rewriting.

### 6. Semantic Retrieval

The query embedding is compared with stored document embeddings using cosine similarity.

### 7. Confidence Evaluation

Retrieved results are analyzed for semantic strength and supporting evidence.

### 8. Context Selection

Weak retrieval results are filtered before prompt construction.

### 9. Grounded Generation

Only selected local evidence is supplied to Phi-4 Mini.

### 10. Output Validation

Fallback behavior and trusted source attribution are applied before returning the final result.

---

## CLI Usage

Start the assistant with:

```bash
python main.py --chat
```

Example:

```text
Local RAG AI Assistant
--------------------------------------------------
Çıkmak için 'exit', 'quit', 'q' veya 'çıkış' yaz.
Komutlar: /clear, /history, /sync

Soru: STM32 nedir?
```

Example response:

```text
==================================================
RAG CEVABI
==================================================

STM32, STMicroelectronics tarafından geliştirilen
ARM tabanlı mikrodenetleyici ailesidir.

Kaynaklar:
- stm32_notes.txt
```

---

## Retrieval Transparency

The CLI exposes retrieval metadata for development and evaluation.

Example:

```text
==================================================
QUERY REWRITE
==================================================
Rewritten       : YES
Original Query  : Peki PWM ne işe yarar?
Retrieval Query : STM32 nedir? Peki PWM ne işe yarar?

==================================================
RETRIEVAL CONFIDENCE
==================================================
Level            : HIGH
Top Score        : 0.6835
Second Score     : 0.4053
Score Gap        : 0.2782
Evidence Coverage: 100.00%
Selected Context : 1/3
Filtered Noise   : 2

==================================================
RETRIEVED DOCUMENTS
==================================================
[1]
Source : stm32_notes.txt
Score  : 0.6835
```

This makes the system easier to debug and evaluate compared with a black-box RAG pipeline.

---

## Evaluation Framework

The project contains a dedicated retrieval evaluation framework.

Evaluation cases define:

```python
EvaluationCase(
    question="STM32 nedir?",
    expected_sources=(
        "stm32_notes.txt",
    ),
)
```

The framework measures retrieval behavior across multiple knowledge domains and harder cross-document questions.

Current evaluation areas include:

- STM32
- PID and line following
- RAG
- SQLite
- Foundry Local
- Cross-document retrieval

---

## Evaluation Metrics

The evaluation framework calculates metrics including:

### Hit Rate

Measures how often at least one expected source appears in the retrieved results.

### Mean Reciprocal Rank

Measures how highly the first relevant document is ranked.

```text
MRR = mean(1 / rank of first relevant result)
```

### Retrieval Latency

The evaluation framework measures:

- Cold-start latency
- Average retrieval latency
- Warm average retrieval latency
- Median latency
- Minimum latency
- Maximum latency

Separating cold-start and warm retrieval performance provides a more realistic view of local model behavior.

---

## Quality Gate

The project includes automated tests and evaluation checks designed to protect RAG quality during development.

Quality dimensions include:

```text
Retrieval Quality
Source Accuracy
Fallback Accuracy
Query Rewrite Behavior
Confidence Classification
Grounding Behavior
Conversation Memory
Database Integrity
Document Ingestion
Folder Synchronization
```

This means changes to retrieval, prompts, confidence policies, memory, or ingestion can be validated before being accepted into the stable system.

---

## Testing

Run the complete test suite with:

```bash
python -m pytest tests -q
```

For verbose output:

```bash
python -m pytest tests -v
```

Individual components can also be tested independently.

Examples:

```bash
python -m pytest tests/test_confidence.py -v
```

```bash
python -m pytest tests/test_assistant.py -v
```

```bash
python -m pytest tests/test_prompts.py -v
```

```bash
python -m pytest tests/test_query_rewriter.py -v
```

```bash
python -m pytest tests/test_file_sync.py -v
```

---

## Running Retrieval Evaluation

Run:

```bash
python run_evaluation.py
```

The evaluation report includes:

```text
Cases
Expected Sources
Retrieved Sources
Hit / Miss
Reciprocal Rank
Latency
Hit Rate
MRR
Cold Start
Warm Average Retrieval
Median Retrieval
Fastest Retrieval
Slowest Retrieval
```

This makes retrieval improvements measurable instead of relying only on manual testing.

---

## Reliability and Safety Design

Several mechanisms are used to reduce unsupported answers.

### Retrieval Thresholding

Weak semantic matches can be rejected before generation.

### Evidence Validation

High vector similarity alone is not treated as sufficient proof that a document answers the question.

### Adaptive Context Selection

Irrelevant Top-K results are removed before generation.

### Strict Grounding Prompt

The model is instructed to rely exclusively on supplied local evidence.

### Controlled Fallback

Unsupported questions receive a deterministic fallback response.

### Trusted Source Handling

Source attribution is controlled by application logic.

### Query Rewrite Isolation

Conversational context is added only when a question is determined to be a follow-up.

Together, these layers create a defense-in-depth approach to RAG grounding.

---

## Privacy

The project follows a local-first architecture.

The RAG pipeline is designed so that:

- Documents remain in the local knowledge base.
- Embeddings are generated locally.
- The chat model runs locally through Foundry Local.
- Conversation memory is maintained inside the application session.
- SQLite stores the knowledge base locally.

This architecture is suitable for experimentation with private or offline document-assistant workflows where cloud-based inference may not be desirable.

---

## Current Development Status

The core RAG architecture currently includes:

- [x] Local LLM integration
- [x] Local embedding generation
- [x] SQLite knowledge base
- [x] Document ingestion
- [x] Semantic retrieval
- [x] Cosine similarity ranking
- [x] Confidence-aware retrieval
- [x] Evidence coverage analysis
- [x] Adaptive context filtering
- [x] Controlled fallback behavior
- [x] Trusted source attribution
- [x] Conversation memory
- [x] Conversational query rewriting
- [x] Smart folder synchronization
- [x] Retrieval evaluation framework
- [x] Automated test suite
- [x] Retrieval latency measurement
- [x] Quality validation

---

## Planned Improvements

Future development may include:

- [ ] PDF document support
- [ ] DOCX document support
- [ ] Richer document metadata
- [ ] Metadata-aware retrieval
- [ ] Improved chunking strategies
- [ ] Larger evaluation datasets
- [ ] Additional retrieval metrics
- [ ] Hybrid semantic and lexical retrieval
- [ ] Reranking experiments
- [ ] Graphical user interface
- [ ] Packaged local desktop application

---

## Design Philosophy

The goal of this project is not simply to connect a language model to a vector search system.

The project focuses on building a RAG pipeline that is:

**Local**  
Models and document processing are designed around local execution.

**Grounded**  
Answers should be supported by retrieved evidence.

**Transparent**  
Retrieval scores, confidence, evidence coverage, and query rewriting can be inspected.

**Testable**  
Core behavior is protected by automated tests.

**Measurable**  
Retrieval quality and latency are evaluated using explicit metrics.

**Maintainable**  
Retrieval, generation, memory, synchronization, prompting, confidence, and evaluation are separated into dedicated modules.

---

## Example

```text
User:
STM32 nedir?

Assistant:
STM32, STMicroelectronics tarafından geliştirilen
ARM tabanlı mikrodenetleyici ailesidir.

Sources:
stm32_notes.txt

Confidence:
HIGH
```

Unsupported query:

```text
User:
Fransa'nın başkenti nedir?

Assistant:
Bu bilgi mevcut yerel belgelerde bulunamadı.

Confidence:
LOW
```

The second example demonstrates an important property of the system: the assistant prioritizes document grounding over the language model's general knowledge.

---

## License

This project is currently intended for educational and experimental use.

---

## Author

**Pelin Özbilgin**

Computer Engineering Student

Areas of interest:

- Artificial Intelligence
- Retrieval-Augmented Generation
- Local AI Systems
- Embedded Systems
- Software Development