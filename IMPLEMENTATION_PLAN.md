# Implementation Plan: Repository Ingestion, Chunking, Embeddings, and Basic RAG

This document records the design and architecture implemented for the Day 3-4 milestone (completed 2026-08-27).

## Goal

Build an end-to-end repository ingestion and semantic RAG pipeline that can:
1. Scan and parse local software repositories (filtered by safety guardrails and inclusion/exclusion rules).
2. Deconstruct source code and documentation into line-aware, symbol-aware chunks with rich metadata (`file_path`, `start_line`, `end_line`, `symbol_name`, `language`).
3. Generate embeddings via an `EmbeddingClient` protocol (OpenAI `text-embedding-3-small` adapter + offline deterministic mock).
4. Store chunks and embeddings in PostgreSQL with `pgvector` (with an interchangeable `VectorStore` protocol and in-memory store for offline test suites).
5. Semantically retrieve relevant code snippets given a user query.
6. Assemble retrieved context into a grounded prompt and generate answers with accurate file-path and line-range citations via FastAPI (`POST /v1/repositories/ingest` and `POST /v1/rag/query`).

---

## Decisions

| Area | Decision | Why |
| --- | --- | --- |
| Parser & Chunking | Python AST parsing with line tracking & symbol extraction; sliding window with overlap for non-Python/text | Ensures functions and classes are chunked as semantic units while preserving exact 1-indexed line numbers for citations. |
| Embedding Abstraction | `EmbeddingClient` protocol with `OpenAIEmbeddingClient` and deterministic `FakeEmbeddingClient` | Allows fast, reproducible unit testing without live API keys or network latency. |
| Vector Storage | `VectorStore` protocol with `PgVectorStore` (PostgreSQL + pgvector) and `InMemoryVectorStore` | Enables production deployment with pgvector while keeping offline unit tests dependency-free and fast. |
| Citation Extraction | Structured prompt guidelines + regex extraction `[filepath:start_line-end_line]` with fallback to top retrieved chunks | Guarantees reliable source attribution for every answer. |
| FastAPI Interface | `POST /v1/repositories/ingest` and `POST /v1/rag/query` | Clean RESTful endpoints integrating ingestion metrics and grounded Q&A. |

---

## Architecture

```text
               +-------------------------------------------+
               |              FastAPI Endpoints            |
               |  POST /v1/repositories/ingest             |
               |  POST /v1/rag/query                       |
               |  POST /v1/generate (Day 1-2)              |
               +---------------------+---------------------+
                                     |
           +-------------------------+-------------------------+
           |                                                   |
           v                                                   v
+-----------------------+                           +---------------------+
|   Ingestion Pipeline  |                           |     RAG Service     |
| - Directory Scanner   |                           | - Query Embedding   |
| - Parser & Chunker    |                           | - Semantic Search   |
| - Embedding Generator |                           | - Context Assembly  |
| - VectorStore Upsert  |                           | - Grounded LLM Call |
+-----------+-----------+                           +----------+----------+
            |                                                  |
            +--------------------+      +----------------------+
                                 |      |
                                 v      v
                    +-----------------------------+
                    |     VectorStore Protocol    |
                    |-----------------------------|
                    | - PgVectorStore (Postgres)  |
                    | - InMemoryVectorStore (Test)|
                    +-----------------------------+
                                 |
                    +-----------------------------+
                    |   EmbeddingClient Protocol  |
                    |-----------------------------|
                    | - OpenAIEmbeddingClient     |
                    | - FakeEmbeddingClient       |
                    +-----------------------------+
```

---

## Implemented Components

1. **`models.py`**:
   - `RepositorySpec`, `FileDocument`, `CodeChunk`, `RetrievalResult`, `Citation`, `IngestionSummary`, `RAGResponse`.
2. **`config.py`**:
   - Runtime configuration for `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, `VECTOR_STORE_TYPE`, `DATABASE_URL`, `CHUNK_SIZE_LINES`, `CHUNK_OVERLAP_LINES`.
3. **`ingestion.py`**:
   - Safe path validation with `ALLOWED_REPOSITORY_ROOTS`.
   - File discovery excluding cache/binary/minified files.
   - AST-aware symbol chunking (`ast.FunctionDef`, `ast.ClassDef`, method tracking) and sliding-window chunking.
4. **`embeddings.py`**:
   - `EmbeddingClient` protocol.
   - `OpenAIEmbeddingClient` with batching.
   - Deterministic `FakeEmbeddingClient` for offline testing.
5. **`vector_store.py`**:
   - `VectorStore` protocol.
   - `InMemoryVectorStore` with cosine similarity and repository filtering.
   - `PgVectorStore` for PostgreSQL with `pgvector` (`vector_cosine_ops`).
6. **`rag.py`**:
   - `RAGService` orchestrating ingestion, retrieval, context prompt formatting, and LLM answer generation with citation extraction.
7. **`app.py`**:
   - `POST /v1/repositories/ingest`
   - `POST /v1/rag/query`

---

## Next Milestone: Day 5-6 - Retrieval Quality And Citations

1. Add BM25 / lexical keyword search baseline.
2. Implement reciprocal rank fusion (RRF) / hybrid retrieval.
3. Add reranking and metadata filters.
4. Run retrieval quality experiments and evaluation benchmarks.
