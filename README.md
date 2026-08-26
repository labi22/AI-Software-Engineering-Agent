# AI Software Engineering Agent

A repository-agnostic AI software engineering assistant that scans, ingests, indexes, and retrieves code to answer technical engineering questions with precise source and line-number citations.

## Features

- **FastAPI API Layer**: REST endpoints for text generation, repository ingestion, and grounded RAG querying.
- **Repository Ingestion & Safe Discovery**: Scans target codebases with security guardrails (`ALLOWED_REPOSITORY_ROOTS`), ignoring binary, build, cache, and lock files.
- **Syntax & Line-Aware Code Chunking**: AST-based chunking for Python (classes, functions, methods) and sliding-window chunking for documentation and other files, tracking exact start and end line ranges.
- **Provider-Neutral Embeddings**: Protocol supporting OpenAI `text-embedding-3-small` and deterministic offline mocks.
- **Vector Storage**: Protocol supporting PostgreSQL with `pgvector` and fast `InMemoryVectorStore`.
- **Grounded Semantic RAG**: Context assembly prompt instructing LLM to answer strictly from retrieved snippets and generate precise `[filepath:start_line-end_line]` citations.

## Prerequisites

- Python 3.11+
- An OpenAI API key (for live inference; offline mock mode available for tests)
- PostgreSQL + pgvector (optional for durable vector persistence)

## Setup & Local Run

Install the project with development dependencies:

```powershell
python -m pip install -e .[dev]
```

Configure your environment:

```powershell
Copy-Item .env.example .env
$env:OPENAI_API_KEY = "your_openai_api_key"
$env:ALLOWED_REPOSITORY_ROOTS = "C:\Users\saksh\Yield Curve Construction and Bond Valuation & Risk Lab"
```

Start the service:

```powershell
uvicorn ai_software_engineering_agent.app:app --reload
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

## API Usage Examples

### 1. Ingest a Repository

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/repositories/ingest" `
  -ContentType "application/json" `
  -Body '{"repository_path": "C:\\Users\\saksh\\Yield Curve Construction and Bond Valuation & Risk Lab", "repo_id": "bond-lab"}'
```

Response:
```json
{
  "repo_id": "bond-lab",
  "files_scanned": 15,
  "files_parsed": 15,
  "chunks_created": 68,
  "total_tokens": 12450,
  "status": "success"
}
```

### 2. Query the Codebase with Grounded Citations

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/rag/query" `
  -ContentType "application/json" `
  -Body '{"query": "How is zero-coupon yield curve bootstrapped?", "repo_id": "bond-lab", "top_k": 5}'
```

Response:
```json
{
  "query": "How is zero-coupon yield curve bootstrapped?",
  "answer": "The yield curve is bootstrapped iteratively across par instruments in [models/bootstrap.py:15-45]...",
  "citations": [
    {
      "file_path": "models/bootstrap.py",
      "start_line": 15,
      "end_line": 45,
      "symbol_name": "bootstrap_curve"
    }
  ],
  "retrieved_chunks": [...],
  "model": "gpt-5.2",
  "provider": "openai"
}
```

### 3. Direct Generation

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/generate" `
  -ContentType "application/json" `
  -Body '{"prompt": "Explain the purpose of a yield curve in two sentences."}'
```

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Credential for OpenAI LLM and embeddings | Required for live OpenAI calls |
| `LLM_PROVIDER` | Selected LLM provider | `openai` |
| `LLM_MODEL` | LLM model identifier | `gpt-5.2` |
| `LLM_REQUEST_TIMEOUT_SECONDS` | OpenAI request timeout | `30` |
| `ALLOWED_REPOSITORY_ROOTS` | Comma-separated approved paths for ingestion/tools | Empty |
| `EMBEDDING_PROVIDER` | Embedding provider (`openai` or `fake`) | `openai` |
| `EMBEDDING_MODEL` | Embedding model identifier | `text-embedding-3-small` |
| `EMBEDDING_DIMENSION` | Embedding vector dimension | `1536` |
| `VECTOR_STORE_TYPE` | Vector store type (`memory` or `pgvector`) | `memory` |
| `DATABASE_URL` | PostgreSQL connection URL for pgvector | `None` |
| `CHUNK_SIZE_LINES` | Maximum line count per code chunk | `60` |
| `CHUNK_OVERLAP_LINES` | Line overlap for sliding-window chunking | `15` |

## Tests

Run the full automated test suite offline:

```powershell
pytest
```
