# Benchmark Results

Run at: 2026-10-09T11:57:08.678083
API: http://127.0.0.1:8000


## Resume Metrics -- AI Software Engineering Agent

### RAG Pipeline (full query incl. LLM generation)
- **Recall@5**: 100.0% on 27 hybrid-retrieval queries  (errors: 3)
- **p50 / p95 / p99 latency**: 6254.0 / 13549.1 / 14745.4 ms
- **Mean latency**: 6885.2 ms

### Retrieval vs. Generation split (real, from the API -- not inferred)
- Retrieval (embed + hybrid search) p50 / mean: 51.5 / 68.6 ms
- Groq generation p50 / mean: 6102.2 / 6787.6 ms

### Embedding Cache (Redis-backed, wired into the query path)
- First call (miss) retrieval: 37.3 ms
- Repeat calls (hit) retrieval mean: 44.7 ms
- Cache hit rate (runs 2-10): 100.0%
- Retrieval speedup on hit: 0.83x

### Ingestion Throughput
- 85 chunks indexed from 14 files in 95.7s (incl. git clone)
- Conservative floor: 0.89 chunks/sec

### Agent Runs (n=20)
- Mean / median duration: 26065.6 / 24020.6 ms
- Average citations per run: 9.5
- Average reasoning steps: 2.5
- Errors: 0

NOTE: any non-zero "errors" above means some queries failed. Check
errors_detail in the JSON output for the full status/detail per failure,
and cross-reference the API server's own console/log output at the
matching timestamp for the underlying traceback.
