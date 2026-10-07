# Benchmark Results

Run at: 2026-10-08T02:23:07.185372
API: http://127.0.0.1:8000


## Resume Metrics -- AI Software Engineering Agent

### RAG Pipeline (full query incl. LLM generation)
- **Recall@5**: 100.0% on 27 hybrid-retrieval queries
- **p50 latency**: 7471.7 ms
- **p95 latency**: 13921.1 ms
- **p99 latency**: 14135.0 ms
- **Mean latency**: 7183.7 ms

### Retrieval-Only Latency (embed + vector search, no LLM)
- Cold run: 3538.6 ms
- Warm mean: 5582.3 ms
- Warm min:  5018.7 ms  (best proxy for pure retrieval path)

### Embedding Cache (Redis, same query x10)
- Cold: 4800.1 ms  |  Warm mean: 3966.8 ms
- Speedup: 1.21x  |  Estimated savings: ~17.4%

### Ingestion Throughput
- 85 chunks indexed from 14 files in 65.1s (incl. git clone)
- Conservative floor: 1.31 chunks/sec

### Agent Runs
- Mean agent run duration: N/A ms
- Average citations per run: N/A
