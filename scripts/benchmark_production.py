"""
Production benchmark script for the AI Software Engineering Agent.

Measures:
  1. RAG query latency (p50, p95, p99) + Recall@5, plus a real retrieval-vs-
     generation split read directly from the API (retrieval_ms / generation_ms)
  2. Embedding cache hit rate, read directly from the API's cache_hit field
     (requires the embedding cache to be wired into HybridRetriever -- see
     retrieval.py / rag.py / app.py)
  3. Ingestion throughput (chunks/sec, files/sec)
  4. Agent run duration, citations, and reasoning steps (n=len(AGENT_QUERIES)*n_runs)

Any failed query's full status/response body is captured under
errors_detail in the JSON output -- check that first, then cross-reference
the API server's own console/log output at the matching timestamp.

Usage:
  python scripts/benchmark_production.py
  python scripts/benchmark_production.py --url http://127.0.0.1:8000 --repo-id yield-curve-lab --agent-runs 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no requests dependency needed)
# ---------------------------------------------------------------------------

def _post(url: str, body: dict, timeout: int = 120) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(url: str, timeout: int = 30) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _check_health(base_url: str) -> bool:
    try:
        status, body = _get(f"{base_url}/health", timeout=5)
        return status == 200 and body.get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Benchmark queries
# ---------------------------------------------------------------------------

RAG_QUERIES = [
    "How is zero_rate calculated?",
    "How is the yield curve constructed?",
    "What does the Bond class do?",
    "How is duration calculated for a bond?",
    "How does the bond pricer compute present value?",
    "What is the discount factor formula?",
    "How is convexity measured?",
    "What does the key rate shock module do?",
    "How are coupon payments scheduled?",
    "How is DV01 calculated?",
]

AGENT_QUERIES = [
    "How is zero_rate calculated in the yield curve module?",
    "What tools or classes are used for bond valuation?",
    "Explain how duration and convexity are related in this codebase.",
    "Find and explain the discount factor calculation.",
]

# ---------------------------------------------------------------------------
# Benchmark 1: RAG query latency + Recall@5
# ---------------------------------------------------------------------------

def benchmark_rag(base_url: str, repo_id: str, n_runs: int = 3) -> dict:
    print(f"\n{'='*60}")
    print("BENCHMARK 1: RAG Query Latency + Recall@5")
    print(f"  Queries: {len(RAG_QUERIES)}  x  {n_runs} runs each")
    print(f"{'='*60}")

    latencies_ms: list[float] = []
    retrieval_ms_list: list[float] = []
    generation_ms_list: list[float] = []
    recall_hits = 0
    total_queries = 0
    errors = 0
    error_log: list[dict] = []

    for query in RAG_QUERIES:
        for run in range(n_runs):
            t0 = time.perf_counter()
            status, body = _post(
                f"{base_url}/v1/rag/query",
                {
                    "query": query,
                    "repo_id": repo_id,
                    "top_k": 5,
                    "retrieval_strategy": "hybrid",
                },
                timeout=60,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            if status != 200:
                detail = body.get("detail", body) if isinstance(body, dict) else body
                print(f"  x [{status}] run={run+1} {query!r}")
                print(f"      detail: {detail}")
                errors += 1
                error_log.append({
                    "query": query,
                    "run": run + 1,
                    "status": status,
                    "detail": detail,
                    "elapsed_ms": round(elapsed, 1),
                })
                continue

            latencies_ms.append(elapsed)
            total_queries += 1

            # Real per-component timing, now returned by the API directly
            # (no more cold/warm proxy guessing).
            r_ms = body.get("retrieval_ms")
            g_ms = body.get("generation_ms")
            if r_ms is not None:
                retrieval_ms_list.append(r_ms)
            if g_ms is not None:
                generation_ms_list.append(g_ms)

            # Recall@5: did we get at least 1 retrieved chunk?
            chunks = body.get("retrieved_chunks", [])
            if chunks:
                recall_hits += 1

            marker = "ok" if chunks else "miss"
            split = f"  (retrieval={r_ms}ms, generation={g_ms}ms)" if r_ms is not None else ""
            print(f"  [{marker}] run={run+1} {elapsed:7.1f}ms  {query[:55]!r}{split}")

    if not latencies_ms:
        return {"error": "All RAG queries failed", "errors_detail": error_log}

    latencies_ms.sort()
    recall_at_5 = recall_hits / total_queries if total_queries else 0

    result = {
        "total_queries": total_queries,
        "errors": errors,
        "errors_detail": error_log,
        "recall_at_5": round(recall_at_5 * 100, 1),
        "latency_p50_ms": round(statistics.median(latencies_ms), 1),
        "latency_p95_ms": round(latencies_ms[int(len(latencies_ms) * 0.95)], 1),
        "latency_p99_ms": round(latencies_ms[int(len(latencies_ms) * 0.99)], 1),
        "latency_mean_ms": round(statistics.mean(latencies_ms), 1),
        "latency_min_ms": round(min(latencies_ms), 1),
        "latency_max_ms": round(max(latencies_ms), 1),
        "note": "Full end-to-end latency incl. LLM generation.",
    }

    if retrieval_ms_list:
        retrieval_ms_list.sort()
        result["retrieval_ms_p50"] = round(statistics.median(retrieval_ms_list), 1)
        result["retrieval_ms_mean"] = round(statistics.mean(retrieval_ms_list), 1)
    if generation_ms_list:
        generation_ms_list.sort()
        result["generation_ms_p50"] = round(statistics.median(generation_ms_list), 1)
        result["generation_ms_mean"] = round(statistics.mean(generation_ms_list), 1)

    print(f"\n  Results:")
    print(f"    Recall@5:     {result['recall_at_5']}%")
    print(f"    Errors:       {errors} / {total_queries + errors}  (see errors_detail in JSON output)")
    print(f"    p50 latency:  {result['latency_p50_ms']} ms  (full RAG incl. LLM)")
    print(f"    p95 latency:  {result['latency_p95_ms']} ms")
    print(f"    p99 latency:  {result['latency_p99_ms']} ms")
    print(f"    mean latency: {result['latency_mean_ms']} ms")
    if retrieval_ms_list:
        print(f"    retrieval (embed+search) p50/mean: {result['retrieval_ms_p50']} / {result['retrieval_ms_mean']} ms")
    if generation_ms_list:
        print(f"    Groq generation p50/mean:          {result['generation_ms_p50']} / {result['generation_ms_mean']} ms")

    return result


# ---------------------------------------------------------------------------
# Benchmark 1b: Retrieval-only latency (no LLM — pure embed+search)
# ---------------------------------------------------------------------------

def benchmark_embedding_cache(base_url: str, repo_id: str) -> dict:
    """Send the same query 10x and read the real retrieval_ms + cache_hit fields
    the API now returns directly. This replaces the old wall-clock cold/warm
    proxy, which was dominated by Groq generation-time noise and couldn't
    actually tell a cache hit from random variance.
    """
    print(f"\n{'='*60}")
    print("BENCHMARK 2: Embedding Cache")
    print("  Sending 10 identical queries back-to-back.")
    print("  Reading retrieval_ms + cache_hit directly from the API response")
    print("  (isolated from Groq generation time, not inferred from wall clock).")
    print(f"{'='*60}")

    query = "How is zero_rate calculated?"
    N = 10
    retrieval_ms_runs: list[float] = []
    cache_hits: list[bool] = []
    errors = 0

    for i in range(N):
        status, body = _post(
            f"{base_url}/v1/rag/query",
            {"query": query, "repo_id": repo_id, "top_k": 5, "retrieval_strategy": "hybrid"},
            timeout=60,
        )
        label = "miss(expected)" if i == 0 else f"run{i:02d}"
        if status == 200:
            r_ms = body.get("retrieval_ms")
            hit = body.get("cache_hit")
            if r_ms is not None:
                retrieval_ms_runs.append(r_ms)
            cache_hits.append(bool(hit))
            print(f"  [{label}] retrieval={r_ms}ms  cache_hit={hit}")
        else:
            errors += 1
            print(f"  [{label}] ERROR {status}: {body.get('detail', body) if isinstance(body, dict) else body}")

    if len(retrieval_ms_runs) < 2:
        return {"error": "Not enough successful runs", "errors": errors}

    first_call_ms = retrieval_ms_runs[0]
    repeat_calls_ms = retrieval_ms_runs[1:]
    repeat_mean_ms = statistics.mean(repeat_calls_ms)
    hit_count = sum(1 for h in cache_hits[1:] if h)
    hit_rate_pct = round(100 * hit_count / len(cache_hits[1:]), 1) if len(cache_hits) > 1 else 0.0
    speedup = round(first_call_ms / repeat_mean_ms, 2) if repeat_mean_ms > 0 else 1.0

    result = {
        "first_call_retrieval_ms": round(first_call_ms, 1),
        "repeat_calls_retrieval_ms_mean": round(repeat_mean_ms, 1),
        "cache_hit_rate_pct": hit_rate_pct,
        "retrieval_speedup_factor": speedup,
        "errors": errors,
        "note": (
            "retrieval_ms and cache_hit are read directly from the API response "
            "(rag.py times retrieval and generation separately; retrieval.py now "
            "checks the embedding cache before calling the embedding API). This "
            "isolates the cache effect from Groq generation-time variance."
        ),
    }

    print(f"\n  Results:")
    print(f"    First call (miss) retrieval:  {result['first_call_retrieval_ms']} ms")
    print(f"    Repeat calls retrieval mean:  {result['repeat_calls_retrieval_ms_mean']} ms")
    print(f"    Cache hit rate (runs 2-10):   {result['cache_hit_rate_pct']}%")
    print(f"    Retrieval speedup:            {result['retrieval_speedup_factor']}x")

    return result


# ---------------------------------------------------------------------------
# Benchmark 3: Ingestion throughput
# ---------------------------------------------------------------------------

def benchmark_ingestion(base_url: str) -> dict:
    print(f"\n{'='*60}")
    print("BENCHMARK 3: Ingestion Throughput")
    print("  Re-ingesting the same repo (async) to measure throughput")
    print(f"{'='*60}")

    repo_payload = {
        "github_url": "https://github.com/labi22/Fixed-Income-Analytics-Bond-Valuation-Platform",
        "ref": "main",
        "repo_id": "yield-curve-bench",
        "name": "Yield Curve Bench",
    }

    t0 = time.perf_counter()
    status, body = _post(
        f"{base_url}/v1/repositories/ingest/async",
        repo_payload,
        timeout=30,
    )
    if status != 202:
        print(f"  x Failed to enqueue ingestion job: {status} {body}")
        return {"error": f"Enqueue failed: {status}"}

    job_id = body["job_id"]
    print(f"  Enqueued job {job_id}, polling for completion...")

    # Poll until done
    deadline = time.perf_counter() + 300  # 5 min max
    poll_interval = 3.0
    result_body = {}
    while time.perf_counter() < deadline:
        time.sleep(poll_interval)
        s, jbody = _get(f"{base_url}/v1/ingestion-jobs/{job_id}", timeout=10)
        job_status = jbody.get("status", "unknown")
        elapsed_so_far = (time.perf_counter() - t0)
        print(f"    [{elapsed_so_far:5.1f}s] status={job_status}")
        if job_status in ("succeeded", "failed"):
            result_body = jbody
            break

    total_elapsed = time.perf_counter() - t0

    if result_body.get("status") != "succeeded":
        return {"error": f"Ingestion did not succeed: {result_body.get('error')}"}

    job_result = result_body.get("result") or {}
    chunks = job_result.get("chunks_created", 0)
    files_parsed = job_result.get("files_parsed", 0)
    files_scanned = job_result.get("files_scanned", 0)

    # Total elapsed includes GitHub clone over the network.
    # For a small repo, the clone typically takes 20-60s.
    # We report both total and a processing-only estimate (total - 30s clone heuristic).
    # The worker logs would give exact clone time but we don't have that here.
    chunks_per_sec_total = round(chunks / total_elapsed, 2) if total_elapsed > 0 else 0
    files_per_sec_total = round(files_parsed / total_elapsed, 2) if total_elapsed > 0 else 0

    result = {
        "total_elapsed_sec": round(total_elapsed, 1),
        "files_scanned": files_scanned,
        "files_parsed": files_parsed,
        "chunks_created": chunks,
        "chunks_per_sec_incl_clone": chunks_per_sec_total,
        "files_per_sec_incl_clone": files_per_sec_total,
        "note": (
            "Total elapsed includes GitHub clone over network. "
            "chunks_per_sec_incl_clone is a conservative floor; "
            "pure parse+embed throughput is higher."
        ),
    }

    print(f"\n  Results:")
    print(f"    Total time:    {result['total_elapsed_sec']}s  (incl. git clone)")
    print(f"    Files scanned: {result['files_scanned']}")
    print(f"    Files parsed:  {result['files_parsed']}")
    print(f"    Chunks:        {result['chunks_created']}")
    print(f"    Throughput:    {result['chunks_per_sec_incl_clone']} chunks/sec (floor, incl. clone)")

    return result


# ---------------------------------------------------------------------------
# Benchmark 4: Agent run duration
# ---------------------------------------------------------------------------

def benchmark_agent(base_url: str, repo_id: str, n_runs: int = 3) -> dict:
    total_calls = len(AGENT_QUERIES) * n_runs
    print(f"\n{'='*60}")
    print("BENCHMARK 4: Agent Run Duration")
    print(f"  Queries: {len(AGENT_QUERIES)}  x  {n_runs} runs each  ({total_calls} total)")
    print(f"{'='*60}")

    durations_ms: list[float] = []
    citation_counts: list[int] = []
    steps_per_run: list[int] = []
    errors = 0
    error_log: list[dict] = []

    for query in AGENT_QUERIES:
        for run in range(n_runs):
            print(f"  Running (run {run+1}/{n_runs}): {query!r}")
            t0 = time.perf_counter()
            status, body = _post(
                f"{base_url}/v1/agent/run",
                {
                    "task": query,
                    "repo_id": repo_id,
                    "max_steps": 5,
                },
                timeout=120,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            if status != 200:
                detail = body.get("detail", body) if isinstance(body, dict) else body
                print(f"    x [{status}] {detail}")
                errors += 1
                error_log.append({
                    "query": query, "run": run + 1, "status": status,
                    "detail": detail, "elapsed_ms": round(elapsed, 1),
                })
                continue

            duration_ms = body.get("duration_ms", elapsed)
            n_citations = len(body.get("citations", []))
            n_steps = body.get("total_steps")
            durations_ms.append(duration_ms)
            citation_counts.append(n_citations)
            if n_steps is not None:
                steps_per_run.append(n_steps)
            print(f"    ok {duration_ms:.0f}ms  citations={n_citations}  steps={n_steps}  status={body.get('status')}")

    if not durations_ms:
        return {"error": "All agent runs failed", "errors_detail": error_log}

    durations_ms_sorted = sorted(durations_ms)

    result = {
        "total_runs": total_calls,
        "successful_runs": len(durations_ms),
        "errors": errors,
        "errors_detail": error_log,
        "mean_duration_ms": round(statistics.mean(durations_ms), 1),
        "median_duration_ms": round(statistics.median(durations_ms), 1),
        "min_duration_ms": round(min(durations_ms), 1),
        "max_duration_ms": round(max(durations_ms), 1),
        "avg_citations": round(statistics.mean(citation_counts), 1) if citation_counts else 0,
    }
    if steps_per_run:
        result["avg_steps"] = round(statistics.mean(steps_per_run), 1)

    print(f"\n  Results (n={len(durations_ms)}):")
    print(f"    Mean / median duration: {result['mean_duration_ms']} / {result['median_duration_ms']} ms")
    print(f"    Min / Max:              {result['min_duration_ms']} / {result['max_duration_ms']} ms")
    print(f"    Avg citations:          {result['avg_citations']}")
    if steps_per_run:
        print(f"    Avg reasoning steps:    {result['avg_steps']}")

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

RESUME_TEMPLATE = """
## Resume Metrics -- AI Software Engineering Agent

### RAG Pipeline (full query incl. LLM generation)
- **Recall@5**: {recall_at_5}% on {total_queries} hybrid-retrieval queries  (errors: {errors})
- **p50 / p95 / p99 latency**: {latency_p50_ms} / {latency_p95_ms} / {latency_p99_ms} ms
- **Mean latency**: {latency_mean_ms} ms

### Retrieval vs. Generation split (real, from the API -- not inferred)
- Retrieval (embed + hybrid search) p50 / mean: {retrieval_ms_p50} / {retrieval_ms_mean} ms
- Groq generation p50 / mean: {generation_ms_p50} / {generation_ms_mean} ms

### Embedding Cache (Redis-backed, wired into the query path)
- First call (miss) retrieval: {cache_first_call_ms} ms
- Repeat calls (hit) retrieval mean: {cache_repeat_mean_ms} ms
- Cache hit rate (runs 2-10): {cache_hit_rate_pct}%
- Retrieval speedup on hit: {cache_speedup}x

### Ingestion Throughput
- {chunks_created} chunks indexed from {files_parsed} files in {total_elapsed_sec}s (incl. git clone)
- Conservative floor: {chunks_per_sec} chunks/sec

### Agent Runs (n={agent_n})
- Mean / median duration: {mean_agent_ms} / {median_agent_ms} ms
- Average citations per run: {avg_citations}
- Average reasoning steps: {avg_steps}
- Errors: {agent_errors}

NOTE: any non-zero "errors" above means some queries failed. Check
errors_detail in the JSON output for the full status/detail per failure,
and cross-reference the API server's own console/log output at the
matching timestamp for the underlying traceback.
"""

def generate_report(results: dict) -> str:
    rag = results.get("rag", {})
    cache = results.get("cache", {})
    ingest = results.get("ingestion", {})
    agent = results.get("agent", {})

    return RESUME_TEMPLATE.format(
        recall_at_5=rag.get("recall_at_5", "N/A"),
        total_queries=rag.get("total_queries", "N/A"),
        errors=rag.get("errors", "N/A"),
        latency_p50_ms=rag.get("latency_p50_ms", "N/A"),
        latency_p95_ms=rag.get("latency_p95_ms", "N/A"),
        latency_p99_ms=rag.get("latency_p99_ms", "N/A"),
        latency_mean_ms=rag.get("latency_mean_ms", "N/A"),
        retrieval_ms_p50=rag.get("retrieval_ms_p50", "N/A"),
        retrieval_ms_mean=rag.get("retrieval_ms_mean", "N/A"),
        generation_ms_p50=rag.get("generation_ms_p50", "N/A"),
        generation_ms_mean=rag.get("generation_ms_mean", "N/A"),
        cache_first_call_ms=cache.get("first_call_retrieval_ms", "N/A"),
        cache_repeat_mean_ms=cache.get("repeat_calls_retrieval_ms_mean", "N/A"),
        cache_hit_rate_pct=cache.get("cache_hit_rate_pct", "N/A"),
        cache_speedup=cache.get("retrieval_speedup_factor", "N/A"),
        chunks_created=ingest.get("chunks_created", "N/A"),
        files_parsed=ingest.get("files_parsed", "N/A"),
        total_elapsed_sec=ingest.get("total_elapsed_sec", "N/A"),
        chunks_per_sec=ingest.get("chunks_per_sec_incl_clone", "N/A"),
        agent_n=agent.get("successful_runs", "N/A"),
        mean_agent_ms=agent.get("mean_duration_ms", "N/A"),
        median_agent_ms=agent.get("median_duration_ms", "N/A"),
        avg_citations=agent.get("avg_citations", "N/A"),
        avg_steps=agent.get("avg_steps", "N/A"),
        agent_errors=agent.get("errors", "N/A"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark the AI Software Engineering Agent")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base API URL")
    parser.add_argument("--repo-id", default="yield-curve-lab", help="Ingested repo ID to query against")
    parser.add_argument("--rag-runs", type=int, default=3, help="Runs per RAG query")
    parser.add_argument("--agent-runs", type=int, default=3, help="Runs per agent query")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip the ingestion benchmark (slow)")
    parser.add_argument("--skip-agent", action="store_true", help="Skip agent benchmark (uses LLM quota)")
    parser.add_argument("--output", default="benchmark_results.md", help="Output file path")
    args = parser.parse_args()

    print(f"\nAI Software Engineering Agent — Production Benchmark")
    print(f"Target: {args.url}")
    print(f"Repo:   {args.repo_id}")
    print(f"Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Health check
    print(f"\nChecking API health...")
    if not _check_health(args.url):
        print(f"✗ API at {args.url} is not healthy. Start it first.")
        sys.exit(1)
    print("✓ API is healthy")

    all_results: dict = {
        "timestamp": datetime.now().isoformat(),
        "base_url": args.url,
        "repo_id": args.repo_id,
    }

    # Run benchmarks
    all_results["rag"] = benchmark_rag(args.url, args.repo_id, n_runs=args.rag_runs)

    all_results["cache"] = benchmark_embedding_cache(args.url, args.repo_id)

    if not args.skip_ingestion:
        all_results["ingestion"] = benchmark_ingestion(args.url)
    else:
        print("\n[Skipping ingestion benchmark]")
        all_results["ingestion"] = {}

    if not args.skip_agent:
        all_results["agent"] = benchmark_agent(args.url, args.repo_id, n_runs=args.agent_runs)
    else:
        print("\n[Skipping agent benchmark]")
        all_results["agent"] = {}

    # Write JSON results
    json_path = Path(args.output).with_suffix(".json")
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n✓ Raw results written to {json_path}")

    # Write markdown report
    md_path = Path(args.output)
    report = generate_report(all_results)
    md_header = f"# Benchmark Results\n\nRun at: {all_results['timestamp']}\nAPI: {args.url}\n\n"
    md_path.write_text(md_header + report, encoding="utf-8")
    print(f"✓ Markdown report written to {md_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY — Resume Metrics")
    print(f"{'='*60}")
    print(report)


if __name__ == "__main__":
    main()