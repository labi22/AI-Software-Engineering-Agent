"""Hybrid retrieval pipeline, Reciprocal Rank Fusion (RRF), and symbol reranking."""

from __future__ import annotations

from typing import Protocol, Sequence

from .embeddings import EmbeddingClient
from .lexical import BM25Index, CodeTokenizer
from .models import (
    CodeChunk,
    MetadataFilter,
    RetrievalDebugInfo,
    RetrievalResult,
    RetrievalStrategy,
)
from .vector_store import VectorStore


def reciprocal_rank_fusion(
    dense_results: Sequence[RetrievalResult],
    bm25_results: Sequence[RetrievalResult],
    k: int = 60,
    limit: int = 5,
) -> list[RetrievalResult]:
    """Fuse dense vector and BM25 lexical results using Reciprocal Rank Fusion (RRF).

    Formula: RRF_Score(d) = sum_{system} ( 1 / (k + rank_system(d)) )
    """
    candidates: dict[str, dict[str, any]] = {}

    # Record dense ranks (1-indexed)
    for rank, res in enumerate(dense_results, start=1):
        cid = res.chunk.chunk_id
        if cid not in candidates:
            candidates[cid] = {
                "chunk": res.chunk,
                "dense_rank": rank,
                "dense_score": res.score,
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            }
        else:
            candidates[cid]["dense_rank"] = rank
            candidates[cid]["dense_score"] = res.score
        candidates[cid]["rrf_score"] += 1.0 / (k + rank)

    # Record BM25 ranks (1-indexed)
    for rank, res in enumerate(bm25_results, start=1):
        cid = res.chunk.chunk_id
        if cid not in candidates:
            candidates[cid] = {
                "chunk": res.chunk,
                "dense_rank": None,
                "dense_score": None,
                "bm25_rank": rank,
                "bm25_score": res.score,
                "rrf_score": 0.0,
            }
        else:
            candidates[cid]["bm25_rank"] = rank
            candidates[cid]["bm25_score"] = res.score
        candidates[cid]["rrf_score"] += 1.0 / (k + rank)

    # Sort descending by fused RRF score
    sorted_candidates = sorted(
        candidates.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    results: list[RetrievalResult] = []
    for item in sorted_candidates[:limit]:
        debug_info = RetrievalDebugInfo(
            dense_rank=item["dense_rank"],
            dense_score=item["dense_score"],
            bm25_rank=item["bm25_rank"],
            bm25_score=item["bm25_score"],
            rerank_boost=0.0,
            fusion_score=round(item["rrf_score"], 6),
        )
        results.append(
            RetrievalResult(
                chunk=item["chunk"],
                score=round(item["rrf_score"], 6),
                debug_info=debug_info,
            )
        )

    return results


class Reranker(Protocol):
    """Protocol for scoring and re-ordering retrieval candidates."""

    def rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        """Rerank candidates based on additional lexical, structural, or semantic signals."""


class SymbolBoostReranker:
    """Reranker that applies relevance boosts to exact symbol and file path matches."""

    def __init__(
        self,
        symbol_exact_boost: float = 0.25,
        symbol_partial_boost: float = 0.10,
        filepath_boost: float = 0.08,
    ) -> None:
        self.symbol_exact_boost = symbol_exact_boost
        self.symbol_partial_boost = symbol_partial_boost
        self.filepath_boost = filepath_boost

    def rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        if not candidates or not query.strip():
            return candidates

        query_lower = query.lower()
        query_tokens = set(CodeTokenizer.tokenize(query))
        reranked: list[RetrievalResult] = []

        for res in candidates:
            chunk = res.chunk
            boost = 0.0

            # 1. Exact symbol match boost
            if chunk.symbol_name:
                symbol_lower = chunk.symbol_name.lower()
                # Check for exact token match or exact substring match
                if symbol_lower in query_lower or any(
                    t == symbol_lower or symbol_lower.endswith(f".{t}") for t in query_tokens
                ):
                    boost += self.symbol_exact_boost
                elif any(t in symbol_lower for t in query_tokens):
                    boost += self.symbol_partial_boost

            # 2. File path match boost
            path_tokens = set(CodeTokenizer.tokenize(chunk.file_path))
            if path_tokens.intersection(query_tokens):
                boost += self.filepath_boost

            new_score = res.score + boost
            old_debug = res.debug_info
            updated_debug = RetrievalDebugInfo(
                dense_rank=old_debug.dense_rank if old_debug else None,
                dense_score=old_debug.dense_score if old_debug else None,
                bm25_rank=old_debug.bm25_rank if old_debug else None,
                bm25_score=old_debug.bm25_score if old_debug else None,
                rerank_boost=round(boost, 4),
                fusion_score=old_debug.fusion_score if old_debug else res.score,
            )

            reranked.append(
                RetrievalResult(
                    chunk=chunk,
                    score=round(new_score, 6),
                    debug_info=updated_debug,
                )
            )

        # Re-sort descending by boosted score
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked


class QueryExpander:
    """Expands domain terminology, common acronyms, and code identifiers."""

    SYNONYMS: dict[str, list[str]] = {
        "ytm": ["yield_to_maturity", "yield", "rate"],
        "npv": ["net_present_value", "discounted_cash_flows"],
        "bootstrap": ["bootstrapping", "zero_curve", "spot_rate"],
        "curve": ["yield_curve", "zero_curve", "tenors"],
        "duration": ["macaulay_duration", "modified_duration"],
        "convexity": ["bond_convexity"],
        "pricing": ["valuation", "price", "discount_factor"],
    }

    @classmethod
    def expand(cls, query: str) -> str:
        tokens = CodeTokenizer.tokenize(query)
        expanded_terms: list[str] = [query]

        for token in tokens:
            if token in cls.SYNONYMS:
                expanded_terms.extend(cls.SYNONYMS[token])

        return " ".join(expanded_terms)


class HybridRetriever:
    """Orchestrates dense vector search, BM25 lexical search, metadata filtering, and reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedding_client: EmbeddingClient,
        reranker: Reranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_client = embedding_client
        self.reranker = reranker or SymbolBoostReranker()

    async def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        filter: MetadataFilter | None = None,
        limit: int = 5,
        k_rrf: int = 60,
    ) -> list[RetrievalResult]:
        """Perform retrieval using the requested strategy (dense, bm25, or hybrid)."""
        if not query.strip():
            return []

        repo_id = filter.repo_id if filter else None

        # 1. Lexical retrieval
        if strategy == RetrievalStrategy.BM25:
            results = self.bm25_index.search(query=query, limit=limit, filter=filter)
            return self.reranker.rerank(query, results)

        # 2. Dense retrieval
        query_embedding = await self.embedding_client.embed_query(query)
        dense_results = await self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit * 2 if strategy == RetrievalStrategy.HYBRID else limit,
            repo_id=repo_id,
        )

        # Apply metadata filtering on dense results if needed
        if filter:
            dense_results = [
                r for r in dense_results if self.bm25_index._matches_filter(r.chunk, filter)
            ]

        if strategy == RetrievalStrategy.DENSE:
            return self.reranker.rerank(query, dense_results[:limit])

        # 3. Hybrid retrieval (Dense + BM25 + RRF + Rerank)
        expanded_query = QueryExpander.expand(query)
        bm25_results = self.bm25_index.search(
            query=expanded_query,
            limit=limit * 2,
            filter=filter,
        )

        fused_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            k=k_rrf,
            limit=limit,
        )

        return self.reranker.rerank(query, fused_results)
