"""Grounded Retrieval-Augmented Generation (RAG) service with hybrid search and citation validation."""

from __future__ import annotations

import re
from typing import Sequence

from .config import Settings
from .embeddings import EmbeddingClient
from .ingestion import chunk_document, discover_files, parse_file
from .lexical import BM25Index
from .llm import LLMClient, LLMConfigurationError, LLMRequest
from .models import (
    Citation,
    CodeChunk,
    FileDocument,
    IngestionSummary,
    MetadataFilter,
    RAGResponse,
    RepositorySpec,
    RetrievalResult,
    RetrievalStrategy,
)
from .retrieval import HybridRetriever, SymbolBoostReranker
from .vector_store import VectorStore


class RAGError(RuntimeError):
    """Base error for RAG operations."""


def format_context_prompt(retrieval_results: Sequence[RetrievalResult]) -> str:
    """Format retrieved code chunks into a grounded context prompt with citation instructions."""
    if not retrieval_results:
        return (
            "You are an AI software engineering assistant. No code context was found for this query. "
            "Please indicate that no relevant code was retrieved from the ingested repository."
        )

    context_sections: list[str] = []
    for idx, res in enumerate(retrieval_results, start=1):
        chunk = res.chunk
        symbol_text = f", Symbol: {chunk.symbol_name}" if chunk.symbol_name else ""
        header = f"[{idx}] File: {chunk.file_path} (Lines {chunk.start_line}-{chunk.end_line}{symbol_text})"
        section = f"{header}\n```\n{chunk.content}\n```"
        context_sections.append(section)

    context_str = "\n\n".join(context_sections)

    return (
        "You are an expert AI software engineering assistant. Your task is to answer questions "
        "about the ingested codebase with strict grounding in the provided context snippets.\n\n"
        "GUIDELINES:\n"
        "1. Base your answer ONLY on the retrieved code and documentation snippets below.\n"
        "2. For every technical claim, function, or class you discuss, cite the exact source using "
        "the format: `[filepath:start_line-end_line]` (e.g. `[src/curve.py:12-45]`).\n"
        "3. If the retrieved snippets do not contain enough information to answer completely, "
        "explicitly state what is missing rather than speculating.\n\n"
        f"--- RETRIEVED CONTEXT ---\n{context_str}\n--- END CONTEXT ---"
    )


def validate_citations(
    citations: Sequence[Citation],
    retrieved_chunks: Sequence[RetrievalResult],
) -> list[Citation]:
    """Verify that cited file paths and line ranges exist within the retrieved code context."""
    validated: list[Citation] = []
    for citation in citations:
        norm_cited_path = citation.file_path.replace("\\", "/").lower()
        is_verified = False

        for res in retrieved_chunks:
            chunk = res.chunk
            norm_chunk_path = chunk.file_path.replace("\\", "/").lower()

            # Check if file path matches or ends with the cited path
            if norm_chunk_path == norm_cited_path or norm_chunk_path.endswith(f"/{norm_cited_path}"):
                # Check for line range overlap: max(start1, start2) <= min(end1, end2)
                overlap_start = max(citation.start_line, chunk.start_line)
                overlap_end = min(citation.end_line, chunk.end_line)
                if overlap_start <= overlap_end or abs(citation.start_line - chunk.start_line) <= 5:
                    is_verified = True
                    break

        validated.append(
            Citation(
                file_path=citation.file_path,
                start_line=citation.start_line,
                end_line=citation.end_line,
                symbol_name=citation.symbol_name,
                snippet=citation.snippet,
                is_verified=is_verified,
            )
        )

    return validated


def extract_citations(
    text: str,
    retrieved_chunks: Sequence[RetrievalResult],
) -> list[Citation]:
    """Extract citations from response text or fall back to top retrieved chunks."""
    citation_pattern = re.compile(
        r"\[([a-zA-Z0-9_\-/\.\\]+):(\d+)-(\d+)(?:\s*\(([^)]+)\))?\]"
    )
    found_matches = citation_pattern.findall(text)
    raw_citations: list[Citation] = []
    seen: set[tuple[str, int, int]] = set()

    for path, start_str, end_str, symbol in found_matches:
        start_line = int(start_str)
        end_line = int(end_str)
        key = (path.replace("\\", "/"), start_line, end_line)
        if key not in seen:
            seen.add(key)
            raw_citations.append(
                Citation(
                    file_path=key[0],
                    start_line=start_line,
                    end_line=end_line,
                    symbol_name=symbol if symbol else None,
                )
            )

    # Fallback to top retrieved chunks if LLM did not emit structured citation tags
    if not raw_citations and retrieved_chunks:
        for res in retrieved_chunks[:3]:
            c = res.chunk
            key = (c.file_path.replace("\\", "/"), c.start_line, c.end_line)
            if key not in seen:
                seen.add(key)
                raw_citations.append(
                    Citation(
                        file_path=c.file_path,
                        start_line=c.start_line,
                        end_line=c.end_line,
                        symbol_name=c.symbol_name,
                        snippet=c.content[:200],
                        is_verified=True,
                    )
                )

    return validate_citations(raw_citations, retrieved_chunks)


class RAGService:
    """Coordinates repository ingestion, hybrid retrieval, and grounded Q&A generation."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient | None = None,
        bm25_index: BM25Index | None = None,
        retriever: HybridRetriever | None = None,
        settings: Settings,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.bm25_index = bm25_index or BM25Index()
        self.retriever = retriever or HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            embedding_client=self.embedding_client,
            reranker=SymbolBoostReranker(),
        )
        self.settings = settings

    async def ingest_repository(self, spec: RepositorySpec) -> IngestionSummary:
        """Scan, parse, chunk, embed, and index a repository across vector and BM25 stores."""
        await self.vector_store.initialize()

        discovered_paths = discover_files(
            spec.root_path,
            allowed_roots=self.settings.allowed_repository_roots,
        )

        all_chunks: list[CodeChunk] = []
        parsed_count = 0
        total_tokens = 0

        for path in discovered_paths:
            doc = parse_file(path, spec.root_path)
            if doc is None:
                continue

            parsed_count += 1
            chunks = chunk_document(
                doc=doc,
                repo_id=spec.repo_id,
                max_lines=self.settings.chunk_size_lines,
                overlap=self.settings.chunk_overlap_lines,
            )
            for chunk in chunks:
                total_tokens += chunk.token_count
                all_chunks.append(chunk)

        if all_chunks:
            # 1. Index in BM25 lexical engine
            self.bm25_index.index_chunks(all_chunks)

            # 2. Embed and persist in VectorStore
            chunk_texts = [c.content for c in all_chunks]
            embeddings = await self.embedding_client.embed(chunk_texts)
            await self.vector_store.store_chunks(all_chunks, embeddings)

        return IngestionSummary(
            repo_id=spec.repo_id,
            files_scanned=len(discovered_paths),
            files_parsed=parsed_count,
            chunks_created=len(all_chunks),
            total_tokens=total_tokens,
        )

    async def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        filter: MetadataFilter | None = None,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve relevant code snippets using the specified retrieval strategy and filters."""
        if not query or not query.strip():
            return []

        return await self.retriever.retrieve(
            query=query,
            strategy=strategy,
            filter=filter,
            limit=top_k,
        )

    async def answer_query(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        filter: MetadataFilter | None = None,
        top_k: int = 5,
    ) -> RAGResponse:
        """Retrieve context via hybrid search and generate a grounded, cited answer."""
        if not query or not query.strip():
            raise RAGError("Query text cannot be empty.")
        if self.llm_client is None:
            raise LLMConfigurationError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai.")

        retrieval_results = await self.retrieve(
            query=query,
            strategy=strategy,
            filter=filter,
            top_k=top_k,
        )
        system_instruction = format_context_prompt(retrieval_results)

        llm_response = await self.llm_client.generate(
            LLMRequest(
                prompt=query,
                system_instruction=system_instruction,
            )
        )

        citations = extract_citations(llm_response.text, retrieval_results)

        return RAGResponse(
            query=query,
            answer=llm_response.text,
            citations=citations,
            retrieved_chunks=retrieval_results,
            model=llm_response.model,
            provider=llm_response.provider,
            strategy_used=strategy.value if isinstance(strategy, RetrievalStrategy) else str(strategy),
        )
