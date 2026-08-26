"""Grounded Retrieval-Augmented Generation (RAG) service and context assembler."""

from __future__ import annotations

import re
from typing import Sequence

from .config import Settings
from .embeddings import EmbeddingClient
from .ingestion import chunk_document, discover_files, parse_file
from .llm import LLMClient, LLMRequest
from .models import (
    Citation,
    CodeChunk,
    FileDocument,
    IngestionSummary,
    RAGResponse,
    RepositorySpec,
    RetrievalResult,
)
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


def extract_citations(
    text: str,
    retrieved_chunks: Sequence[RetrievalResult],
) -> list[Citation]:
    """Extract citations from response text or fall back to retrieved top results."""
    citation_pattern = re.compile(
        r"\[([a-zA-Z0-9_\-/\.\\]+):(\d+)-(\d+)(?:\s*\(([^)]+)\))?\]"
    )
    found_matches = citation_pattern.findall(text)
    citations: list[Citation] = []
    seen: set[tuple[str, int, int]] = set()

    for path, start_str, end_str, symbol in found_matches:
        start_line = int(start_str)
        end_line = int(end_str)
        key = (path.replace("\\", "/"), start_line, end_line)
        if key not in seen:
            seen.add(key)
            citations.append(
                Citation(
                    file_path=key[0],
                    start_line=start_line,
                    end_line=end_line,
                    symbol_name=symbol if symbol else None,
                )
            )

    # If the LLM didn't explicitly format tags, populate citations from the top retrieved chunks
    if not citations and retrieved_chunks:
        for res in retrieved_chunks[:3]:
            c = res.chunk
            key = (c.file_path.replace("\\", "/"), c.start_line, c.end_line)
            if key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        file_path=c.file_path,
                        start_line=c.start_line,
                        end_line=c.end_line,
                        symbol_name=c.symbol_name,
                        snippet=c.content[:200],
                    )
                )

    return citations


class RAGService:
    """Coordinates repository ingestion, semantic retrieval, and grounded Q&A generation."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient,
        settings: Settings,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.settings = settings

    async def ingest_repository(self, spec: RepositorySpec) -> IngestionSummary:
        """Scan, parse, chunk, embed, and store a repository."""
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
        repo_id: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Perform semantic search across stored repository chunks."""
        if not query or not query.strip():
            return []

        query_embedding = await self.embedding_client.embed_query(query)
        return await self.vector_store.search(
            query_embedding=query_embedding,
            limit=top_k,
            repo_id=repo_id,
        )

    async def answer_query(
        self,
        query: str,
        repo_id: str | None = None,
        top_k: int = 5,
    ) -> RAGResponse:
        """Retrieve relevant context and generate a grounded, cited answer."""
        if not query or not query.strip():
            raise RAGError("Query text cannot be empty.")

        retrieval_results = await self.retrieve(query, repo_id=repo_id, top_k=top_k)
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
        )
