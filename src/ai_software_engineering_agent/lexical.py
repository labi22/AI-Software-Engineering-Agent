"""Lexical search baseline using code-aware tokenization and Okapi BM25 ranking."""

from __future__ import annotations

from collections import Counter, defaultdict
import fnmatch
import math
import re
from typing import Sequence

from .models import CodeChunk, MetadataFilter, RetrievalResult

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


class CodeTokenizer:
    """Tokenizer aware of code syntax, CamelCase, snake_case, and symbol names."""

    # Regex for splitting camelCase and PascalCase identifiers
    _CAMEL_REGEX = re.compile(r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    _WORD_SPLIT_REGEX = re.compile(r"[^a-zA-Z0-9_]+")

    @classmethod
    def tokenize(cls, text: str, filter_stopwords: bool = True) -> list[str]:
        """Tokenize text into lowercased terms with CamelCase and snake_case decomposition."""
        if not text:
            return []

        tokens: list[str] = []
        raw_words = cls._WORD_SPLIT_REGEX.split(text)

        for raw_word in raw_words:
            if not raw_word:
                continue

            # Split snake_case
            subparts = raw_word.split("_")
            if len(subparts) > 1:
                tokens.append(raw_word.lower())

            for subpart in subparts:
                if not subpart:
                    continue

                # Split CamelCase
                matches = cls._CAMEL_REGEX.finditer(subpart)
                camel_pieces = [m.group(0).lower() for m in matches if m.group(0)]
                if len(camel_pieces) > 1:
                    tokens.append(subpart.lower())  # Include the combined word
                    tokens.extend(camel_pieces)    # Include split parts
                else:
                    tokens.append(subpart.lower())

        if filter_stopwords:
            return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
        return [t for t in tokens if len(t) > 1]


class BM25Index:
    """In-memory Okapi BM25 index with support for metadata filtering."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        self._chunks: dict[str, CodeChunk] = {}
        self._doc_lengths: dict[str, int] = {}
        self._doc_frequencies: Counter[str] = Counter()
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)  # term -> {chunk_id: count}
        self._total_docs: int = 0
        self._avg_doc_length: float = 0.0

    def index_chunks(self, chunks: Sequence[CodeChunk]) -> None:
        """Add and index a batch of code chunks into the BM25 inverted index."""
        if not chunks:
            return

        total_length = self._avg_doc_length * self._total_docs

        for chunk in chunks:
            # If replacing an existing chunk, remove its old statistics first
            if chunk.chunk_id in self._chunks:
                self._remove_chunk(chunk.chunk_id)

            # Combine content and symbol name for indexing
            text_to_index = f"{chunk.symbol_name or ''}\n{chunk.file_path}\n{chunk.content}"
            tokens = CodeTokenizer.tokenize(text_to_index)
            doc_len = len(tokens)

            self._chunks[chunk.chunk_id] = chunk
            self._doc_lengths[chunk.chunk_id] = doc_len
            total_length += doc_len
            self._total_docs += 1

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self._postings[term][chunk.chunk_id] = count
                self._doc_frequencies[term] += 1

        self._avg_doc_length = total_length / max(1, self._total_docs)

    def _remove_chunk(self, chunk_id: str) -> None:
        old_chunk = self._chunks.pop(chunk_id, None)
        if old_chunk is None:
            return

        old_len = self._doc_lengths.pop(chunk_id, 0)
        self._avg_doc_length = (
            (self._avg_doc_length * self._total_docs - old_len)
            / max(1, self._total_docs - 1)
        )
        self._total_docs -= 1

        # Remove postings
        for term in list(self._postings.keys()):
            if chunk_id in self._postings[term]:
                del self._postings[term][chunk_id]
                self._doc_frequencies[term] -= 1
                if self._doc_frequencies[term] <= 0:
                    del self._doc_frequencies[term]
                    del self._postings[term]

    def delete_repository(self, repo_id: str) -> None:
        """Remove all chunks associated with a repository."""
        to_delete = [
            cid for cid, chunk in self._chunks.items() if chunk.repo_id == repo_id
        ]
        for cid in to_delete:
            self._remove_chunk(cid)

    def _matches_filter(self, chunk: CodeChunk, filter: MetadataFilter | None) -> bool:
        if filter is None:
            return True

        if filter.repo_id and chunk.repo_id != filter.repo_id:
            return False

        if filter.languages and chunk.language not in filter.languages:
            return False

        if filter.symbol_only and not chunk.symbol_name:
            return False

        if filter.path_patterns:
            matches_pattern = any(
                fnmatch.fnmatch(chunk.file_path, pat) or fnmatch.fnmatch(f"*/{chunk.file_path}", pat)
                for pat in filter.path_patterns
            )
            if not matches_pattern:
                return False

        return True

    def search(
        self,
        query: str,
        limit: int = 10,
        filter: MetadataFilter | None = None,
    ) -> list[RetrievalResult]:
        """Compute BM25 scores for matching chunks and return ranked retrieval results."""
        if not query.strip() or self._total_docs == 0:
            return []

        query_tokens = CodeTokenizer.tokenize(query)
        if not query_tokens:
            return []

        scores: dict[str, float] = defaultdict(float)
        n_docs = self._total_docs
        avgdl = max(1.0, self._avg_doc_length)

        for term in query_tokens:
            if term not in self._postings:
                continue

            df = self._doc_frequencies[term]
            # Standard Okapi BM25 IDF formula with +1 smoothing
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            if idf <= 0:
                idf = 0.05  # Floor small or negative IDFs for ubiquitous terms

            for chunk_id, tf in self._postings[term].items():
                chunk = self._chunks.get(chunk_id)
                if not chunk or not self._matches_filter(chunk, filter):
                    continue

                doc_len = self._doc_lengths.get(chunk_id, 1)
                # BM25 Term Score
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                scores[chunk_id] += idf * (numerator / denominator)

        if not scores:
            return []

        # Sort candidate results descending by score
        ranked_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        results: list[RetrievalResult] = []
        for cid in ranked_chunk_ids[:limit]:
            chunk = self._chunks[cid]
            score = scores[cid]
            results.append(RetrievalResult(chunk=chunk, score=round(score, 4)))

        return results

    def count(self, repo_id: str | None = None) -> int:
        if repo_id is None:
            return len(self._chunks)
        return sum(1 for c in self._chunks.values() if c.repo_id == repo_id)
