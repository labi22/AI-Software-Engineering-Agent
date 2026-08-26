"""Repository discovery, file parsing, and syntax/line-aware code chunking."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from typing import Sequence

from .models import CodeChunk, FileDocument, RepositorySpec

# Directories and files to exclude from code ingestion
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    "dist",
    "build",
    ".eggs",
    ".idea",
    ".vscode",
    ".coverage",
    "htmlcov",
}

DEFAULT_EXCLUDED_EXTENSIONS = {
    # Binaries, compiled, caches
    ".pyc",
    ".pyo",
    ".pyd",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".class",
    ".jar",
    # Images & Media
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".webp",
    ".mp3",
    ".mp4",
    ".pdf",
    # Archives
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    # Data & Database dumps
    ".db",
    ".sqlite",
    ".sqlite3",
    ".parquet",
    ".pkl",
    ".pickle",
    ".h5",
    ".hdf5",
    # Fonts
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    # Lockfiles & minified assets
    ".min.js",
    ".min.css",
}

LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "restructuredtext",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".ps1": "powershell",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".html": "html",
    ".css": "css",
    ".txt": "text",
    ".csv": "csv",
}

MAX_FILE_SIZE_BYTES = 500 * 1024  # 500 KB limit per file


class IngestionError(Exception):
    """Base error for repository ingestion and parsing."""


class RepositoryAccessError(IngestionError):
    """Raised when repository path is unauthorized or unreadable."""


def validate_repository_path(
    path: Path | str,
    allowed_roots: Sequence[Path] = (),
) -> Path:
    """Validate that the repository path exists, is a directory, and is inside allowed roots."""
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise RepositoryAccessError(f"Repository path does not exist: {resolved_path}")
    if not resolved_path.is_dir():
        raise RepositoryAccessError(f"Repository path is not a directory: {resolved_path}")

    if allowed_roots:
        resolved_allowed = [p.expanduser().resolve() for p in allowed_roots]
        is_allowed = any(
            resolved_path == allowed or allowed in resolved_path.parents
            for allowed in resolved_allowed
        )
        if not is_allowed:
            raise RepositoryAccessError(
                f"Repository path '{resolved_path}' is not within ALLOWED_REPOSITORY_ROOTS."
            )

    return resolved_path


def discover_files(
    repo_path: Path | str,
    allowed_roots: Sequence[Path] = (),
) -> list[Path]:
    """Discover all supported text and source code files in the repository."""
    valid_root = validate_repository_path(repo_path, allowed_roots)
    found_files: list[Path] = []

    for root, dirs, files in os.walk(valid_root):
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in DEFAULT_EXCLUDED_DIRS and not d.endswith(".egg-info")]

        for filename in files:
            file_path = Path(root) / filename
            suffix = file_path.suffix.lower()

            if suffix in DEFAULT_EXCLUDED_EXTENSIONS:
                continue
            if any(file_path.name.endswith(ext) for ext in DEFAULT_EXCLUDED_EXTENSIONS):
                continue
            if filename.startswith(".") and suffix != ".env":
                continue

            try:
                if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue

            found_files.append(file_path)

    return sorted(found_files)


def parse_file(file_path: Path, repo_root: Path) -> FileDocument | None:
    """Parse a single file into a FileDocument, returning None if unreadable or binary."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None

    try:
        relative_path = file_path.relative_to(repo_root).as_posix()
    except ValueError:
        relative_path = file_path.name

    extension = file_path.suffix.lower()
    language = LANGUAGE_EXTENSIONS.get(extension, "text")
    lines = content.splitlines()

    return FileDocument(
        file_path=file_path,
        relative_path=relative_path,
        language=language,
        content=content,
        line_count=len(lines),
        size_bytes=len(content.encode("utf-8")),
    )


def _generate_chunk_id(repo_id: str, relative_path: str, start_line: int, end_line: int) -> str:
    key = f"{repo_id}:{relative_path}:{start_line}:{end_line}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _estimate_tokens(text: str) -> int:
    """Fast estimate of tokens based on word and character counts."""
    return max(1, len(text) // 4)


def chunk_document(
    doc: FileDocument,
    repo_id: str,
    max_lines: int = 60,
    overlap: int = 15,
) -> list[CodeChunk]:
    """Chunk a FileDocument using AST/symbol extraction for Python or sliding window for other files."""
    if not doc.content.strip():
        return []

    if doc.language == "python":
        chunks = _chunk_python_ast(doc, repo_id, max_lines, overlap)
        if chunks:
            return chunks

    return _chunk_sliding_window(doc, repo_id, max_lines, overlap)


def _chunk_python_ast(
    doc: FileDocument,
    repo_id: str,
    max_lines: int,
    overlap: int,
) -> list[CodeChunk]:
    """Extract AST-aware functions, classes, and top-level blocks from Python code."""
    try:
        tree = ast.parse(doc.content, filename=doc.relative_path)
    except SyntaxError:
        return []

    lines = doc.content.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return []

    chunks: list[CodeChunk] = []
    covered_lines: set[int] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", node.lineno)
            symbol_name = node.name

            # Include decorators if present
            if node.decorator_list:
                start_line = min(d.lineno for d in node.decorator_list)

            # If node is a class, also inspect its methods for fine-grained indexing
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_start = item.lineno
                        if item.decorator_list:
                            method_start = min(d.lineno for d in item.decorator_list)
                        method_end = getattr(item, "end_lineno", item.lineno)
                        method_symbol = f"{symbol_name}.{item.name}"
                        method_lines = lines[method_start - 1 : method_end]
                        method_content = "\n".join(method_lines)

                        chunk_id = _generate_chunk_id(
                            repo_id, doc.relative_path, method_start, method_end
                        )
                        chunks.append(
                            CodeChunk(
                                chunk_id=chunk_id,
                                repo_id=repo_id,
                                file_path=doc.relative_path,
                                start_line=method_start,
                                end_line=method_end,
                                symbol_name=method_symbol,
                                language="python",
                                content=method_content,
                                token_count=_estimate_tokens(method_content),
                            )
                        )

            node_lines = lines[start_line - 1 : end_line]
            node_content = "\n".join(node_lines)
            
            # If the entire function/class fits within reasonable size, save as a chunk
            if (end_line - start_line + 1) <= max_lines * 2:
                chunk_id = _generate_chunk_id(repo_id, doc.relative_path, start_line, end_line)
                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        repo_id=repo_id,
                        file_path=doc.relative_path,
                        start_line=start_line,
                        end_line=end_line,
                        symbol_name=symbol_name,
                        language="python",
                        content=node_content,
                        token_count=_estimate_tokens(node_content),
                    )
                )
            else:
                # Subchunk large classes/functions using sliding window
                subchunks = _chunk_lines(
                    lines=node_lines,
                    base_start=start_line,
                    repo_id=repo_id,
                    relative_path=doc.relative_path,
                    language="python",
                    symbol_name=symbol_name,
                    max_lines=max_lines,
                    overlap=overlap,
                )
                chunks.extend(subchunks)

            for ln in range(start_line, end_line + 1):
                covered_lines.add(ln)

    # Capture uncovered top-level statements (module docstring, imports, constants, scripts)
    uncovered_blocks: list[tuple[int, int]] = []
    current_start: int | None = None

    for i in range(1, total_lines + 1):
        if i not in covered_lines:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                uncovered_blocks.append((current_start, i - 1))
                current_start = None
    if current_start is not None:
        uncovered_blocks.append((current_start, total_lines))

    for start_l, end_l in uncovered_blocks:
        block_lines = lines[start_l - 1 : end_l]
        block_content = "\n".join(block_lines).strip()
        if not block_content:
            continue
        subchunks = _chunk_lines(
            lines=block_lines,
            base_start=start_l,
            repo_id=repo_id,
            relative_path=doc.relative_path,
            language="python",
            symbol_name=None,
            max_lines=max_lines,
            overlap=overlap,
        )
        chunks.extend(subchunks)

    return sorted(chunks, key=lambda c: (c.start_line, c.end_line))


def _chunk_sliding_window(
    doc: FileDocument,
    repo_id: str,
    max_lines: int,
    overlap: int,
) -> list[CodeChunk]:
    """Chunk document lines with a sliding window and overlap."""
    lines = doc.content.splitlines()
    if not lines:
        return []
    return _chunk_lines(
        lines=lines,
        base_start=1,
        repo_id=repo_id,
        relative_path=doc.relative_path,
        language=doc.language,
        symbol_name=None,
        max_lines=max_lines,
        overlap=overlap,
    )


def _chunk_lines(
    lines: list[str],
    base_start: int,
    repo_id: str,
    relative_path: str,
    language: str,
    symbol_name: str | None,
    max_lines: int,
    overlap: int,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    total = len(lines)
    if total == 0:
        return []

    step = max(1, max_lines - overlap)
    for idx in range(0, total, step):
        chunk_slice = lines[idx : min(idx + max_lines, total)]
        content = "\n".join(chunk_slice)
        if not content.strip():
            continue

        start_line = base_start + idx
        end_line = base_start + min(idx + max_lines, total) - 1
        chunk_id = _generate_chunk_id(repo_id, relative_path, start_line, end_line)

        chunks.append(
            CodeChunk(
                chunk_id=chunk_id,
                repo_id=repo_id,
                file_path=relative_path,
                start_line=start_line,
                end_line=end_line,
                symbol_name=symbol_name,
                language=language,
                content=content,
                token_count=_estimate_tokens(content),
            )
        )
        if idx + max_lines >= total:
            break

    return chunks
