"""Optional Tree-sitter symbol extraction for non-Python source files.

The runtime and each grammar are intentionally optional. If they are absent, a
grammar rejects a file, or the source has syntax errors, callers receive an
empty result and use the repository's safe line-window fallback instead.
"""

from __future__ import annotations

import hashlib
import importlib
from typing import Any

from .models import CodeChunk, FileDocument


_GRAMMAR_MODULES: dict[str, tuple[str, str]] = {
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "java": ("tree_sitter_java", "language"),
    "go": ("tree_sitter_go", "language"),
    "csharp": ("tree_sitter_c_sharp", "language"),
}

_SYMBOL_NODE_TYPES: dict[str, set[str]] = {
    "javascript": {"class_declaration", "function_declaration", "method_definition"},
    "typescript": {
        "class_declaration", "interface_declaration", "enum_declaration",
        "function_declaration", "method_definition",
    },
    "tsx": {
        "class_declaration", "interface_declaration", "enum_declaration",
        "function_declaration", "method_definition",
    },
    "java": {
        "class_declaration", "interface_declaration", "enum_declaration",
        "annotation_type_declaration", "method_declaration", "constructor_declaration",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration", "type_spec"},
    "csharp": {
        "class_declaration", "interface_declaration", "struct_declaration", "enum_declaration",
        "record_declaration", "method_declaration", "constructor_declaration",
    },
}

_SCOPE_NODE_TYPES = {
    "class_declaration", "interface_declaration", "struct_declaration", "enum_declaration",
    "record_declaration", "annotation_type_declaration",
}


def tree_sitter_available(language: str) -> bool:
    """Return whether the runtime and grammar for a language can be imported."""
    if language not in _GRAMMAR_MODULES:
        return False
    try:
        importlib.import_module("tree_sitter")
        importlib.import_module(_GRAMMAR_MODULES[language][0])
    except ImportError:
        return False
    return True


def _chunk_id(repo_id: str, path: str, start_line: int, end_line: int) -> str:
    return hashlib.sha256(f"{repo_id}:{path}:{start_line}:{end_line}".encode()).hexdigest()[:16]


def _token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _node_name(node: Any, source: bytes) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return source[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="replace")
    for child in node.named_children:
        if child.type in {"identifier", "type_identifier", "property_identifier"}:
            return source[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
    return None


def _line_chunks(
    lines: list[str], repo_id: str, path: str, language: str, start_line: int,
    max_lines: int, overlap: int, symbol_name: str | None = None,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    step = max(1, max_lines - overlap)
    for index in range(0, len(lines), step):
        segment = lines[index : index + max_lines]
        content = "\n".join(segment)
        if content.strip():
            first = start_line + index
            last = first + len(segment) - 1
            chunks.append(CodeChunk(
                chunk_id=_chunk_id(repo_id, path, first, last), repo_id=repo_id,
                file_path=path, start_line=first, end_line=last,
                symbol_name=symbol_name, language=language, content=content,
                token_count=_token_count(content),
            ))
        if index + max_lines >= len(lines):
            break
    return chunks


def chunk_tree_sitter_document(
    doc: FileDocument, repo_id: str, max_lines: int, overlap: int,
) -> list[CodeChunk]:
    """Return declaration-aware chunks, or ``[]`` to request line-window fallback."""
    grammar_info = _GRAMMAR_MODULES.get(doc.language)
    if grammar_info is None:
        return []
    try:
        from tree_sitter import Language, Parser

        module = importlib.import_module(grammar_info[0])
        language = Language(getattr(module, grammar_info[1])())
        tree = Parser(language).parse(doc.content.encode("utf-8"))
    except (ImportError, AttributeError, TypeError, ValueError):
        return []

    # Syntax-error recovery is useful to editors, but not reliable enough for
    # retrieval symbols. Preserve all text through the existing fallback.
    if tree.root_node.has_error:
        return []

    source = doc.content.encode("utf-8")
    lines = doc.content.splitlines()
    declarations: list[tuple[Any, str | None]] = []

    def visit(node: Any, scope: str | None = None) -> None:
        node_name = _node_name(node, source)
        next_scope = node_name if node.type in _SCOPE_NODE_TYPES and node_name else scope
        if node.type in _SYMBOL_NODE_TYPES[doc.language]:
            # Go's type_declaration is a wrapper around type_spec. Keep only
            # type_spec to avoid duplicate chunks with an ambiguous name.
            if not (doc.language == "go" and node.type == "type_declaration"):
                symbol = node_name
                if scope and node.type in {"method_definition", "method_declaration", "constructor_declaration"} and symbol:
                    symbol = f"{scope}.{symbol}"
                declarations.append((node, symbol))
        for child in node.named_children:
            visit(child, next_scope)

    visit(tree.root_node)
    if not declarations:
        return []

    chunks: list[CodeChunk] = []
    covered_lines: set[int] = set()
    for node, symbol_name in declarations:
        start_line, end_line = node.start_point.row + 1, node.end_point.row + 1
        node_lines = lines[start_line - 1 : end_line]
        if end_line - start_line + 1 <= max_lines * 2:
            content = "\n".join(node_lines)
            chunks.append(CodeChunk(
                chunk_id=_chunk_id(repo_id, doc.relative_path, start_line, end_line),
                repo_id=repo_id, file_path=doc.relative_path, start_line=start_line,
                end_line=end_line, symbol_name=symbol_name, language=doc.language,
                content=content, token_count=_token_count(content),
            ))
        else:
            chunks.extend(_line_chunks(
                node_lines, repo_id, doc.relative_path, doc.language, start_line,
                max_lines, overlap, symbol_name,
            ))
        covered_lines.update(range(start_line, end_line + 1))

    # Retain imports, package declarations, constants, and comments that do not
    # belong to a declaration, mirroring the existing Python AST strategy.
    block_start: int | None = None
    for line_number in range(1, len(lines) + 1):
        if line_number not in covered_lines and block_start is None:
            block_start = line_number
        elif line_number in covered_lines and block_start is not None:
            chunks.extend(_line_chunks(
                lines[block_start - 1 : line_number - 1], repo_id, doc.relative_path,
                doc.language, block_start, max_lines, overlap,
            ))
            block_start = None
    if block_start is not None:
        chunks.extend(_line_chunks(
            lines[block_start - 1 :], repo_id, doc.relative_path, doc.language,
            block_start, max_lines, overlap,
        ))
    return sorted(chunks, key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.symbol_name or ""))
