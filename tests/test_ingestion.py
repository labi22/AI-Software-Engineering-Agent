"""Tests for repository discovery, filtering, and AST/line-aware code chunking."""

from pathlib import Path
import pytest

from ai_software_engineering_agent.ingestion import (
    RepositoryAccessError,
    chunk_document,
    discover_files,
    parse_file,
    validate_repository_path,
)
from ai_software_engineering_agent.models import FileDocument


def test_validate_repository_path_rejects_nonexistent_directory(tmp_path: Path):
    non_existent = tmp_path / "non_existent_dir"
    with pytest.raises(RepositoryAccessError, match="does not exist"):
        validate_repository_path(non_existent)


def test_validate_repository_path_enforces_allowed_roots(tmp_path: Path):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    forbidden_dir = tmp_path / "forbidden"
    forbidden_dir.mkdir()

    # Allowed dir succeeds
    valid = validate_repository_path(allowed_dir, allowed_roots=[allowed_dir])
    assert valid == allowed_dir.resolve()

    # Forbidden dir fails
    with pytest.raises(RepositoryAccessError, match="not within ALLOWED_REPOSITORY_ROOTS"):
        validate_repository_path(forbidden_dir, allowed_roots=[allowed_dir])


def test_discover_files_ignores_excluded_dirs_and_binaries(tmp_path: Path):
    repo = tmp_path / "test_repo"
    repo.mkdir()

    # Create allowed files
    (repo / "main.py").write_text("print('hello')", encoding="utf-8")
    (repo / "README.md").write_text("# Test Repo", encoding="utf-8")
    sub = repo / "pkg"
    sub.mkdir()
    (sub / "logic.py").write_text("def run(): pass", encoding="utf-8")

    # Create excluded directories and files
    git_dir = repo / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config", encoding="utf-8")

    pycache_dir = repo / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "main.cpython-311.pyc").write_bytes(b"\x00\x01\x02")

    (repo / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    discovered = discover_files(repo)
    file_names = [p.name for p in discovered]

    assert "main.py" in file_names
    assert "README.md" in file_names
    assert "logic.py" in file_names
    assert "config" not in file_names
    assert "main.cpython-311.pyc" not in file_names
    assert "image.png" not in file_names


def test_parse_file_extracts_metadata(tmp_path: Path):
    file_path = tmp_path / "calc.py"
    file_path.write_text("x = 10\ny = 20\nprint(x + y)\n", encoding="utf-8")

    doc = parse_file(file_path, tmp_path)
    assert doc is not None
    assert doc.language == "python"
    assert doc.line_count == 3
    assert doc.relative_path == "calc.py"
    assert doc.size_bytes > 0


def test_chunk_document_python_ast_extracts_functions_and_classes():
    code = (
        "import math\n"
        "\n"
        "class YieldCalculator:\n"
        "    \"\"\"Calculates bond yield.\"\"\"\n"
        "    def __init__(self, face_value: float):\n"
        "        self.face_value = face_value\n"
        "\n"
        "    def calculate_ytm(self, price: float, rate: float) -> float:\n"
        "        return (rate * self.face_value) / price\n"
        "\n"
        "def discount_factor(rate: float, t: float) -> float:\n"
        "    return math.exp(-rate * t)\n"
    )
    doc = FileDocument(
        file_path=Path("models/yield.py"),
        relative_path="models/yield.py",
        language="python",
        content=code,
        line_count=len(code.splitlines()),
        size_bytes=len(code.encode("utf-8")),
    )

    chunks = chunk_document(doc, repo_id="bond-lab")
    assert len(chunks) >= 3

    symbols = [c.symbol_name for c in chunks if c.symbol_name]
    assert "YieldCalculator" in symbols or any("YieldCalculator" in (s or "") for s in symbols)
    assert any("calculate_ytm" in (s or "") for s in symbols)
    assert any("discount_factor" in (s or "") for s in symbols)

    for c in chunks:
        assert c.repo_id == "bond-lab"
        assert c.file_path == "models/yield.py"
        assert c.start_line <= c.end_line
        assert len(c.content) > 0


def test_chunk_document_sliding_window_for_markdown():
    text = "\n".join([f"Line {i}: Documentation content about topic {i}" for i in range(1, 101)])
    doc = FileDocument(
        file_path=Path("docs/guide.md"),
        relative_path="docs/guide.md",
        language="markdown",
        content=text,
        line_count=100,
        size_bytes=len(text.encode("utf-8")),
    )

    chunks = chunk_document(doc, repo_id="test-repo", max_lines=40, overlap=10)
    assert len(chunks) >= 3
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 40
    assert chunks[1].start_line == 31  # 40 - 10 + 1
