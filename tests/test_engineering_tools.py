"""Tests for safe engineering tools: list_files, read_file, search_code, run_python, run_tests, git_diff."""

from pathlib import Path
import pytest

from ai_software_engineering_agent.agent_state import AgentState
from ai_software_engineering_agent.engineering_tools import (
    EngineeringToolContext,
    create_engineering_tool_registry,
    create_git_diff_handler,
    create_list_files_handler,
    create_read_file_handler,
    create_run_python_handler,
    create_run_tests_handler,
    create_search_code_handler,
)


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a sample repository directory with mock source files."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    src = repo / "src"
    src.mkdir()
    (src / "calc.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
        "\n"
        "def multiply(a: int, b: int) -> int:\n"
        "    return a * b\n",
        encoding="utf-8",
    )

    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_calc.py").write_text(
        "from src.calc import add, multiply\n"
        "\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
        "\n"
        "def test_multiply():\n"
        "    assert multiply(3, 4) == 12\n",
        encoding="utf-8",
    )

    # Ignored directory
    cache = repo / "__pycache__"
    cache.mkdir()
    (cache / "calc.cpython-311.pyc").write_bytes(b"\x00\x01\x02")

    return repo


@pytest.mark.asyncio
async def test_list_files_finds_files_and_ignores_caches(sample_repo: Path):
    context = EngineeringToolContext(repo_root=sample_repo)
    handler = create_list_files_handler(context)
    state = AgentState(task="test")

    res = await handler({"recursive": True}, state)
    assert "src/calc.py" in res
    assert "tests/test_calc.py" in res
    assert "__pycache__" not in res
    assert ".pyc" not in res


@pytest.mark.asyncio
async def test_list_files_rejects_path_traversal(sample_repo: Path):
    context = EngineeringToolContext(repo_root=sample_repo)
    handler = create_list_files_handler(context)
    state = AgentState(task="test")

    res = await handler({"path": "../outside"}, state)
    assert "Security Error" in res


@pytest.mark.asyncio
async def test_read_file_slices_lines_with_numbering(sample_repo: Path):
    context = EngineeringToolContext(repo_root=sample_repo)
    handler = create_read_file_handler(context)
    state = AgentState(task="test")

    # Read entire file
    res = await handler({"file_path": "src/calc.py"}, state)
    assert "def add" in res
    assert "1: def add" in res
    assert "4: def multiply" in res

    # Read line slice
    slice_res = await handler({"file_path": "src/calc.py", "start_line": 4, "end_line": 5}, state)
    assert "4: def multiply" in slice_res
    assert "1: def add" not in slice_res


@pytest.mark.asyncio
async def test_read_file_rejects_path_traversal_and_nonexistent(sample_repo: Path):
    context = EngineeringToolContext(repo_root=sample_repo)
    handler = create_read_file_handler(context)
    state = AgentState(task="test")

    res_traversal = await handler({"file_path": "../../secret.txt"}, state)
    assert "Security Error" in res_traversal

    res_missing = await handler({"file_path": "src/nonexistent.py"}, state)
    assert "Error: File 'src/nonexistent.py' does not exist." in res_missing


@pytest.mark.asyncio
async def test_search_code_finds_literal_and_regex(sample_repo: Path):
    context = EngineeringToolContext(repo_root=sample_repo)
    handler = create_search_code_handler(context)
    state = AgentState(task="test")

    # Literal search
    res = await handler({"pattern": "multiply"}, state)
    assert "src/calc.py:4:" in res
    assert "tests/test_calc.py:6:" in res

    # Regex search
    res_regex = await handler({"pattern": r"def\s+add", "is_regex": True}, state)
    assert "src/calc.py:1:" in res_regex

    # Path pattern filter
    res_filtered = await handler({"pattern": "def", "path_pattern": "tests/*.py"}, state)
    assert "tests/test_calc.py" in res_filtered
    assert "src/calc.py" not in res_filtered


@pytest.mark.asyncio
async def test_run_python_executes_code_and_captures_output(sample_repo: Path):
    context = EngineeringToolContext(repo_root=sample_repo)
    handler = create_run_python_handler(context)
    state = AgentState(task="test")

    # Test importing from repo root
    code = "from src.calc import add\nprint(f'SUM={add(10, 20)}')"
    res = await handler({"code": code}, state)
    assert "Exit Code: 0" in res
    assert "SUM=30" in res


@pytest.mark.asyncio
async def test_run_python_captures_errors_and_handles_timeout(sample_repo: Path):
    context = EngineeringToolContext(repo_root=sample_repo)
    handler = create_run_python_handler(context)
    state = AgentState(task="test")

    # Syntax error
    res_err = await handler({"code": "def invalid syntax"}, state)
    assert "Exit Code: 1" in res_err or "SyntaxError" in res_err

    # Timeout
    slow_code = "import time\ntime.sleep(2.0)"
    res_timeout = await handler({"code": slow_code, "timeout_seconds": 0.2}, state)
    assert "Execution Timeout Error" in res_timeout


@pytest.mark.asyncio
async def test_run_tests_runs_pytest_on_repository(sample_repo: Path):
    context = EngineeringToolContext(repo_root=sample_repo)
    handler = create_run_tests_handler(context)
    state = AgentState(task="test")

    res = await handler({"test_target": "tests/test_calc.py"}, state)
    assert "Pytest Run:" in res
    assert "passed" in res.lower()

    # Unauthorized argument rejected
    res_unauth = await handler({"extra_args": ["--override-ini=foo=bar"]}, state)
    assert "Security Error:" in res_unauth


@pytest.mark.asyncio
async def test_git_diff_handles_non_git_repo_cleanly(sample_repo: Path):
    context = EngineeringToolContext(repo_root=sample_repo)
    handler = create_git_diff_handler(context)
    state = AgentState(task="test")

    res = await handler({}, state)
    assert "is not a git repository" in res


def test_create_engineering_tool_registry_registers_all_six_tools(sample_repo: Path):
    context = EngineeringToolContext(repo_root=sample_repo)
    registry = create_engineering_tool_registry(context)

    tool_names = set(registry.list_names())
    expected = {"list_files", "read_file", "search_code", "run_python", "run_tests", "git_diff", "final_answer"}
    assert expected.issubset(tool_names)
