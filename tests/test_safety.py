"""Tests for security guardrails, path validation, and command sanitization."""

from pathlib import Path
import pytest

from ai_software_engineering_agent.safety import (
    CommandSecurityError,
    PathSecurityError,
    sanitize_git_ref,
    sanitize_pytest_args,
    truncate_output,
    validate_safe_path,
)


def test_validate_safe_path_accepts_valid_internal_paths(tmp_path: Path):
    repo_root = tmp_path / "my_repo"
    repo_root.mkdir()
    sub_dir = repo_root / "src"
    sub_dir.mkdir()
    file_path = sub_dir / "app.py"
    file_path.write_text("print('hello')", encoding="utf-8")

    # Relative path
    res = validate_safe_path("src/app.py", repo_root=repo_root)
    assert res == file_path.resolve()

    # Empty path resolves to repo_root
    assert validate_safe_path("", repo_root=repo_root) == repo_root.resolve()

    # Current dir "." resolves to repo_root
    assert validate_safe_path(".", repo_root=repo_root) == repo_root.resolve()


def test_validate_safe_path_rejects_directory_traversal(tmp_path: Path):
    repo_root = tmp_path / "my_repo"
    repo_root.mkdir()

    # Traversal via ..
    with pytest.raises(PathSecurityError, match="escapes repository boundary"):
        validate_safe_path("../outside.py", repo_root=repo_root)

    with pytest.raises(PathSecurityError, match="escapes repository boundary"):
        validate_safe_path("src/../../outside.py", repo_root=repo_root)


def test_validate_safe_path_rejects_absolute_path_outside_root(tmp_path: Path):
    repo_root = tmp_path / "my_repo"
    repo_root.mkdir()
    outside = tmp_path / "other_repo" / "secret.txt"

    with pytest.raises(PathSecurityError, match="escapes repository boundary"):
        validate_safe_path(outside, repo_root=repo_root)


def test_validate_safe_path_enforces_allowed_roots(tmp_path: Path):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    repo_root = allowed_dir / "repo"
    repo_root.mkdir()

    unallowed_repo = tmp_path / "forbidden_repo"
    unallowed_repo.mkdir()

    # Within allowed roots
    assert validate_safe_path("test.py", repo_root=repo_root, allowed_roots=[allowed_dir])

    # Outside allowed roots
    with pytest.raises(PathSecurityError, match="outside ALLOWED_REPOSITORY_ROOTS"):
        validate_safe_path("test.py", repo_root=unallowed_repo, allowed_roots=[allowed_dir])


def test_validate_safe_path_rejects_nonexistent_root(tmp_path: Path):
    fake_root = tmp_path / "does_not_exist"
    with pytest.raises(PathSecurityError, match="does not exist"):
        validate_safe_path("test.py", repo_root=fake_root)


def test_truncate_output_preserves_short_text():
    text = "Short output"
    assert truncate_output(text, max_chars=100) == text


def test_truncate_output_caps_long_text():
    text = "A" * 1000
    truncated = truncate_output(text, max_chars=200)
    assert len(truncated) > 200  # includes notice
    assert truncated.startswith("A" * 200)
    assert "[... Truncated 800 characters to preserve context ...]" in truncated


def test_sanitize_git_ref_accepts_valid_references():
    assert sanitize_git_ref("main") == "main"
    assert sanitize_git_ref("feature/foo-bar_1") == "feature/foo-bar_1"
    assert sanitize_git_ref("HEAD~1") == "HEAD~1"
    assert sanitize_git_ref("v1.2.3^0") == "v1.2.3^0"
    assert sanitize_git_ref("a1b2c3d4e5") == "a1b2c3d4e5"


def test_sanitize_git_ref_rejects_dangerous_inputs():
    with pytest.raises(CommandSecurityError):
        sanitize_git_ref("")

    with pytest.raises(CommandSecurityError, match="cannot begin with a dash"):
        sanitize_git_ref("--option")

    with pytest.raises(CommandSecurityError, match="Invalid or unsafe git ref"):
        sanitize_git_ref("main; rm -rf /")

    with pytest.raises(CommandSecurityError, match="Invalid or unsafe git ref"):
        sanitize_git_ref("HEAD && calc.exe")


def test_sanitize_pytest_args_accepts_whitelisted_flags():
    args = ["-v", "-q", "-x", "--tb=short"]
    sanitized = sanitize_pytest_args(args)
    assert sanitized == ["-v", "-q", "-x", "--tb=short"]


def test_sanitize_pytest_args_accepts_valid_pattern_options():
    args = ["-k", "test_addition", "-v"]
    sanitized = sanitize_pytest_args(args)
    assert sanitized == ["-k", "test_addition", "-v"]

    args_eq = ["-k=test_subtraction"]
    assert sanitize_pytest_args(args_eq) == ["-k=test_subtraction"]


def test_sanitize_pytest_args_rejects_unauthorized_and_dangerous_flags():
    # Dangerous arbitrary pytest flag
    with pytest.raises(CommandSecurityError, match="Unauthorized pytest argument"):
        sanitize_pytest_args(["--override-ini=cache_dir=/tmp"])

    with pytest.raises(CommandSecurityError, match="Unauthorized pytest argument"):
        sanitize_pytest_args(["--pdb"])

    # Injection in -k value
    with pytest.raises(CommandSecurityError, match="Dangerous characters detected"):
        sanitize_pytest_args(["-k", "test_name; rm -rf"])
