"""Safe engineering tools for codebase inspection, navigation, execution, testing, and diffing."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .agent_state import AgentState
from .ingestion import DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_EXTENSIONS, MAX_FILE_SIZE_BYTES
from .safety import (
    CommandSecurityError,
    PathSecurityError,
    sanitize_git_ref,
    sanitize_pytest_args,
    truncate_output,
    validate_safe_path,
)
from .tools import ToolHandler, ToolRegistry, ToolSchema, create_default_tool_registry


@dataclass(frozen=True)
class EngineeringToolContext:
    """Execution context and configuration boundaries for engineering tools."""

    repo_root: Path
    allowed_roots: tuple[Path, ...] = ()
    python_executable: str = sys.executable
    default_timeout_seconds: float = 30.0
    max_output_chars: int = 8000
    rag_service: Any | None = None


# ---------------------------------------------------------------------------
# Tool Schemas
# ---------------------------------------------------------------------------

LIST_FILES_SCHEMA = ToolSchema(
    name="list_files",
    description=(
        "List files and directories within the repository. Supports recursive directory "
        "exploration with depth controls, excluding build and cache directories."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative directory path within repository to list (default: root '').",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to list recursively through subdirectories (default: true).",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum directory depth for recursive listing (default: 5).",
            },
            "max_files": {
                "type": "integer",
                "description": "Maximum number of file entries to return (default: 100, max: 500).",
            },
        },
    },
)

READ_FILE_SCHEMA = ToolSchema(
    name="read_file",
    description=(
        "Read file content with line numbers from the repository. Supports slicing specific "
        "line ranges and validates that the file is safe and within the target repository."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path to the file within the repository.",
            },
            "start_line": {
                "type": "integer",
                "description": "Optional 1-indexed starting line number (inclusive).",
            },
            "end_line": {
                "type": "integer",
                "description": "Optional 1-indexed ending line number (inclusive).",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to retrieve (default: 200, max: 500).",
            },
        },
        "required": ["file_path"],
    },
)

SEARCH_CODE_SCHEMA = ToolSchema(
    name="search_code",
    description=(
        "Perform a fast text or regex grep search across repository files. Returns matched "
        "file paths, line numbers, and code lines."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Literal string or regular expression pattern to search for.",
            },
            "is_regex": {
                "type": "boolean",
                "description": "Whether the pattern is a regular expression (default: false).",
            },
            "path_pattern": {
                "type": "string",
                "description": "Optional glob filter for file paths, e.g. '*.py' or 'src/**'.",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether the search is case-sensitive (default: false).",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines to return (default: 30, max: 100).",
            },
        },
        "required": ["pattern"],
    },
)

RUN_PYTHON_SCHEMA = ToolSchema(
    name="run_python",
    description=(
        "Execute a Python script or snippet in a sandboxed subprocess with working directory "
        "pinned to the repository root. Captures stdout, stderr, and exit code."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code string to execute.",
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Maximum execution time in seconds (default: 10.0, max: 30.0).",
            },
        },
        "required": ["code"],
    },
)

RUN_TESTS_SCHEMA = ToolSchema(
    name="run_tests",
    description=(
        "Run the repository test suite using pytest. Executes in a subprocess within the repository "
        "and returns a summary of passing, failing, or errored tests."
    ),
    parameters={
        "type": "object",
        "properties": {
            "test_target": {
                "type": "string",
                "description": "Optional specific test file or directory path relative to repository.",
            },
            "extra_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional whitelisted pytest flags (e.g. ['-k', 'test_name', '-v']).",
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Execution timeout in seconds (default: 30.0, max: 60.0).",
            },
        },
    },
)

GIT_DIFF_SCHEMA = ToolSchema(
    name="git_diff",
    description=(
        "View read-only git diff of working tree changes or compare against a commit/branch "
        "in the repository."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Optional specific file path relative to repository to inspect.",
            },
            "staged": {
                "type": "boolean",
                "description": "Whether to view staged (--cached) changes (default: false).",
            },
            "commit": {
                "type": "string",
                "description": "Optional commit hash, tag, or branch to diff against (e.g. 'HEAD~1').",
            },
        },
    },
)


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------

def create_list_files_handler(context: EngineeringToolContext) -> ToolHandler:
    """Create list_files handler bound to an EngineeringToolContext."""

    async def list_files_handler(arguments: dict[str, Any], state: AgentState) -> str:
        raw_path = arguments.get("path", "")
        recursive = bool(arguments.get("recursive", True))
        max_depth = int(arguments.get("max_depth", 5))
        max_files = min(int(arguments.get("max_files", 100)), 500)

        try:
            target_dir = validate_safe_path(
                raw_path,
                repo_root=context.repo_root,
                allowed_roots=context.allowed_roots,
            )
        except PathSecurityError as error:
            return f"Security Error: {error}"

        if not target_dir.is_dir():
            return f"Error: '{raw_path}' is not a directory."

        found_entries: list[tuple[str, int, bool]] = []  # (relative_path, size, is_dir)

        if not recursive:
            try:
                for entry in sorted(target_dir.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
                    if entry.name in DEFAULT_EXCLUDED_DIRS:
                        continue
                    if entry.name.startswith(".") and entry.name != ".env":
                        continue
                    rel_p = entry.relative_to(context.repo_root).as_posix()
                    is_d = entry.is_dir()
                    sz = 0 if is_d else entry.stat().st_size
                    found_entries.append((rel_p, sz, is_d))
                    if len(found_entries) >= max_files:
                        break
            except OSError as exc:
                return f"Error reading directory: {exc}"
        else:
            base_depth = len(target_dir.parts)
            for root, dirs, files in os.walk(target_dir):
                current_depth = len(Path(root).parts) - base_depth
                if current_depth >= max_depth:
                    dirs.clear()
                    continue

                # Exclude directories in-place
                dirs[:] = [
                    d for d in dirs
                    if d not in DEFAULT_EXCLUDED_DIRS and not d.startswith(".") and not d.endswith(".egg-info")
                ]

                for f in sorted(files):
                    if len(found_entries) >= max_files:
                        break
                    file_path = Path(root) / f
                    if file_path.suffix.lower() in DEFAULT_EXCLUDED_EXTENSIONS:
                        continue
                    if f.startswith(".") and f != ".env":
                        continue

                    try:
                        sz = file_path.stat().st_size
                    except OSError:
                        sz = 0

                    rel_p = file_path.relative_to(context.repo_root).as_posix()
                    found_entries.append((rel_p, sz, False))

                if len(found_entries) >= max_files:
                    break

        if not found_entries:
            return f"No files found in '{raw_path or '.'}'."

        lines = [f"Files in '{raw_path or '.'}' (Total: {len(found_entries)}):"]
        for p, sz, is_d in found_entries:
            if is_d:
                lines.append(f"  [DIR]  {p}/")
            else:
                lines.append(f"  [FILE] {p} ({sz:,} bytes)")

        if len(found_entries) >= max_files:
            lines.append(f"\n[Note: Result capped at {max_files} entries]")

        return "\n".join(lines)

    return list_files_handler


def create_read_file_handler(context: EngineeringToolContext) -> ToolHandler:
    """Create read_file handler bound to an EngineeringToolContext."""

    async def read_file_handler(arguments: dict[str, Any], state: AgentState) -> str:
        file_path_arg = arguments.get("file_path", "")
        if not file_path_arg:
            return "Error: 'file_path' argument is required."

        try:
            target_file = validate_safe_path(
                file_path_arg,
                repo_root=context.repo_root,
                allowed_roots=context.allowed_roots,
            )
        except PathSecurityError as error:
            return f"Security Error: {error}"

        if not target_file.exists():
            return f"Error: File '{file_path_arg}' does not exist."
        if not target_file.is_file():
            return f"Error: '{file_path_arg}' is a directory, not a file."

        if target_file.suffix.lower() in DEFAULT_EXCLUDED_EXTENSIONS:
            return f"Error: File '{file_path_arg}' has binary or excluded extension ({target_file.suffix})."

        try:
            file_stat = target_file.stat()
            if file_stat.st_size > MAX_FILE_SIZE_BYTES:
                return f"Error: File '{file_path_arg}' is too large ({file_stat.st_size:,} bytes > {MAX_FILE_SIZE_BYTES:,} bytes limit)."
            content = target_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: File '{file_path_arg}' appears to be binary and cannot be decoded as UTF-8."
        except OSError as exc:
            return f"Error reading file '{file_path_arg}': {exc}"

        all_lines = content.splitlines()
        total_lines = len(all_lines)

        start_line = max(1, int(arguments.get("start_line", 1)))
        max_lines = min(int(arguments.get("max_lines", 200)), 500)

        end_line_arg = arguments.get("end_line")
        if end_line_arg is not None:
            end_line = min(total_lines, int(end_line_arg))
        else:
            end_line = min(total_lines, start_line + max_lines - 1)

        if start_line > total_lines:
            return f"Error: start_line ({start_line}) exceeds total file lines ({total_lines})."
        if start_line > end_line:
            return f"Error: start_line ({start_line}) cannot be greater than end_line ({end_line})."

        # Cap lines retrieved
        if (end_line - start_line + 1) > max_lines:
            end_line = start_line + max_lines - 1

        selected_slice = all_lines[start_line - 1 : end_line]
        numbered_lines = [
            f"{line_num:4d}: {line}"
            for line_num, line in enumerate(selected_slice, start=start_line)
        ]

        header = f"File: {file_path_arg} (Lines {start_line}-{end_line} of {total_lines})"
        return f"{header}\n```\n" + "\n".join(numbered_lines) + "\n```"

    return read_file_handler


def create_search_code_handler(context: EngineeringToolContext) -> ToolHandler:
    """Create search_code handler for literal or regex grep search across repository files."""

    async def search_code_handler(arguments: dict[str, Any], state: AgentState) -> str:
        pattern = arguments.get("pattern", "")
        if not pattern:
            return "Error: 'pattern' argument is required."

        is_regex = bool(arguments.get("is_regex", False))
        path_pattern = arguments.get("path_pattern", "")
        case_sensitive = bool(arguments.get("case_sensitive", False))
        max_results = min(int(arguments.get("max_results", 30)), 100)

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            if is_regex:
                compiled_regex = re.compile(pattern, flags)
            else:
                compiled_regex = re.compile(re.escape(pattern), flags)
        except re.error as exc:
            return f"Regex Error: Invalid pattern '{pattern}': {exc}"

        matches: list[str] = []

        for root, dirs, files in os.walk(context.repo_root):
            dirs[:] = [
                d for d in dirs
                if d not in DEFAULT_EXCLUDED_DIRS and not d.startswith(".") and not d.endswith(".egg-info")
            ]

            for f in sorted(files):
                file_path = Path(root) / f
                if file_path.suffix.lower() in DEFAULT_EXCLUDED_EXTENSIONS:
                    continue
                if f.startswith(".") and f != ".env":
                    continue

                rel_p = file_path.relative_to(context.repo_root).as_posix()
                if path_pattern and not fnmatch.fnmatch(rel_p, path_pattern):
                    continue

                try:
                    if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                        continue
                    text = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue

                for line_no, line in enumerate(text.splitlines(), start=1):
                    if compiled_regex.search(line):
                        clean_line = line.strip()
                        if len(clean_line) > 160:
                            clean_line = clean_line[:160] + "..."
                        matches.append(f"{rel_p}:{line_no}: {clean_line}")
                        if len(matches) >= max_results:
                            break

                if len(matches) >= max_results:
                    break

            if len(matches) >= max_results:
                break

        if not matches:
            return f"No matches found for pattern '{pattern}'."

        result_text = f"Found {len(matches)} match(es) for '{pattern}':\n\n" + "\n".join(matches)
        if len(matches) >= max_results:
            result_text += f"\n\n[Note: Search capped at {max_results} results]"

        return result_text

    return search_code_handler


def create_run_python_handler(context: EngineeringToolContext) -> ToolHandler:
    """Create run_python handler executing Python scripts in an isolated subprocess."""

    async def run_python_handler(arguments: dict[str, Any], state: AgentState) -> str:
        code = arguments.get("code", "")
        if not code or not code.strip():
            return "Error: 'code' argument cannot be empty."

        timeout_sec = min(float(arguments.get("timeout_seconds", 10.0)), 30.0)
        start_time = time.time()

        # Build clean environment with PYTHONPATH pointing to repository root
        sub_env = os.environ.copy()
        existing_pythonpath = sub_env.get("PYTHONPATH", "")
        repo_str = str(context.repo_root.resolve())
        sub_env["PYTHONPATH"] = f"{repo_str}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else repo_str
        sub_env["PYTHONUNBUFFERED"] = "1"

        try:
            process = await asyncio.create_subprocess_exec(
                context.python_executable,
                "-c",
                code,
                cwd=str(context.repo_root),
                env=sub_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_sec,
                )
            except asyncio.TimeoutError:
                try:
                    process.kill()
                    await process.wait()
                except OSError:
                    pass
                return f"Execution Timeout Error: Script exceeded {timeout_sec:.1f}s timeout limit."

        except OSError as exc:
            return f"Subprocess Error: Failed to execute Python: {exc}"

        duration = time.time() - start_time
        stdout_str = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()
        exit_code = process.returncode

        output_parts: list[str] = [
            f"Execution Finished in {duration:.2f}s (Exit Code: {exit_code})"
        ]

        if stdout_str:
            output_parts.append(f"--- STDOUT ---\n{stdout_str}")
        if stderr_str:
            output_parts.append(f"--- STDERR ---\n{stderr_str}")
        if not stdout_str and not stderr_str:
            output_parts.append("(No output produced)")

        full_output = "\n\n".join(output_parts)
        return truncate_output(full_output, max_chars=context.max_output_chars)

    return run_python_handler


def create_run_tests_handler(context: EngineeringToolContext) -> ToolHandler:
    """Create run_tests handler invoking pytest with safety validation and timeouts."""

    async def run_tests_handler(arguments: dict[str, Any], state: AgentState) -> str:
        test_target = arguments.get("test_target", "")
        extra_args = arguments.get("extra_args", [])
        timeout_sec = min(float(arguments.get("timeout_seconds", 30.0)), 60.0)

        # Validate test_target path safety if provided
        target_arg_list: list[str] = []
        if test_target:
            try:
                validated_target = validate_safe_path(
                    test_target,
                    repo_root=context.repo_root,
                    allowed_roots=context.allowed_roots,
                )
                target_arg_list.append(str(validated_target.relative_to(context.repo_root)))
            except PathSecurityError as error:
                return f"Security Error: Invalid test target: {error}"

        # Sanitize extra_args against command injection or dangerous flags
        if extra_args:
            if not isinstance(extra_args, list):
                return "Error: 'extra_args' must be a list of string arguments."
            try:
                sanitized_args = sanitize_pytest_args(extra_args)
            except CommandSecurityError as sec_err:
                return f"Security Error: {sec_err}"
        else:
            sanitized_args = ["-v"]

        cmd = [
            context.python_executable,
            "-m",
            "pytest",
            *sanitized_args,
            *target_arg_list,
        ]

        start_time = time.time()
        sub_env = os.environ.copy()
        existing_pythonpath = sub_env.get("PYTHONPATH", "")
        repo_str = str(context.repo_root.resolve())
        sub_env["PYTHONPATH"] = f"{repo_str}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else repo_str

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(context.repo_root),
                env=sub_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_sec,
                )
            except asyncio.TimeoutError:
                try:
                    process.kill()
                    await process.wait()
                except OSError:
                    pass
                return f"Test Execution Timeout Error: Tests exceeded {timeout_sec:.1f}s timeout limit."

        except OSError as exc:
            return f"Subprocess Error: Failed to invoke pytest: {exc}"

        duration = time.time() - start_time
        stdout_str = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()
        exit_code = process.returncode

        status_str = "PASSED" if exit_code == 0 else f"FAILED (Exit Code {exit_code})"
        output_parts: list[str] = [
            f"Pytest Run: {status_str} in {duration:.2f}s",
            f"Command: {' '.join(cmd[2:])}",
        ]

        if stdout_str:
            output_parts.append(f"--- STDOUT ---\n{stdout_str}")
        if stderr_str:
            output_parts.append(f"--- STDERR ---\n{stderr_str}")

        return truncate_output("\n\n".join(output_parts), max_chars=context.max_output_chars)

    return run_tests_handler


def create_git_diff_handler(context: EngineeringToolContext) -> ToolHandler:
    """Create git_diff handler for inspecting working tree and commit changes."""

    async def git_diff_handler(arguments: dict[str, Any], state: AgentState) -> str:
        # Check that repo_root is a git repository
        git_dir = context.repo_root / ".git"
        if not git_dir.exists():
            return f"Notice: Directory '{context.repo_root.name}' is not a git repository."

        file_path_arg = arguments.get("file_path", "")
        staged = bool(arguments.get("staged", False))
        commit = arguments.get("commit", "")

        cmd = ["git", "diff"]
        if staged:
            cmd.append("--cached")

        if commit:
            try:
                sanitized_commit = sanitize_git_ref(commit)
                cmd.append(sanitized_commit)
            except CommandSecurityError as error:
                return f"Security Error: {error}"

        if file_path_arg:
            try:
                validated_file = validate_safe_path(
                    file_path_arg,
                    repo_root=context.repo_root,
                    allowed_roots=context.allowed_roots,
                )
                cmd.extend(["--", str(validated_file.relative_to(context.repo_root))])
            except PathSecurityError as error:
                return f"Security Error: {error}"

        start_time = time.time()
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(context.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            return "Git Diff Timeout Error: git diff exceeded 15s timeout limit."
        except OSError as exc:
            return f"Subprocess Error: Failed to execute git: {exc}"

        diff_output = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr_output = stderr_bytes.decode("utf-8", errors="replace").strip()

        if process.returncode != 0:
            return f"Git Error (Exit Code {process.returncode}): {stderr_output or 'Failed to get git diff.'}"

        if not diff_output:
            scope = f" for '{file_path_arg}'" if file_path_arg else ""
            state_desc = "staged" if staged else "working tree"
            return f"No {state_desc} git changes found{scope}."

        header = f"Git Diff ({' '.join(cmd[1:])}):"
        return truncate_output(f"{header}\n\n{diff_output}", max_chars=context.max_output_chars)

    return git_diff_handler


def create_engineering_tool_registry(context: EngineeringToolContext) -> ToolRegistry:
    """Create a ToolRegistry configured with all 6 safe engineering tools and standard agent termination."""
    registry = ToolRegistry()

    registry.register("list_files", create_list_files_handler(context), LIST_FILES_SCHEMA)
    registry.register("read_file", create_read_file_handler(context), READ_FILE_SCHEMA)
    registry.register("search_code", create_search_code_handler(context), SEARCH_CODE_SCHEMA)
    registry.register("run_python", create_run_python_handler(context), RUN_PYTHON_SCHEMA)
    registry.register("run_tests", create_run_tests_handler(context), RUN_TESTS_SCHEMA)
    registry.register("git_diff", create_git_diff_handler(context), GIT_DIFF_SCHEMA)

    # If RAGService is available in context, optionally register semantic answer_query
    if context.rag_service:
        from .tools import ANSWER_QUERY_SCHEMA, create_answer_query_handler
        registry.register("answer_query", create_answer_query_handler(context.rag_service), ANSWER_QUERY_SCHEMA)

    from .tools import FINAL_ANSWER_SCHEMA, create_final_answer_handler
    registry.register("final_answer", create_final_answer_handler(), FINAL_ANSWER_SCHEMA)

    return registry
