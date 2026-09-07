"""Security guardrails, canonical path confinement, and execution sandboxing."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence


class PathSecurityError(PermissionError):
    """Raised when an operation attempts to access paths outside allowed boundaries."""


class ExecutionTimeoutError(TimeoutError):
    """Raised when an execution subprocess exceeds its allotted timeout."""


class CommandSecurityError(ValueError):
    """Raised when an unauthorized or potentially dangerous argument is detected."""


# Whitelisted flags for test execution runner (pytest)
ALLOWED_PYTEST_FLAGS = {
    "-v",
    "--verbose",
    "-q",
    "--quiet",
    "-s",
    "--capture=no",
    "-x",
    "--exitfirst",
    "--tb=short",
    "--tb=line",
    "--tb=no",
    "--tb=auto",
    "--disable-warnings",
    "--show-capture=no",
    "-l",
    "--showlocals",
}

# Whitelisted option prefixes that require arguments (e.g. -k <pattern>, -m <mark>)
ALLOWED_PYTEST_OPTION_PREFIXES = {"-k", "-m", "--maxfail"}

GIT_REF_PATTERN = re.compile(r"^[a-zA-Z0-9_\.\-\~^/]+$")


def validate_safe_path(
    target_path: Path | str,
    repo_root: Path | str,
    allowed_roots: Sequence[Path] = (),
) -> Path:
    """Validate and resolve target path ensuring it remains strictly inside the repository boundary."""
    root = Path(repo_root).expanduser().resolve()
    if not root.exists():
        raise PathSecurityError(f"Repository root does not exist: {root}")
    if not root.is_dir():
        raise PathSecurityError(f"Repository root is not a directory: {root}")

    # Check root against global allowed roots if specified
    if allowed_roots:
        resolved_allowed = [p.expanduser().resolve() for p in allowed_roots]
        is_root_allowed = any(
            root == allowed or allowed in root.parents
            for allowed in resolved_allowed
        )
        if not is_root_allowed:
            raise PathSecurityError(
                f"Repository root '{root}' is outside ALLOWED_REPOSITORY_ROOTS."
            )

    target = Path(target_path).expanduser()
    if not target.is_absolute():
        resolved_target = (root / target).resolve()
    else:
        resolved_target = target.resolve()

    # Verify that resolved target is inside root or equals root
    try:
        resolved_target.relative_to(root)
    except ValueError:
        raise PathSecurityError(
            f"Path '{target_path}' resolves to '{resolved_target}', which escapes repository boundary '{root}'."
        )

    # If target is a symlink, verify that the real destination is also inside root
    if target.is_symlink():
        real_dest = resolved_target.resolve()
        try:
            real_dest.relative_to(root)
        except ValueError:
            raise PathSecurityError(
                f"Symlink target '{real_dest}' escapes repository boundary '{root}'."
            )

    return resolved_target


def truncate_output(text: str, max_chars: int = 8000) -> str:
    """Truncate text to max_chars and append a helpful notice if exceeded."""
    if len(text) <= max_chars:
        return text

    truncated_count = len(text) - max_chars
    return text[:max_chars] + f"\n\n[... Truncated {truncated_count} characters to preserve context ...]"


def sanitize_git_ref(ref: str) -> str:
    """Ensure git commit hash, branch, or tag name contains no dangerous shell characters."""
    stripped = ref.strip()
    if not stripped:
        raise CommandSecurityError("Git ref cannot be empty.")

    if not GIT_REF_PATTERN.match(stripped):
        raise CommandSecurityError(
            f"Invalid or unsafe git ref '{stripped}'. Only alphanumeric characters and .-_~^/ are permitted."
        )

    if stripped.startswith("-"):
        raise CommandSecurityError(f"Git ref cannot begin with a dash: '{stripped}'.")

    return stripped


def sanitize_pytest_args(args: Sequence[str]) -> list[str]:
    """Filter and sanitize pytest arguments to only permit safe whitelisted options."""
    sanitized: list[str] = []
    idx = 0
    while idx < len(args):
        raw_arg = args[idx].strip()
        if not raw_arg:
            idx += 1
            continue

        # Check direct flag match
        if raw_arg in ALLOWED_PYTEST_FLAGS:
            sanitized.append(raw_arg)
            idx += 1
            continue

        # Check option with value (e.g. -k pattern or -k=pattern)
        matched_prefix = False
        for prefix in ALLOWED_PYTEST_OPTION_PREFIXES:
            if raw_arg == prefix and idx + 1 < len(args):
                val = args[idx + 1].strip()
                # Basic safety check on pattern argument
                if any(c in val for c in (";", "&", "|", "`", "$", "(", ")", "\n")):
                    raise CommandSecurityError(f"Dangerous characters detected in pytest argument value for '{prefix}'.")
                sanitized.extend([prefix, val])
                idx += 2
                matched_prefix = True
                break
            elif raw_arg.startswith(f"{prefix}="):
                val = raw_arg[len(prefix) + 1 :].strip()
                if any(c in val for c in (";", "&", "|", "`", "$", "(", ")", "\n")):
                    raise CommandSecurityError(f"Dangerous characters detected in pytest argument value for '{prefix}'.")
                sanitized.append(raw_arg)
                idx += 1
                matched_prefix = True
                break

        if not matched_prefix:
            # If not whitelisted flag, reject rather than executing arbitrary flags
            raise CommandSecurityError(
                f"Unauthorized pytest argument '{raw_arg}'. Only standard inspection flags ({', '.join(sorted(ALLOWED_PYTEST_FLAGS | ALLOWED_PYTEST_OPTION_PREFIXES))}) are permitted."
            )

    return sanitized
