"""Planning utilities for a minimal AI software engineering agent."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class AgentPlan:
    """Structured plan derived from a user request."""

    objective: str
    steps: list[str]


def extract_keywords(text: str) -> list[str]:
    """Return normalized keywords from a user request."""
    if not text or not text.strip():
        raise ValueError("Task text cannot be empty.")

    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
    return [word for word in words if word not in stopwords and len(word) > 2]


def build_plan(task: str) -> AgentPlan:
    """Turn a task description into a simple execution plan."""
    if not task or not task.strip():
        raise ValueError("Task text cannot be empty.")

    keywords = extract_keywords(task)
    objective = task.strip()
    steps = [
        "Clarify the objective and constraints.",
        "Break the work into smaller deliverables and dependencies.",
    ]

    if "api" in keywords:
        steps.append("Design the API contract and data model before implementation.")
    if any(item in keywords for item in ("test", "tests", "testing")):
        steps.append("Add or update the validation tests for the expected behavior.")
    if "deploy" in keywords or "release" in keywords:
        steps.append("Prepare deployment steps and rollback considerations.")
    if "bug" in keywords or "fix" in keywords:
        steps.append("Reproduce the issue and validate the root cause before changing code.")

    steps.append("Implement the core changes and check the result against the task.")
    steps.append("Review the final output and summarize the outcome for follow-up work.")

    return AgentPlan(objective=objective, steps=steps)


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:])
    plan = build_plan(prompt)
    print(plan.objective)
    for index, step in enumerate(plan.steps, start=1):
        print(f"{index}. {step}")
