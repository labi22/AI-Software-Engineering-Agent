"""Tests for the AI software engineering agent scaffold."""

import pytest

from ai_software_engineering_agent.agent import AgentPlan, build_plan, extract_keywords


def test_extract_keywords_filters_common_words():
    keywords = extract_keywords("Build a REST API with tests for the login flow")
    assert "build" in keywords
    assert "rest" in keywords
    assert "api" in keywords
    assert "tests" in keywords
    assert "login" in keywords
    assert "flow" in keywords
    assert "with" not in keywords


def test_build_plan_generates_steps_for_api_work():
    plan = build_plan("Build a REST API with tests for the login flow")

    assert isinstance(plan, AgentPlan)
    assert plan.objective.startswith("Build a REST API")
    assert len(plan.steps) >= 5
    assert any("API contract" in step for step in plan.steps)
    assert any("validation tests" in step for step in plan.steps)


def test_build_plan_rejects_empty_task():
    with pytest.raises(ValueError, match="Task text cannot be empty"):
        build_plan("   ")
