"""ReAct-style agent loop with LLM-driven tool selection and trace logging."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

from .agent_state import AgentResult, AgentState, AgentStatus, Observation, ToolCall
from .config import Settings
from .llm import LLMClient, LLMConfigurationError, LLMRequest
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy planning utilities (backward compatible with Day 1-2)
# ---------------------------------------------------------------------------


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
        "a", "an", "and", "as", "at", "be", "by", "for", "from",
        "in", "into", "is", "it", "of", "on", "or", "that", "the",
        "this", "to", "with",
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


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

# Regex to find <tool_call>...</tool_call> blocks in LLM responses
_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


class AgentError(RuntimeError):
    """Error raised during agent execution."""


class Agent:
    """ReAct-style agent that reasons, selects tools, and iterates until done.

    The loop follows the pattern:
        Think → Act → Observe → Think → ... → Final Answer

    The agent uses the LLM to decide which tool to call (or whether to answer
    directly) at each step. Tool calls are communicated via ``<tool_call>``
    XML tags in the LLM response text, keeping the agent portable across
    LLM providers.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        settings: Settings,
        max_steps: int | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.settings = settings
        self.max_steps = max_steps or settings.agent_max_steps

    async def run(self, task: str) -> AgentResult:
        """Execute the ReAct loop for a user task and return the result with full trace."""
        if not task or not task.strip():
            raise AgentError("Task text cannot be empty.")

        state = AgentState(task=task, max_steps=self.max_steps)
        system_prompt = self._build_system_prompt()

        logger.info("Agent started: task=%r, max_steps=%d", task, self.max_steps)

        while not state.is_done():
            if not state.has_budget():
                logger.warning("Agent exceeded step limit (%d steps)", self.max_steps)
                state.fail(f"Step limit exceeded ({self.max_steps} steps). The agent could not complete the task within the allowed budget.")
                break

            # 1. THINK — send conversation to LLM
            conversation = self._format_conversation(state)
            try:
                llm_response = await self.llm_client.generate(
                    LLMRequest(
                        prompt=conversation,
                        system_instruction=system_prompt,
                    )
                )
            except LLMConfigurationError:
                raise
            except Exception as exc:
                logger.error("LLM call failed at step %d: %s", state.current_step + 1, exc)
                state.fail(f"LLM generation error: {exc}")
                break

            response_text = llm_response.text.strip()

            # 2. PARSE — extract tool call or final answer
            parsed = self._parse_llm_response(response_text)

            if isinstance(parsed, ToolCall):
                # Record the thinking and action
                state.add_thinking_step(response_text)
                state.add_action_step(parsed)
                logger.info(
                    "Step %d: calling tool=%s args=%s",
                    state.current_step, parsed.tool_name, parsed.arguments,
                )

                # 3. ACT — execute the tool
                observation = await self.tool_registry.execute(parsed, state)

                # 4. OBSERVE — record the result only if the tool did not already
                # terminate the loop (e.g. final_answer calls state.finish() during
                # execute, and add_observation_step would overwrite the FINISHED status).
                already_done = state.is_done()
                if not already_done:
                    state.add_observation_step(observation)
                logger.info(
                    "Step %d: observation from %s (error=%s, %dms)",
                    state.current_step, parsed.tool_name, observation.is_error, observation.duration_ms,
                )

                if already_done or state.is_done():
                    break

            else:
                # LLM produced a direct answer without a tool call
                state.add_thinking_step(response_text)
                state.finish(parsed)
                logger.info("Agent finished with direct answer at step %d", state.current_step)

        answer = state.final_answer or "The agent could not produce an answer."

        result = AgentResult(
            task=task,
            answer=answer,
            steps=state.steps,
            total_steps=state.current_step,
            status=state.status.value,
            duration_ms=state.elapsed_ms(),
            citations=state.citations,
        )

        logger.info(
            "Agent completed: status=%s, steps=%d, duration=%.0fms",
            result.status, result.total_steps, result.duration_ms,
        )
        return result

    def _build_system_prompt(self) -> str:
        """Construct the system prompt with tool schemas and ReAct instructions."""
        tools_block = self.tool_registry.format_for_prompt()

        return (
            "You are an expert AI software engineering agent. You solve tasks by reasoning "
            "step-by-step and using the available tools to search, read, and understand code.\n\n"
            "## How to use tools\n\n"
            "When you want to use a tool, wrap the call in XML tags like this:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
            "</tool_call>\n\n"
            "After each tool call, you will receive the tool's output as an observation. "
            "Use the observation to inform your next reasoning step.\n\n"
            "## When to stop\n\n"
            "When you have gathered enough information to answer the user's task completely, "
            "call the `final_answer` tool with your complete response:\n"
            "<tool_call>\n"
            '{"name": "final_answer", "arguments": {"answer": "Your complete answer here..."}}\n'
            "</tool_call>\n\n"
            "If you can answer the task directly from your own knowledge without needing "
            "any tools, you may respond with a plain text answer (no tool_call tags).\n\n"
            "## Important rules\n\n"
            "1. Always reason about what you know and what you need before calling a tool.\n"
            "2. Use `search_code` to find relevant code snippets in the codebase.\n"
            "3. Use `answer_query` when you want a complete cited answer generated from retrieved code.\n"
            "4. Always call `final_answer` when you are done — do not just stop.\n"
            "5. Cite specific files and line numbers when referencing code.\n"
            "6. If a tool returns an error, try a different approach or query.\n\n"
            f"## {tools_block}\n"
        )

    def _format_conversation(self, state: AgentState) -> str:
        """Format the agent's task and step history into a prompt for the LLM."""
        parts: list[str] = [f"Task: {state.task}"]

        for step in state.steps:
            if step.status == AgentStatus.THINKING:
                parts.append(f"\n[Thought]\n{step.reasoning}")
            elif step.status == AgentStatus.ACTING and step.tool_call:
                tc = step.tool_call
                parts.append(
                    f"\n[Action] Called tool: {tc.tool_name}\n"
                    f"Arguments: {json.dumps(tc.arguments)}"
                )
            elif step.status == AgentStatus.OBSERVING and step.observation:
                obs = step.observation
                error_tag = " (ERROR)" if obs.is_error else ""
                parts.append(f"\n[Observation{error_tag}]\n{obs.content}")

        parts.append("\n[Your next reasoning step]")
        return "\n".join(parts)

    def _parse_llm_response(self, text: str) -> ToolCall | str:
        """Parse the LLM response to extract a tool call or treat it as a direct answer.

        Returns a ``ToolCall`` if the response contains ``<tool_call>`` tags,
        otherwise returns the raw text as a direct answer string.
        """
        match = _TOOL_CALL_PATTERN.search(text)
        if not match:
            return text

        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning("Failed to parse tool_call JSON: %s", match.group(1))
            return text

        tool_name = payload.get("name", "")
        arguments = payload.get("arguments", {})

        if not tool_name:
            return text

        return ToolCall(tool_name=tool_name, arguments=arguments)


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:])
    plan = build_plan(prompt)
    print(plan.objective)
    for index, step in enumerate(plan.steps, start=1):
        print(f"{index}. {step}")
