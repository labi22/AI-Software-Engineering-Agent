"""Agent state machine, step records, and result container for the ReAct loop."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .models import Citation


class AgentStatus(str, Enum):
    """Lifecycle phases of the agent loop."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"


@dataclass(frozen=True)
class ToolCall:
    """A structured record of the agent deciding to invoke a tool."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(frozen=True)
class Observation:
    """The result returned by a tool execution."""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    duration_ms: float = 0.0


@dataclass
class AgentStep:
    """A single step in the agent trace: thinking, acting, or observing."""

    step_number: int
    status: AgentStatus
    reasoning: str | None = None
    tool_call: ToolCall | None = None
    observation: Observation | None = None
    timestamp: float = field(default_factory=time.time)


class AgentState:
    """Mutable container tracking conversation history, tool results, and step counter."""

    def __init__(self, task: str, max_steps: int = 10) -> None:
        self.task = task
        self.max_steps = max_steps
        self.steps: list[AgentStep] = []
        self.status: AgentStatus = AgentStatus.IDLE
        self.current_step: int = 0
        self.final_answer: str | None = None
        self.citations: list[Citation] = []
        self._start_time: float = time.time()

    def add_thinking_step(self, reasoning: str) -> AgentStep:
        """Record a reasoning / thinking phase."""
        self.current_step += 1
        self.status = AgentStatus.THINKING
        step = AgentStep(
            step_number=self.current_step,
            status=AgentStatus.THINKING,
            reasoning=reasoning,
        )
        self.steps.append(step)
        return step

    def add_action_step(self, tool_call: ToolCall) -> AgentStep:
        """Record the agent deciding to invoke a tool."""
        self.status = AgentStatus.ACTING
        step = AgentStep(
            step_number=self.current_step,
            status=AgentStatus.ACTING,
            tool_call=tool_call,
        )
        self.steps.append(step)
        return step

    def add_observation_step(self, observation: Observation) -> AgentStep:
        """Record the result returned by a tool."""
        self.status = AgentStatus.OBSERVING
        step = AgentStep(
            step_number=self.current_step,
            status=AgentStatus.OBSERVING,
            observation=observation,
        )
        self.steps.append(step)
        return step

    def finish(self, answer: str, citations: list[Citation] | None = None) -> None:
        """Mark the agent as finished with a final answer."""
        self.status = AgentStatus.FINISHED
        self.final_answer = answer
        self.citations = citations or []

    def fail(self, reason: str) -> None:
        """Mark the agent as failed with an error reason."""
        self.status = AgentStatus.ERROR
        self.final_answer = f"Agent failed: {reason}"

    def is_done(self) -> bool:
        """Check if the agent has finished or errored."""
        return self.status in (AgentStatus.FINISHED, AgentStatus.ERROR)

    def has_budget(self) -> bool:
        """Check if the agent has remaining step budget."""
        return self.current_step < self.max_steps

    def elapsed_ms(self) -> float:
        """Return elapsed time since agent start in milliseconds."""
        return (time.time() - self._start_time) * 1000


@dataclass(frozen=True)
class AgentResult:
    """Complete result of an agent run including the answer and full trace."""

    task: str
    answer: str
    steps: list[AgentStep]
    total_steps: int
    status: str
    duration_ms: float
    citations: list[Citation] = field(default_factory=list)
