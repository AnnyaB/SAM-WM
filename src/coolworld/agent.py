from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .planner import CandidateOutcome, CoolingDecision, choose_cooling_action


class CoolingTool(Protocol):
    name: str

    def run(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class AgentStep:
    tool: str
    purpose: str
    result_summary: str


@dataclass(frozen=True, slots=True)
class AgentRun:
    status: str
    steps: tuple[AgentStep, ...]
    decision: CoolingDecision | None
    payload: dict[str, Any] = field(default_factory=dict)


class CoolingAgent:
    """Bounded tool-using agent for the cooling workflow.

    The core planner is deterministic and auditable. An optional small language
    model may translate user text into this structured request or verbalize the
    final result, but it does not choose physical actions or override evidence.
    """

    def __init__(self, tools: dict[str, CoolingTool]) -> None:
        self.tools = tools

    def run_structured(
        self,
        *,
        heatmap_args: dict[str, Any],
        candidate_args: dict[str, Any] | None = None,
        budget: float | None = None,
    ) -> AgentRun:
        steps: list[AgentStep] = []
        heat_tool = self.tools.get("heatmap")
        if heat_tool is None:
            return AgentRun("MISSING_HEATMAP_TOOL", tuple(steps), None)
        heat = heat_tool.run(**heatmap_args)
        steps.append(AgentStep("heatmap", "measure current thermal field", "completed"))

        if candidate_args is None:
            return AgentRun("OBSERVATION_COMPLETE", tuple(steps), None, {"heatmap": heat})

        candidate_tool = self.tools.get("counterfactual")
        if candidate_tool is None:
            return AgentRun("MODEL_NOT_READY", tuple(steps), None, {"heatmap": heat})
        outcomes_raw = candidate_tool.run(observation=heat, **candidate_args)
        outcomes = tuple(CandidateOutcome(**x) for x in outcomes_raw)
        steps.append(
            AgentStep(
                "counterfactual",
                "evaluate candidate cooling interventions",
                f"{len(outcomes)} candidates",
            )
        )
        decision = choose_cooling_action(outcomes, budget=budget)
        steps.append(
            AgentStep("planner", "rank cooling actions under uncertainty", decision.status)
        )
        return AgentRun(
            decision.status, tuple(steps), decision, {"heatmap": heat, "outcomes": outcomes_raw}
        )
