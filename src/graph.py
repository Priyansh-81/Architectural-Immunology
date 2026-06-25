from typing import Optional

from src.schemas import AgentState
from src.agents import (
    PlannerAgent,
    RetrieverAgent,
    ReasonerAgent,
    ValidatorAgent
)


class GaiaAgentGraph:
    """
    Orchestrates multi-agent execution.
    """

    def __init__(
        self,
        planner: PlannerAgent,
        retriever: RetrieverAgent,
        reasoner: ReasonerAgent,
        validator: ValidatorAgent,
        drift_monitor=None,
        immune_system=None
    ):
        self.planner = planner
        self.retriever = retriever
        self.reasoner = reasoner
        self.validator = validator

        self.drift_monitor = drift_monitor
        self.immune_system = immune_system

    # ========================================================
    # Nodes
    # ========================================================

    def planner_node(self, state: AgentState):
        state.planner_output = self.planner.run(state)
        state.current_step += 1
        return state

    def retriever_node(self, state: AgentState):
        state.retriever_output = self.retriever.run(state)
        state.current_step += 1
        return state

    def reasoner_node(self, state: AgentState):
        state.reasoner_output = self.reasoner.run(state)
        state.current_step += 1
        return state

    def validator_node(self, state: AgentState):
        state.validator_output = self.validator.run(state)
        state.current_step += 1
        return state

    def drift_node(self, state: AgentState):
        """
        Runs drift detection if enabled.
        """
        if self.drift_monitor:
            metrics = self.drift_monitor.compute(state)
            state.drift_metrics = metrics

            if metrics.overall_drift_score > 0.7:
                state.intervention_triggered = True

        return state

    def immune_node(self, state: AgentState):
        """
        Run immune intervention if drift detected.
        """
        if self.immune_system and state.intervention_triggered:
            action = self.immune_system.intervene(
                state.drift_metrics
            )

            if action != "none":
                print(f"[IMMUNE] Action taken: {action}")

        return state

    # ========================================================
    # Graph Execution
    # ========================================================

    def execute(self, state: AgentState):
        """
        Main graph execution.
        """

        # Node 1
        state = self.planner_node(state)

        # Node 2
        state = self.retriever_node(state)

        # Drift checkpoint
        state = self.drift_node(state)

        # Conditional branch
        if state.intervention_triggered:
            state = self.immune_node(state)

        # Node 3
        state = self.reasoner_node(state)

        # Drift checkpoint again
        state = self.drift_node(state)

        if state.intervention_triggered:
            state = self.immune_node(state)

        # Final validation
        state = self.validator_node(state)

        return state