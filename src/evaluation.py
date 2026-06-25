import json
import re
import time
from statistics import mean
from typing import List, Dict, Any, Optional

from src.schemas import (
    GaiaTask,
    AgentState,
    ExecutionTrace
)
from src.graph import GaiaAgentGraph
from src.config import CONFIG


class EvaluationRunner:
    """
    Benchmark runner for GAIA tasks.

    Responsibilities:
    1. Run benchmark tasks through graph
    2. Store execution traces
    3. Compute aggregate metrics
    4. Compare baseline vs immune system
    """

    def __init__(self, graph: GaiaAgentGraph):
        self.graph = graph
        self.traces: List[ExecutionTrace] = []

    # ============================================================
    # Utility
    # ============================================================

    def normalize_answer(self, text: Optional[str]) -> str:
        """
        Normalize answer for fair comparison.
        Handles punctuation, whitespace, case.
        """
        if not text:
            return ""

        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join(text.split())
        return text

    # ============================================================
    # Single Task Evaluation
    # ============================================================

    def run_task(self, task: GaiaTask) -> ExecutionTrace:
        """
        Execute a single task through agent graph.
        """
        state = AgentState(task=task)

        start_time = time.time()

        try:
            state = self.graph.execute(state)
            runtime = time.time() - start_time

        except Exception as e:
            runtime = time.time() - start_time

            trace = ExecutionTrace(
                task_id=task.task_id,
                final_answer=f"ERROR: {str(e)}",
                success=False
            )

            trace.execution_time = runtime
            trace.step_count = 0

            self.traces.append(trace)
            return trace

        trace = ExecutionTrace(
            task_id=task.task_id,
            final_answer=(
                state.validator_output.content
                if state.validator_output
                else None
            ),
            success=self.evaluate_answer(task, state)
        )

        # Core metrics
        trace.execution_time = runtime
        trace.step_count = state.current_step

        # Step history
        trace.steps = [
            f"Total Steps: {state.current_step}"
        ]

        # Agent outputs
        trace.agent_outputs = [
            output for output in [
                state.planner_output,
                state.retriever_output,
                state.reasoner_output,
                state.validator_output
            ] if output is not None
        ]

        # Drift history
        if state.drift_metrics is not None:
            trace.drift_history.append(
                state.drift_metrics
            )

        # Immune intervention
        trace.intervention_triggered = (
            getattr(state, "immune_intervention", False)
        )

        self.traces.append(trace)
        return trace

    # ============================================================
    # Answer Evaluation
    # ============================================================

    def evaluate_answer(
        self,
        task: GaiaTask,
        state: AgentState
    ) -> bool:
        """
        Flexible answer matching.
        """
        if not task.expected_answer:
            return False

        if not state.validator_output:
            return False

        predicted = self.normalize_answer(
            state.validator_output.content
        )

        expected = self.normalize_answer(
            task.expected_answer
        )

        return expected in predicted

    # ============================================================
    # Benchmark Run
    # ============================================================

    def run_benchmark(
        self,
        tasks: List[GaiaTask]
    ) -> List[ExecutionTrace]:

        traces = []

        for i, task in enumerate(tasks):
            print(
                f"[Benchmark] Running task "
                f"{i+1}/{len(tasks)}"
            )

            trace = self.run_task(task)
            traces.append(trace)

        return traces

    # ============================================================
    # Aggregate Metrics
    # ============================================================

    def compute_metrics(self) -> Dict[str, Any]:
        if not self.traces:
            return {}

        total_tasks = len(self.traces)

        success_rate = (
            sum(
                1 for trace in self.traces
                if trace.success
            ) / total_tasks
        )

        avg_steps = mean([
            trace.step_count
            for trace in self.traces
        ])

        avg_latency = mean([
            trace.execution_time
            for trace in self.traces
        ])

        drift_scores = []

        for trace in self.traces:
            for drift in trace.drift_history:
                if drift is not None:
                    drift_scores.append(
                        drift.overall_drift_score
                    )

        avg_drift = (
            mean(drift_scores)
            if drift_scores else 0
        )

        intervention_rate = (
            sum(
                1 for trace in self.traces
                if getattr(
                    trace,
                    "intervention_triggered",
                    False
                )
            ) / total_tasks
        )

        return {
            "total_tasks": total_tasks,
            "success_rate": success_rate,
            "average_steps": avg_steps,
            "average_latency_seconds": avg_latency,
            "average_drift_score": avg_drift,
            "immune_intervention_rate": intervention_rate
        }

    # ============================================================
    # Compare Two Systems
    # ============================================================

    def compare_systems(
        self,
        baseline_metrics: Dict[str, Any],
        immune_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare baseline vs immune architecture.
        """

        return {
            "success_gain":
                immune_metrics["success_rate"]
                - baseline_metrics["success_rate"],

            "drift_reduction":
                baseline_metrics["average_drift_score"]
                - immune_metrics["average_drift_score"],

            "step_difference":
                immune_metrics["average_steps"]
                - baseline_metrics["average_steps"],

            "latency_overhead":
                immune_metrics["average_latency_seconds"]
                - baseline_metrics["average_latency_seconds"]
        }

    # ============================================================
    # Save Results
    # ============================================================

    def save_results(
        self,
        filename: str = "results.json"
    ):
        metrics = self.compute_metrics()

        output_path = (
            CONFIG.paths.logs_dir / filename
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                metrics,
                f,
                indent=4
            )

        print(
            f"Results saved to {output_path}"
        )