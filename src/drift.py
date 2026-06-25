import math
from collections import Counter
from typing import List

from src.schemas import AgentState, DriftMetrics
from src.memory import MemoryManager


class DriftMonitor:
    """
    Computes drift metrics from agent state.
    """

    def __init__(self, memory: MemoryManager):
        self.memory = memory

    # ============================================================
    # Entropy
    # ============================================================

    def compute_entropy(self, text: str) -> float:
        if not text:
            return 0.0

        tokens = text.lower().split()

        if len(tokens) == 0:
            return 0.0

        freq = Counter(tokens)
        total = len(tokens)

        entropy = 0.0

        for count in freq.values():
            p = count / total
            entropy -= p * math.log2(p)

        return entropy

    # ============================================================
    # Approx Perplexity
    # ============================================================

    def compute_perplexity(self, text: str) -> float:
        if not text:
            return 0.0

        tokens = text.split()

        if len(tokens) == 0:
            return 0.0

        unique_ratio = len(set(tokens)) / len(tokens)

        if unique_ratio == 0:
            return 100.0

        return 1 / unique_ratio

    # ============================================================
    # PID Redundancy Proxy
    # ============================================================

    def compute_pid_redundancy(self, outputs: List[str]) -> float:
        """
        Approximate redundancy using word overlap.
        """

        if len(outputs) < 2:
            return 1.0

        token_sets = [
            set(text.lower().split())
            for text in outputs if text
        ]

        if not token_sets:
            return 0.0

        overlap = token_sets[0]

        for s in token_sets[1:]:
            overlap = overlap.intersection(s)

        union = set.union(*token_sets)

        if len(union) == 0:
            return 0.0

        redundancy = len(overlap) / len(union)
        return redundancy

    # ============================================================
    # Tool Oscillation
    # ============================================================

    def compute_tool_oscillation(self, tool_calls: List[str]) -> float:
        """
        Measures repeated tool usage loops.
        """

        if len(tool_calls) <= 1:
            return 0.0

        repeats = 0

        for i in range(1, len(tool_calls)):
            if tool_calls[i] == tool_calls[i - 1]:
                repeats += 1

        return repeats / (len(tool_calls) - 1)

    # ============================================================
    # Context Pollution
    # ============================================================

    def compute_context_pollution(self) -> float:
        return self.memory.context_pollution_ratio()

    # ============================================================
    # Composite Score
    # ============================================================

    def compute_overall_score(
        self,
        entropy: float,
        perplexity: float,
        redundancy: float,
        tool_oscillation: float,
        pollution: float
    ) -> float:
        """
        Weighted drift score.
        Higher = more drift.
        """

        entropy_score = min(entropy / 6.0, 1.0)
        perplexity_score = min(perplexity / 10.0, 1.0)
        redundancy_score = 1 - redundancy

        score = (
            0.25 * entropy_score +
            0.20 * perplexity_score +
            0.25 * redundancy_score +
            0.15 * tool_oscillation +
            0.15 * pollution
        )

        return score

    # ============================================================
    # Main Compute
    # ============================================================

    def compute(self, state: AgentState) -> DriftMetrics:
        outputs = []

        if state.planner_output:
            outputs.append(state.planner_output.content)

        if state.retriever_output:
            outputs.append(state.retriever_output.content)

        if state.reasoner_output:
            outputs.append(state.reasoner_output.content)

        combined_text = " ".join(outputs)

        entropy = self.compute_entropy(combined_text)
        perplexity = self.compute_perplexity(combined_text)
        redundancy = self.compute_pid_redundancy(outputs)

        tool_calls = []

        if state.retriever_output:
            tool_calls.extend(state.retriever_output.tool_calls)

        tool_oscillation = self.compute_tool_oscillation(tool_calls)

        pollution = self.compute_context_pollution()

        overall = self.compute_overall_score(
            entropy,
            perplexity,
            redundancy,
            tool_oscillation,
            pollution
        )

        return DriftMetrics(
            entropy=entropy,
            perplexity=perplexity,
            pid_redundancy=redundancy,
            tool_oscillation_score=tool_oscillation,
            context_pollution_ratio=pollution,
            overall_drift_score=overall
        )