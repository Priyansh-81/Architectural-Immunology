from typing import Optional

from src.schemas import DriftMetrics
from src.memory import MemoryManager


class ImmuneSystem:
    """
    Corrective intervention system for drift mitigation.
    """

    def __init__(self, memory: MemoryManager):
        self.memory = memory

    # ============================================================
    # Memory Pruning
    # ============================================================

    def prune_memory(self) -> str:
        """
        Remove low-value memory chunks.
        """
        self.memory.prune_memory(
            importance_threshold=0.5
        )
        return "memory_pruning"

    # ============================================================
    # Behavioral Anchoring
    # ============================================================

    def behavioral_anchor(self) -> str:
        """
        Reinforce stable task behavior.
        """
        anchor_prompt = (
            "Re-anchor to original task. "
            "Ignore irrelevant context. "
            "Follow initial objective strictly."
        )

        self.memory.add_to_short_term(
            content=anchor_prompt,
            source="immune_anchor",
            importance=1.0
        )

        return "behavioral_anchor"

    # ============================================================
    # Context Reset
    # ============================================================

    def context_reset(self) -> str:
        """
        Hard reset short-term memory.
        """
        important_chunks = []

        for chunk in self.memory.stm.chunks:
            if chunk.importance >= 0.8:
                important_chunks.append(chunk)

        self.memory.hard_reset()

        for chunk in important_chunks:
            self.memory.stm.add(chunk)

        return "context_reset"

    # ============================================================
    # Agent Replacement
    # ============================================================

    def replace_agent(self) -> str:
        """
        Placeholder for agent replacement logic.
        """
        return "agent_replacement"

    # ============================================================
    # Intervention Policy
    # ============================================================

    def intervene(self, metrics: DriftMetrics) -> str:
        """
        Decide corrective action.
        """

        drift_score = metrics.overall_drift_score

        if drift_score < 0.5:
            return "none"

        elif drift_score < 0.7:
            return self.prune_memory()

        elif drift_score < 0.85:
            return self.behavioral_anchor()

        elif drift_score < 0.95:
            return self.context_reset()

        else:
            return self.replace_agent()